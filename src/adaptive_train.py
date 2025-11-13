import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from config_fftchain import get_args
from utils import replace_linear_with_fftchain, save_checkpoint, load_checkpoint
import os
import gc
import glob
from tqdm import tqdm
import argparse

def get_adaptive_args():
    parser = argparse.ArgumentParser(description='FFTChain Adaptive Layer-wise Training')
    
    parser.add_argument('--model_path', type=str, 
                       default='/home/zx/Proj/llama-2-13b-chat-hf_debug_FFTChain/model')
    parser.add_argument('--checkpoint_dir', type=str, default='./adaptive_checkpoints')
    parser.add_argument('--device', type=str, default='cuda:2')
    
    parser.add_argument('--layer_start', type=int, default=0)
    parser.add_argument('--layer_end', type=int, default=39)
    parser.add_argument('--target_matrix', type=str, default='down_proj',
                       choices=['down_proj', 'up_proj', 'gate_proj'])
    
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_epochs', type=int, default=15)
    
    parser.add_argument('--loss_threshold_abs', type=float, default=2.0)
    parser.add_argument('--loss_threshold_relative', type=float, default=4.0)
    
    return parser.parse_args()

def train_single_layer(model, layer_idx, all_replaced_layers, args, K, block_size):
    distill_data = torch.load(os.path.join('./checkpoints', 'distill_data.pt'))
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr*0.1
    )
    
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    accumulation_steps = 8
    
    best_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    
    epoch_pbar = tqdm(
        range(args.epochs), 
        desc=f"L{layer_idx} K{K} B{block_size}", 
        ncols=100,
        position=0,
        leave=True
    )
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        total_hidden_loss = 0
        total_logit_loss = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(distill_data):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            target_hidden_states = [h.to(args.device) for h in batch['target_hidden_states']]
            target_logits = batch['logits'].to(args.device)
            
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            hidden_loss = 0
            for i, replaced_layer in enumerate(all_replaced_layers):
                pred_hidden = outputs.hidden_states[replaced_layer + 1]
                target_hidden = target_hidden_states[i]
                hidden_loss += mse_loss(pred_hidden.float(), target_hidden.float())
            
            pred_logits = outputs.logits
            logit_loss = kl_loss(
                torch.log_softmax(pred_logits.float(), dim=-1),
                torch.softmax(target_logits.float(), dim=-1)
            )
            
            loss = (0.2 * hidden_loss + 1.2 * logit_loss) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            total_hidden_loss += hidden_loss.item()
            total_logit_loss += logit_loss.item()
            
            del outputs, pred_hidden, pred_logits, hidden_loss, logit_loss, loss
            del input_ids, attention_mask, target_hidden_states, target_logits
            torch.cuda.empty_cache()
        
        scheduler.step()
        
        avg_loss = total_loss / len(distill_data)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            no_improve_count = 0
            status = "↓NEW"
        else:
            no_improve_count += 1
            status = f"×{no_improve_count}"
        
        epoch_pbar.set_postfix({
            'L': f'{avg_loss:.1f}',
            'Best': f'{best_loss:.1f}@E{best_epoch}',
            'S': status
        })
        
        if epoch >= args.min_epochs and no_improve_count >= args.patience:
            epoch_pbar.close()
            print(f"\n[Train] Early stop at E{epoch+1} (best: E{best_epoch}, loss: {best_loss:.2f})")
            
            del optimizer, scheduler, distill_data
            gc.collect()
            torch.cuda.empty_cache()
            
            return best_loss, epoch + 1
    
    epoch_pbar.close()
    print(f"\n[Train] Completed {args.epochs} epochs (best: E{best_epoch}, loss: {best_loss:.2f})")
    
    del optimizer, scheduler, distill_data
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_loss, args.epochs

def train_layer_with_config(args, layer_idx, all_replaced_layers, K, block_size, prev_layers_info):
    print(f"\nLoading model from {args.model_path} on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    
    replaced_modules = []
    for layer in all_replaced_layers:
        if layer == layer_idx:
            layer_K = K
            layer_block_size = block_size
        elif layer in prev_layers_info:
            layer_K = prev_layers_info[layer]['K']
            layer_block_size = prev_layers_info[layer]['block_size']
        else:
            raise ValueError(f"Layer {layer} not in prev_layers_info and not current layer")
        
        print(f"  Replacing Layer {layer} with K={layer_K}, block_size={layer_block_size}")
        modules = replace_linear_with_fftchain(
            model, [layer], args.target_matrix, 
            layer_block_size, layer_K
        )
        replaced_modules.extend(modules)
    
    print(f"[Train] Converting {len(replaced_modules)} modules to float32...")
    for module in replaced_modules:
        module.to(args.device).to(torch.float32)
        for param in module.parameters():
            param.data = param.data.to(args.device).to(torch.float32)
    
    if len(prev_layers_info) > 0:
        print(f"[Train] Loading {len(prev_layers_info)} previous layer checkpoints...")
        for prev_layer_idx, prev_config in prev_layers_info.items():
            prev_checkpoint = os.path.join(
                args.checkpoint_dir,
                f'fftchain_L{prev_layer_idx}_K{prev_config["K"]}_B{prev_config["block_size"]}_{args.target_matrix}_final.pt'
            )
            if os.path.exists(prev_checkpoint):
                print(f"    Loading Layer {prev_layer_idx} (K={prev_config['K']}, B={prev_config['block_size']})")
                checkpoint = torch.load(prev_checkpoint, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                module_idx = all_replaced_layers.index(prev_layer_idx)
                replaced_modules[module_idx].to(args.device)
            else:
                print(f"    Warning: Checkpoint not found for Layer {prev_layer_idx}")
    
    for param in model.parameters():
        param.requires_grad = False
    
    current_module = replaced_modules[-1]
    for param in current_module.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Train] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    best_loss, stopped_epoch = train_single_layer(
        model, layer_idx, all_replaced_layers, args, K, block_size
    )
    
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'fftchain_L{layer_idx}_K{K}_B{block_size}_{args.target_matrix}_final.pt'
    )
    config = {
        'layer_idx': layer_idx,
        'target_matrix': args.target_matrix,
        'block_size': block_size,
        'num_fft_matrices': K,
        'trained_epochs': stopped_epoch,
        'best_loss': best_loss
    }
    
    temp_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    
    model_cpu = model.cpu()
    save_checkpoint(model_cpu, temp_optimizer, stopped_epoch-1, checkpoint_path, config=config)
    
    del model, model_cpu, temp_optimizer, replaced_modules
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_loss

def adaptive_train_layer(args, layer_idx, layer_position, total_layers, prev_loss, prev_layers_info):
    print(f"\n{'='*80}")
    print(f"Training Layer {layer_idx} ({layer_position}/{total_layers})")
    print(f"{'='*80}")
    
    if layer_idx <= 10:
        configs_to_try = [
            (4, 1024),
            (6, 1024),
            (8, 1024),
            (8, 512),
            (12, 512),
            (16, 512),
        ]
        print(f"[Adaptive] Using early-layer configs (more capacity)")
    else:
        configs_to_try = [
            (4, 1024),
            (6, 1024),
            (6, 512),
            (8, 512),
            (8, 256)
        ]
        print(f"[Adaptive] Using standard configs")
    
    all_replaced_layers = list(range(args.layer_start, layer_idx + 1))
    
    threshold_abs = args.loss_threshold_abs
    if prev_loss is not None:
        if prev_loss <= threshold_abs:
            threshold_rel = prev_loss * 1.5
            threshold = min(threshold_abs, threshold_rel)
            print(f"[Adaptive] Threshold: min({threshold_abs:.2f}, {prev_loss:.2f}*1.5) = {threshold:.2f}")
        else:
            threshold = threshold_abs
            print(f"[Adaptive] Threshold: {threshold_abs:.2f} (prev layer loss too high)")
    else:
        threshold = threshold_abs
        print(f"[Adaptive] Threshold: {threshold_abs:.2f} (first layer)")
    
    for attempt_idx, (K, block_size) in enumerate(configs_to_try):
        print(f"\n[Adaptive] Attempt {attempt_idx + 1}/{len(configs_to_try)}: K={K}, block_size={block_size}")
        print(f"{'-'*80}")
        
        best_loss = train_layer_with_config(
            args, layer_idx, all_replaced_layers, K, block_size, prev_layers_info
        )
        
        print(f"\n[Train] Layer {layer_idx} finished: loss={best_loss:.4f} (K={K}, block_size={block_size})")
        
        if best_loss <= threshold:
            print(f"[Adaptive] Layer {layer_idx} accepted: loss={best_loss:.4f} <= {threshold:.2f}")
            if attempt_idx > 0:
                print(f"[Adaptive] (Accepted after {attempt_idx} adjustment(s))")
            
            prev_layers_info[layer_idx] = {
                'K': K,
                'block_size': block_size,
                'loss': best_loss
            }
            
            return best_loss, prev_layers_info
        else:
            print(f"[Adaptive] Layer {layer_idx} rejected: loss={best_loss:.4f} > {threshold:.2f}")
            
            if attempt_idx < len(configs_to_try) - 1:
                next_K, next_block_size = configs_to_try[attempt_idx + 1]
                print(f"[Adaptive] Adjusting to K={next_K}, block_size={next_block_size}...")
                
                old_checkpoint = os.path.join(
                    args.checkpoint_dir,
                    f'fftchain_L{layer_idx}_K{K}_B{block_size}_{args.target_matrix}_final.pt'
                )
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
            else:
                print(f"[Adaptive] Warning: All configurations tried, accepting loss={best_loss:.4f}")
                prev_layers_info[layer_idx] = {
                    'K': K,
                    'block_size': block_size,
                    'loss': best_loss
                }
                return best_loss, prev_layers_info
    
    return best_loss, prev_layers_info

def adaptive_train_all_layers(args):
    all_layers = list(range(args.layer_start, args.layer_end + 1))
    
    print(f"Cleaning old adaptive checkpoints...")
    if os.path.exists(args.checkpoint_dir):
        old_checkpoints = glob.glob(os.path.join(args.checkpoint_dir, 'fftchain_*.pt'))
        if old_checkpoints:
            for ckpt in old_checkpoints:
                os.remove(ckpt)
            print(f"[Train] Removed {len(old_checkpoints)} old checkpoints")
    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    prev_loss = None
    prev_layers_info = {}
    
    for layer_position, layer_idx in enumerate(all_layers, start=1):
        prev_loss, prev_layers_info = adaptive_train_layer(
            args, layer_idx, layer_position, len(all_layers), prev_loss, prev_layers_info
        )
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(args.device) / 1024**3
            reserved = torch.cuda.memory_reserved(args.device) / 1024**3
            print(f"[Train] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved\n")
    
    print(f"\n{'='*80}")
    print(f"[Train] All {len(all_layers)} layers trained successfully!")
    print(f"{'='*80}")
    print(f"\nFinal Configuration Summary:")
    print(f"{'-'*80}")
    for layer_idx in sorted(prev_layers_info.keys()):
        info = prev_layers_info[layer_idx]
        print(f"Layer {layer_idx}: K={info['K']}, block_size={info['block_size']}, loss={info['loss']:.4f}")
    print(f"{'='*80}\n")

def main():
    args = get_adaptive_args()
    
    print("\n" + "="*80)
    print("FFTChain Adaptive Training Configuration")
    print("="*80)
    print(f"Model Path:           {args.model_path}")
    print(f"Device:               {args.device}")
    print(f"Layers:               {args.layer_start}-{args.layer_end}")
    print(f"Target Matrix:        {args.target_matrix}")
    print(f"Max Epochs:           {args.epochs}")
    print(f"Early Stop Patience:  {args.patience}")
    print(f"Loss Threshold (abs): {args.loss_threshold_abs}")
    print(f"Loss Threshold (rel): {args.loss_threshold_relative}x")
    print(f"Checkpoint Dir:       {args.checkpoint_dir}")
    print("="*80)
    
    distill_data_path = os.path.join('./checkpoints', 'distill_data.pt')
    if not os.path.exists(distill_data_path):
        print(f"\n[Error] Distillation data not found at {distill_data_path}")
        print(f"[Error] Please run generate_distill_data.py first!")
        return
    
    adaptive_train_all_layers(args)
    
    print("\nAdaptive training completed!")

if __name__ == '__main__':
    main()