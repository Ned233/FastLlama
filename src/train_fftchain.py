import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from config_fftchain import get_args
from utils import replace_linear_with_fftchain, save_checkpoint, load_checkpoint
import os
import gc
import glob
from tqdm import tqdm

def train_layer_by_layer(args):
    all_layers = list(range(args.layer_start, args.layer_end + 1))
    
    print(f"Cleaning old model checkpoints...")
    if os.path.exists(args.checkpoint_dir):
        old_checkpoints = glob.glob(os.path.join(args.checkpoint_dir, 'fftchain_*.pt'))
        if old_checkpoints:
            for ckpt in old_checkpoints:
                os.remove(ckpt)
            print(f"[Train] Removed {len(old_checkpoints)} old checkpoints")
    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for layer_idx, current_layer in enumerate(all_layers):
        print(f"\n{'='*80}")
        print(f"Training Layer {current_layer} ({layer_idx+1}/{len(all_layers)})")
        print(f"{'='*80}\n")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Loading model from {args.model_path} on {args.device}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map={"": args.device}
        )
        
        layers_to_replace = list(range(args.layer_start, current_layer + 1))
        print(f"Replacing {args.target_matrix} in layers {layers_to_replace}...")
        replaced_modules = replace_linear_with_fftchain(
            model, layers_to_replace, args.target_matrix, 
            args.block_size, args.num_fft_matrices
        )
        
        print(f"[Train] Converting {len(replaced_modules)} modules to float32...")
        for module in replaced_modules:
            module.to(args.device).to(torch.float32)
            for param in module.parameters():
                param.data = param.data.to(args.device).to(torch.float32)
        
        # 加载之前训练好的层（先加载到CPU，再移动到GPU）
        if current_layer > args.layer_start:
            prev_checkpoint = os.path.join(
                args.checkpoint_dir,
                f'fftchain_L{args.layer_start}-{current_layer-1}_{args.target_matrix}_final.pt'
            )
            if os.path.exists(prev_checkpoint):
                print(f"[Train] Loading previous checkpoint...")
                # 先加载到CPU
                load_checkpoint(model, None, prev_checkpoint, device='cpu')
                # 再移动到GPU
                for module in replaced_modules[:-1]:  # 除了最后一层（当前训练层）
                    module.to(args.device)
        
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 只解冻当前层
        current_module = replaced_modules[-1]
        for param in current_module.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Train] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # 训练当前层（带早停）
        best_loss, stopped_epoch = train_single_layer(
            model, current_layer, layers_to_replace, args
        )
        
        # 保存checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f'fftchain_L{args.layer_start}-{current_layer}_{args.target_matrix}_final.pt'
        )
        config = {
            'layer_start': args.layer_start,
            'layer_end': current_layer,
            'target_matrix': args.target_matrix,
            'block_size': args.block_size,
            'num_fft_matrices': args.num_fft_matrices,
            'trained_epochs': stopped_epoch,
            'best_loss': best_loss
        }
        
        # 创建临时优化器用于保存
        temp_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )
        
        # 保存前先移动到CPU（节省显存）
        model_cpu = model.cpu()
        save_checkpoint(model_cpu, temp_optimizer, stopped_epoch-1, checkpoint_path, config=config)
        print(f"\n[Train] ✓ Saved layer {current_layer} (loss: {best_loss:.4f}, epochs: {stopped_epoch})")
        
        # 彻底清理
        del model, model_cpu, temp_optimizer, replaced_modules
        gc.collect()
        torch.cuda.empty_cache()
        
        # 打印显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(args.device) / 1024**3
            reserved = torch.cuda.memory_reserved(args.device) / 1024**3
            print(f"[Train] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    print(f"\n{'='*80}")
    print(f"[Train] ✓ All {len(all_layers)} layers trained successfully!")
    print(f"[Train] Final: fftchain_L{args.layer_start}-{args.layer_end}_{args.target_matrix}_final.pt")
    print(f"{'='*80}\n")

def train_single_layer(model, layer_idx, all_replaced_layers, args):
    """训练单个层（带早停 + 优化的进度条）"""
    
    # 加载蒸馏数据
    distill_data = torch.load(os.path.join(args.checkpoint_dir, 'distill_data.pt'))
    
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
    
    # 早停参数
    best_loss = float('inf')
    best_epoch = 0
    patience = args.patience if hasattr(args, 'patience') else 10
    no_improve_count = 0
    min_epochs = args.min_epochs if hasattr(args, 'min_epochs') else 20
    
    print(f"[Train] Layer {layer_idx}: max {args.epochs} epochs, patience={patience}, min_epochs={min_epochs}\n")
    
    # 只创建epoch进度条
    epoch_pbar = tqdm(
        range(args.epochs), 
        desc=f"Layer {layer_idx}", 
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
        
        # 不显示batch进度条，只在内部循环
        for batch_idx, batch in enumerate(distill_data):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            target_hidden_states = [h.to(args.device) for h in batch['target_hidden_states']]
            target_logits = batch['logits'].to(args.device)
            
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # 计算所有已替换层的hidden loss
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
        avg_hidden = total_hidden_loss / len(distill_data)
        avg_logit = total_logit_loss / len(distill_data)
        
        # 早停逻辑
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            no_improve_count = 0
            status = "↓NEW"
        else:
            no_improve_count += 1
            status = f"×{no_improve_count}"
        
        # 更新epoch进度条（精简信息）
        epoch_pbar.set_postfix({
            'L': f'{avg_loss:.1f}',
            'Best': f'{best_loss:.1f}@E{best_epoch}',
            'S': status
        })
        
        # 早停检查
        if epoch >= min_epochs and no_improve_count >= patience:
            epoch_pbar.close()
            print(f"\n[Train] ⚠ Early stop at E{epoch+1} (best: E{best_epoch}, loss: {best_loss:.2f})")
            
            # 清理
            del optimizer, scheduler, distill_data
            gc.collect()
            torch.cuda.empty_cache()
            
            return best_loss, epoch + 1
    
    epoch_pbar.close()
    print(f"\n[Train] ✓ Completed {args.epochs} epochs (best: E{best_epoch}, loss: {best_loss:.2f})")
    
    # 清理
    del optimizer, scheduler, distill_data
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_loss, args.epochs

if __name__ == '__main__':
    args = get_args()
    train_layer_by_layer(args)