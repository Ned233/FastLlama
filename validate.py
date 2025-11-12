import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_fftchain import get_args
from utils import replace_linear_with_fftchain, load_checkpoint
import os
import gc
import glob

def validate(args):
    # 先清理显存
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"[Validate] Loading model from {args.model_path} on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    layer_indices = list(range(args.layer_start, args.layer_end + 1))
    print(f"[Validate] Replacing {args.target_matrix} in layers {layer_indices}...")
    replaced_modules = replace_linear_with_fftchain(
        model, layer_indices, args.target_matrix, args.block_size, args.num_fft_matrices
    )
    
    # 先转换到float32但保持在CPU
    print(f"[Validate] Converting {len(replaced_modules)} modules to float32...")
    for module in replaced_modules:
        module.to('cpu').to(torch.float32)
    
    # 查找最终的checkpoint
    checkpoint_name = f'fftchain_L{args.layer_start}-{args.layer_end}_{args.target_matrix}_final.pt'
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"[Validate] Loading checkpoint: {checkpoint_name}")
        
        # 先加载到CPU
        checkpoint = load_checkpoint(model, None, checkpoint_path, device='cpu')
        
        # 显示训练信息
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'trained_epochs' in config and 'best_loss' in config:
                print(f"[Validate] Trained: {config['trained_epochs']} epochs, best loss: {config['best_loss']:.4f}")
        
        # 再移动到GPU
        print(f"[Validate] Moving model to {args.device}...")
        for module in replaced_modules:
            module.to(args.device)
        
        checkpoint_loaded = True
    else:
        print(f"[Validate] ✗ No checkpoint found: {checkpoint_name}")
        available = glob.glob(os.path.join(args.checkpoint_dir, 'fftchain_*.pt'))
        if available:
            print(f"[Validate] Available checkpoints:")
            for ckpt in sorted(available):
                print(f"  - {os.path.basename(ckpt)}")
        print(f"[Validate] Using random initialization!")
        
        # 移动到GPU
        for module in replaced_modules:
            module.to(args.device)
        
        checkpoint_loaded = False
    
    model.eval()
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant land",
        "The most important scientific discovery was",
        "In a world where technology advances rapidly",
        "Scientists have recently discovered that"
    ]
    
    print("\n" + "="*80)
    if checkpoint_loaded:
        print("✓ Using trained FFTChain model")
    else:
        print("✗ Using random initialization (NOT trained)")
    print("="*80 + "\n")
    
    with torch.no_grad():
        for idx, prompt in enumerate(test_prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[{idx+1}] Prompt: {prompt}")
            print(f"Generated: {generated_text}\n")
            
            # 清理
            del inputs, outputs
            torch.cuda.empty_cache()
    
    print("="*80)
    print("[Validate] Completed")
    
    # 最终清理
    del model, tokenizer, replaced_modules
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = get_args()
    validate(args)