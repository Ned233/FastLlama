import torch
import time
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_fftchain import get_args
from utils import replace_linear_with_fftchain, load_checkpoint
import gc
import os
from scipy import stats

def benchmark_with_trained_checkpoint(args):
    print("="*80)
    print("REALISTIC THRESHOLD TEST - TRAINED CHECKPOINT")
    print("="*80)
    print(f"Testing configuration: L{args.layer_start}-{args.layer_end}, "
          f"B={args.block_size}, K={args.num_fft_matrices}")
    print()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances",
        "Scientists have recently discovered"
    ]
    
    num_runs = 30
    seq_len = 128
    
    print("Step 1: Benchmarking original model...")
    print("-" * 80)
    
    original_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    original_model.eval()
    
    orig_times = []
    with torch.no_grad():
        for _ in range(5):
            prompt = test_prompts[_ % len(test_prompts)]
            inputs = tokenizer(prompt, return_tensors="pt", 
                             padding='max_length', max_length=seq_len, 
                             truncation=True).to(args.device)
            _ = original_model(**inputs)
            torch.cuda.synchronize()
            del inputs
        
        for run in range(num_runs):
            prompt = test_prompts[run % len(test_prompts)]
            inputs = tokenizer(prompt, return_tensors="pt",
                             padding='max_length', max_length=seq_len,
                             truncation=True).to(args.device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = original_model(**inputs)
            torch.cuda.synchronize()
            orig_times.append((time.perf_counter() - start) * 1000)
            
            del inputs
            
            if (run + 1) % 10 == 0:
                print(f"  Progress: {run+1}/{num_runs} runs")
    
    orig_mean = np.mean(orig_times)
    orig_std = np.std(orig_times)
    
    print(f"\nOriginal Model Results:")
    print(f"  Mean: {orig_mean:.2f} ± {orig_std:.2f} ms")
    print(f"  Min:  {np.min(orig_times):.2f} ms")
    print(f"  Max:  {np.max(orig_times):.2f} ms")
    
    del original_model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
    print("\nStep 2: Benchmarking FFTChain model...")
    print("-" * 80)
    
    fft_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    
    layer_indices = list(range(args.layer_start, args.layer_end + 1))
    replaced_modules = replace_linear_with_fftchain(
        fft_model, layer_indices, args.target_matrix, 
        args.block_size, args.num_fft_matrices
    )
    
    for module in replaced_modules:
        module.to('cpu').to(torch.float32)
    
    checkpoint_name = f'fftchain_L{args.layer_start}-{args.layer_end}_{args.target_matrix}_final.pt'
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_name}")
        load_checkpoint(fft_model, None, checkpoint_path, device='cpu')
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print(f"Using random initialization")
    
    for module in replaced_modules:
        module.to(args.device)
    
    fft_model.eval()
    
    fft_times = []
    with torch.no_grad():
        for _ in range(5):
            prompt = test_prompts[_ % len(test_prompts)]
            inputs = tokenizer(prompt, return_tensors="pt",
                             padding='max_length', max_length=seq_len,
                             truncation=True).to(args.device)
            _ = fft_model(**inputs)
            torch.cuda.synchronize()
            del inputs
        
        for run in range(num_runs):
            prompt = test_prompts[run % len(test_prompts)]
            inputs = tokenizer(prompt, return_tensors="pt",
                             padding='max_length', max_length=seq_len,
                             truncation=True).to(args.device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = fft_model(**inputs)
            torch.cuda.synchronize()
            fft_times.append((time.perf_counter() - start) * 1000)
            
            del inputs
            
            if (run + 1) % 10 == 0:
                print(f"  Progress: {run+1}/{num_runs} runs")
    
    fft_mean = np.mean(fft_times)
    fft_std = np.std(fft_times)
    
    print(f"\nFFTChain Model Results:")
    print(f"  Mean: {fft_mean:.2f} ± {fft_std:.2f} ms")
    print(f"  Min:  {np.min(fft_times):.2f} ms")
    print(f"  Max:  {np.max(fft_times):.2f} ms")
    
    del fft_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    speedup = orig_mean / fft_mean
    
    print(f"\nForward Pass:")
    print(f"  Original:  {orig_mean:.2f} ± {orig_std:.2f} ms")
    print(f"  FFTChain:  {fft_mean:.2f} ± {fft_std:.2f} ms")
    
    if speedup > 1.0:
        print(f"  Speedup:   {speedup:.2f}x ✓ FASTER")
    else:
        print(f"  Speedup:   {speedup:.2f}x ✗ SLOWER")
    
    num_layers = args.layer_end - args.layer_start + 1
    n_blocks = ((13824 + args.block_size - 1) // args.block_size)
    m_blocks = ((5120 + args.block_size - 1) // args.block_size)
    
    linear_params = 13824 * 5120 * num_layers
    fft_params = args.num_fft_matrices * n_blocks * m_blocks * args.block_size * num_layers
    compression = linear_params / fft_params
    
    print(f"\nParameters:")
    print(f"  Linear:       {linear_params/1e6:.1f}M")
    print(f"  FFTChain:     {fft_params/1e6:.1f}M")
    print(f"  Compression:  {compression:.0f}x")
    
    print(f"\nStatistical Significance:")
    t_stat, p_value = stats.ttest_ind(orig_times, fft_times)
    print(f"  T-statistic: {t_stat:.2f}")
    print(f"  P-value:     {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result:      Statistically significant ✓")
    else:
        print(f"  Result:      Not significant")
    
    result = {
        'config': {
            'layer_start': args.layer_start,
            'layer_end': args.layer_end,
            'block_size': args.block_size,
            'k': args.num_fft_matrices
        },
        'original': {
            'mean_ms': float(orig_mean),
            'std_ms': float(orig_std),
            'min_ms': float(np.min(orig_times)),
            'max_ms': float(np.max(orig_times)),
            'times': [float(t) for t in orig_times]
        },
        'fftchain': {
            'mean_ms': float(fft_mean),
            'std_ms': float(fft_std),
            'min_ms': float(np.min(fft_times)),
            'max_ms': float(np.max(fft_times)),
            'times': [float(t) for t in fft_times]
        },
        'speedup': float(speedup),
        'compression': float(compression),
        'statistical': {
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    }
    
    output_file = f'realistic_test_L{args.layer_start}-{args.layer_end}_B{args.block_size}_K{args.num_fft_matrices}.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    args = get_args()
    
    print(f"Device: {args.device}")
    print(f"Model: {args.model_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print()
    
    try:
        benchmark_with_trained_checkpoint(args)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()