import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import get_args
from src import replace_linear_with_fftchain, load_checkpoint
import os
import numpy as np

def get_memory_usage(device):
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(device) / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024**3
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}

def profile_fft_operations(model, tokenizer, device, num_runs=10):
    torch.cuda.synchronize()
    
    prompt = "The future of artificial intelligence and machine learning in the next decade will"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    fft_times = []
    ifft_times = []
    multiplication_times = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            
            start_fft = time.perf_counter()
            _ = model(**inputs, output_hidden_states=True)
            torch.cuda.synchronize()
            
            fft_times.append(time.perf_counter() - start_fft)
    
    return {
        'forward_time_ms': np.mean(fft_times) * 1000,
        'forward_std_ms': np.std(fft_times) * 1000
    }

def benchmark_forward_pass(model, tokenizer, device, num_runs=50):
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant land",
        "Scientists have recently discovered",
        "In the world of technology",
        "The most important aspect of machine learning"
    ]
    
    model.eval()
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            prompt = prompts[_ % len(prompts)]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(**inputs)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
            
            del inputs
            torch.cuda.empty_cache()
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }

def benchmark_generation(model, tokenizer, device, num_runs=20):
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "In the future",
        "Scientists believe",
        "Technology has"
    ]
    
    model.eval()
    times = []
    tokens_per_second = []
    
    max_length = 100
    
    with torch.no_grad():
        for i in range(num_runs):
            prompt = prompts[i % len(prompts)]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs['input_ids'].shape[1]
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            generated_length = outputs.shape[1] - input_length
            elapsed = end - start
            
            times.append(elapsed)
            tokens_per_second.append(generated_length / elapsed)
            
            del inputs, outputs
            torch.cuda.empty_cache()
    
    return {
        'mean_time_s': np.mean(times),
        'std_time_s': np.std(times),
        'mean_tokens_per_s': np.mean(tokens_per_second),
        'std_tokens_per_s': np.std(tokens_per_second)
    }

def load_original_model(args):
    print(f"\n{'='*80}")
    print("Loading Original Model")
    print(f"{'='*80}\n")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    return model, tokenizer

def load_fftchain_model(args):
    print(f"\n{'='*80}")
    print("Loading FFTChain Model")
    print(f"{'='*80}\n")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    layer_indices = list(range(args.layer_start, args.layer_end + 1))
    print(f"Replacing {args.target_matrix} in layers {layer_indices}...")
    replaced_modules = replace_linear_with_fftchain(
        model, layer_indices, args.target_matrix, args.block_size, args.num_fft_matrices
    )
    
    for module in replaced_modules:
        module.to('cpu').to(torch.float32)
    
    checkpoint_name = f'fftchain_L{args.layer_start}-{args.layer_end}_{args.target_matrix}_final.pt'
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_name}")
        load_checkpoint(model, None, checkpoint_path, device='cpu')
        
        for module in replaced_modules:
            module.to(args.device)
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        for module in replaced_modules:
            module.to(args.device)
    
    model.eval()
    
    return model, tokenizer

def run_benchmark(args):
    print("\n" + "="*80)
    print("BENCHMARK: Original Model vs FFTChain Model")
    print("="*80)
    
    original_model, tokenizer = load_original_model(args)
    
    print("\nBenchmarking Original Model...")
    print("-"*80)
    
    torch.cuda.reset_peak_memory_stats(args.device)
    
    print("Forward Pass Performance (50 runs):")
    orig_forward = benchmark_forward_pass(original_model, tokenizer, args.device, num_runs=50)
    print(f"  Mean: {orig_forward['mean_ms']:.2f} ms")
    print(f"  Std:  {orig_forward['std_ms']:.2f} ms")
    print(f"  Min:  {orig_forward['min_ms']:.2f} ms")
    print(f"  Max:  {orig_forward['max_ms']:.2f} ms")
    
    print("\nGeneration Performance (20 runs, max_length=100):")
    orig_gen = benchmark_generation(original_model, tokenizer, args.device, num_runs=20)
    print(f"  Mean Time: {orig_gen['mean_time_s']:.2f} s")
    print(f"  Tokens/s:  {orig_gen['mean_tokens_per_s']:.2f} ± {orig_gen['std_tokens_per_s']:.2f}")
    
    orig_memory = get_memory_usage(args.device)
    print(f"\nMemory Usage:")
    print(f"  Allocated:     {orig_memory['allocated_gb']:.2f} GB")
    print(f"  Reserved:      {orig_memory['reserved_gb']:.2f} GB")
    print(f"  Peak Allocated: {orig_memory['max_allocated_gb']:.2f} GB")
    
    del original_model
    gc.collect()
    torch.cuda.empty_cache()
    
    fftchain_model, tokenizer = load_fftchain_model(args)
    
    print("\n" + "="*80)
    print("Benchmarking FFTChain Model...")
    print("-"*80)
    
    torch.cuda.reset_peak_memory_stats(args.device)
    
    print("Forward Pass Performance (50 runs):")
    fft_forward = benchmark_forward_pass(fftchain_model, tokenizer, args.device, num_runs=50)
    print(f"  Mean: {fft_forward['mean_ms']:.2f} ms")
    print(f"  Std:  {fft_forward['std_ms']:.2f} ms")
    print(f"  Min:  {fft_forward['min_ms']:.2f} ms")
    print(f"  Max:  {fft_forward['max_ms']:.2f} ms")
    
    print("\nGeneration Performance (20 runs, max_length=100):")
    fft_gen = benchmark_generation(fftchain_model, tokenizer, args.device, num_runs=20)
    print(f"  Mean Time: {fft_gen['mean_time_s']:.2f} s")
    print(f"  Tokens/s:  {fft_gen['mean_tokens_per_s']:.2f} ± {fft_gen['std_tokens_per_s']:.2f}")
    
    fft_memory = get_memory_usage(args.device)
    print(f"\nMemory Usage:")
    print(f"  Allocated:     {fft_memory['allocated_gb']:.2f} GB")
    print(f"  Reserved:      {fft_memory['reserved_gb']:.2f} GB")
    print(f"  Peak Allocated: {fft_memory['max_allocated_gb']:.2f} GB")
    
    print("\n" + "="*80)
    print("FFT Operations Profiling (10 runs)")
    print("-"*80)
    
    fft_profile = profile_fft_operations(fftchain_model, tokenizer, args.device, num_runs=10)
    print(f"FFT Forward Pass: {fft_profile['forward_time_ms']:.2f} ± {fft_profile['forward_std_ms']:.2f} ms")
    
    del fftchain_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nForward Pass:")
    speedup = orig_forward['mean_ms'] / fft_forward['mean_ms']
    print(f"  Original:  {orig_forward['mean_ms']:.2f} ms")
    print(f"  FFTChain:  {fft_forward['mean_ms']:.2f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    
    print("\nGeneration:")
    gen_speedup = orig_gen['mean_tokens_per_s'] / fft_gen['mean_tokens_per_s']
    print(f"  Original:  {orig_gen['mean_tokens_per_s']:.2f} tokens/s")
    print(f"  FFTChain:  {fft_gen['mean_tokens_per_s']:.2f} tokens/s")
    print(f"  Ratio:     {1/gen_speedup:.2f}x")
    
    print("\nMemory (Peak Allocated):")
    mem_reduction = (1 - fft_memory['max_allocated_gb'] / orig_memory['max_allocated_gb']) * 100
    print(f"  Original:  {orig_memory['max_allocated_gb']:.2f} GB")
    print(f"  FFTChain:  {fft_memory['max_allocated_gb']:.2f} GB")
    print(f"  Reduction: {mem_reduction:.2f}%")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    args = get_args()
    run_benchmark(args)