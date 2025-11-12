import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_fftchain import get_args
from utils import replace_linear_with_fftchain, load_checkpoint
import os
import gc
import time
import argparse
from contextlib import contextmanager

class FFTChainProfiler:
    def __init__(self):
        self.timings = {
            'input_padding': [],
            'input_reshape': [],
            'dtype_convert_to_fft': [],
            'fft_forward': [],
            'fft_unsqueeze': [],
            'circulant_fft': [],
            'freq_multiply': [],
            'freq_sum': [],
            'channel_weight': [],
            'ifft_forward': [],
            'output_reshape': [],
            'output_unpad': [],
            'dtype_convert_back': [],
            'total_forward': []
        }
        self.events = {}
        
    def create_events(self):
        for key in self.timings.keys():
            self.events[f'{key}_start'] = torch.cuda.Event(enable_timing=True)
            self.events[f'{key}_end'] = torch.cuda.Event(enable_timing=True)
    
    def record_start(self, name):
        self.events[f'{name}_start'].record()
    
    def record_end(self, name):
        self.events[f'{name}_end'].record()
    
    def sync_and_save(self):
        torch.cuda.synchronize()
        for key in self.timings.keys():
            start_key = f'{key}_start'
            end_key = f'{key}_end'
            if start_key in self.events and end_key in self.events:
                try:
                    elapsed = self.events[start_key].elapsed_time(self.events[end_key])
                    self.timings[key].append(elapsed)
                except:
                    pass
    
    def get_average_timings(self):
        avg_timings = {}
        for key, values in self.timings.items():
            if values:
                avg_timings[key] = sum(values) / len(values)
            else:
                avg_timings[key] = 0.0
        return avg_timings
    
    def reset(self):
        for key in self.timings.keys():
            self.timings[key] = []

class FFTChainMatrixProfiled(nn.Module):
    def __init__(self, original_module, profiler):
        super().__init__()
        self.in_features = original_module.in_features
        self.out_features = original_module.out_features
        self.block_size = original_module.block_size
        self.num_fft_matrices = original_module.num_fft_matrices
        self.dtype = original_module.dtype
        
        self.padded_in_features = original_module.padded_in_features
        self.padded_out_features = original_module.padded_out_features
        self.in_padding = original_module.in_padding
        self.out_padding = original_module.out_padding
        self.num_in_blocks = original_module.num_in_blocks
        self.num_out_blocks = original_module.num_out_blocks
        
        self.circulant_params = original_module.circulant_params
        self.channel_weights = original_module.channel_weights
        
        self.profiler = profiler
        
    def forward(self, x):
        self.profiler.record_start('total_forward')
        
        batch_size, seq_len, in_feat = x.shape
        original_dtype = x.dtype
        
        if self.in_padding > 0:
            self.profiler.record_start('input_padding')
            x = torch.nn.functional.pad(x, (0, self.in_padding), mode='constant', value=0)
            self.profiler.record_end('input_padding')
        
        self.profiler.record_start('input_reshape')
        x_blocks = x.view(batch_size, seq_len, self.num_in_blocks, self.block_size)
        self.profiler.record_end('input_reshape')
        
        self.profiler.record_start('dtype_convert_to_fft')
        x_blocks = x_blocks.to(self.dtype)
        self.profiler.record_end('dtype_convert_to_fft')
        
        self.profiler.record_start('fft_forward')
        x_fft = torch.fft.rfft(x_blocks, dim=-1)
        self.profiler.record_end('fft_forward')
        
        self.profiler.record_start('fft_unsqueeze')
        x_fft = x_fft.unsqueeze(2).unsqueeze(2)
        self.profiler.record_end('fft_unsqueeze')
        
        self.profiler.record_start('circulant_fft')
        circ_fft = torch.fft.rfft(self.circulant_params, dim=-1)
        circ_fft = circ_fft.unsqueeze(0).unsqueeze(0)
        self.profiler.record_end('circulant_fft')
        
        self.profiler.record_start('freq_multiply')
        result_fft = x_fft * circ_fft
        self.profiler.record_end('freq_multiply')
        
        self.profiler.record_start('freq_sum')
        result_fft = result_fft.sum(dim=4)
        self.profiler.record_end('freq_sum')
        
        self.profiler.record_start('channel_weight')
        result_fft_weighted = (result_fft * self.channel_weights.view(1, 1, -1, 1, 1)).sum(dim=2)
        self.profiler.record_end('channel_weight')
        
        self.profiler.record_start('ifft_forward')
        output = torch.fft.irfft(result_fft_weighted, n=self.block_size, dim=-1)
        self.profiler.record_end('ifft_forward')
        
        self.profiler.record_start('output_reshape')
        output = output.view(batch_size, seq_len, self.padded_out_features)
        self.profiler.record_end('output_reshape')
        
        if self.out_padding > 0:
            self.profiler.record_start('output_unpad')
            output = output[..., :-self.out_padding]
            self.profiler.record_end('output_unpad')
        
        self.profiler.record_start('dtype_convert_back')
        output = output.to(original_dtype)
        self.profiler.record_end('dtype_convert_back')
        
        self.profiler.record_end('total_forward')
        
        return output

def replace_with_profiled_fftchain(model, profiler):
    replaced_count = 0
    for name, module in model.named_modules():
        if 'FFTChainMatrix' in str(type(module)):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            profiled_module = FFTChainMatrixProfiled(module, profiler)
            setattr(parent, parts[-1], profiled_module)
            replaced_count += 1
    
    return replaced_count

def benchmark_original_model(args):
    print("\n" + "="*80)
    print("BENCHMARKING ORIGINAL MODEL")
    print("="*80 + "\n")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Loading original model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant land",
        "The most important scientific discovery was",
        "In a world where technology advances rapidly",
        "Scientists have recently discovered that"
    ]
    
    print(f"\nWarming up ({args.warmup_rounds} rounds)...")
    with torch.no_grad():
        for _ in range(args.warmup_rounds):
            inputs = tokenizer(test_prompts[0], return_tensors="pt").to(args.device)
            _ = model.generate(**inputs, max_length=args.max_length, pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
    
    print(f"Running benchmark ({args.test_rounds} rounds per prompt)...\n")
    
    results = {
        'total_time': [],
        'tokens_generated': [],
        'tokens_per_second': [],
        'memory_allocated': [],
        'memory_reserved': []
    }
    
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(test_prompts):
            print(f"[{prompt_idx+1}/{len(test_prompts)}] Prompt: {prompt}")
            
            for round_idx in range(args.test_rounds):
                torch.cuda.reset_peak_memory_stats(args.device)
                
                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                input_length = inputs['input_ids'].shape[1]
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                outputs = model.generate(
                    **inputs,
                    max_length=args.max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )
                end_event.record()
                
                torch.cuda.synchronize()
                
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0
                output_length = outputs.shape[1]
                tokens_generated = output_length - input_length
                tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
                
                memory_allocated = torch.cuda.max_memory_allocated(args.device) / 1024**3
                memory_reserved = torch.cuda.max_memory_reserved(args.device) / 1024**3
                
                results['total_time'].append(elapsed_time)
                results['tokens_generated'].append(tokens_generated)
                results['tokens_per_second'].append(tokens_per_sec)
                results['memory_allocated'].append(memory_allocated)
                results['memory_reserved'].append(memory_reserved)
                
                del inputs, outputs
                torch.cuda.empty_cache()
            
            print(f"  Avg tokens/s: {sum(results['tokens_per_second'][-args.test_rounds:]) / args.test_rounds:.2f}")
    
    avg_results = {
        'total_time': sum(results['total_time']) / len(results['total_time']),
        'tokens_generated': sum(results['tokens_generated']) / len(results['tokens_generated']),
        'tokens_per_second': sum(results['tokens_per_second']) / len(results['tokens_per_second']),
        'memory_allocated': sum(results['memory_allocated']) / len(results['memory_allocated']),
        'memory_reserved': sum(results['memory_reserved']) / len(results['memory_reserved'])
    }
    
    print("\n" + "-"*80)
    print("ORIGINAL MODEL RESULTS (Average)")
    print("-"*80)
    print(f"Total Time:           {avg_results['total_time']:.4f} s")
    print(f"Tokens Generated:     {avg_results['tokens_generated']:.1f}")
    print(f"Tokens/Second:        {avg_results['tokens_per_second']:.2f}")
    print(f"Memory Allocated:     {avg_results['memory_allocated']:.3f} GB")
    print(f"Memory Reserved:      {avg_results['memory_reserved']:.3f} GB")
    print("-"*80 + "\n")
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_results

def benchmark_fftchain_model(args):
    print("\n" + "="*80)
    print("BENCHMARKING FFTCHAIN MODEL")
    print("="*80 + "\n")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Loading FFTChain model from {args.model_path}...")
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
    
    print(f"Converting {len(replaced_modules)} modules to float32...")
    for module in replaced_modules:
        module.to('cpu').to(torch.float32)
    
    checkpoint_name = f'fftchain_L{args.layer_start}-{args.layer_end}_{args.target_matrix}_final.pt'
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_name}")
        checkpoint = load_checkpoint(model, None, checkpoint_path, device='cpu')
        
        print(f"Moving model to {args.device}...")
        for module in replaced_modules:
            module.to(args.device)
    else:
        print(f"WARNING: No checkpoint found, using random initialization!")
        for module in replaced_modules:
            module.to(args.device)
    
    profiler = FFTChainProfiler()
    profiler.create_events()
    
    num_replaced = replace_with_profiled_fftchain(model, profiler)
    print(f"Replaced {num_replaced} modules with profiled versions\n")
    
    model.eval()
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant land",
        "The most important scientific discovery was",
        "In a world where technology advances rapidly",
        "Scientists have recently discovered that"
    ]
    
    print(f"Warming up ({args.warmup_rounds} rounds)...")
    with torch.no_grad():
        for _ in range(args.warmup_rounds):
            inputs = tokenizer(test_prompts[0], return_tensors="pt").to(args.device)
            _ = model.generate(**inputs, max_length=args.max_length, pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
    
    profiler.reset()
    
    print(f"Running benchmark ({args.test_rounds} rounds per prompt)...\n")
    
    results = {
        'total_time': [],
        'tokens_generated': [],
        'tokens_per_second': [],
        'memory_allocated': [],
        'memory_reserved': []
    }
    
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(test_prompts):
            print(f"[{prompt_idx+1}/{len(test_prompts)}] Prompt: {prompt}")
            
            for round_idx in range(args.test_rounds):
                torch.cuda.reset_peak_memory_stats(args.device)
                
                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                input_length = inputs['input_ids'].shape[1]
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                outputs = model.generate(
                    **inputs,
                    max_length=args.max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )
                end_event.record()
                
                torch.cuda.synchronize()
                profiler.sync_and_save()
                
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0
                output_length = outputs.shape[1]
                tokens_generated = output_length - input_length
                tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
                
                memory_allocated = torch.cuda.max_memory_allocated(args.device) / 1024**3
                memory_reserved = torch.cuda.max_memory_reserved(args.device) / 1024**3
                
                results['total_time'].append(elapsed_time)
                results['tokens_generated'].append(tokens_generated)
                results['tokens_per_second'].append(tokens_per_sec)
                results['memory_allocated'].append(memory_allocated)
                results['memory_reserved'].append(memory_reserved)
                
                del inputs, outputs
                torch.cuda.empty_cache()
            
            print(f"  Avg tokens/s: {sum(results['tokens_per_second'][-args.test_rounds:]) / args.test_rounds:.2f}")
    
    avg_results = {
        'total_time': sum(results['total_time']) / len(results['total_time']),
        'tokens_generated': sum(results['tokens_generated']) / len(results['tokens_generated']),
        'tokens_per_second': sum(results['tokens_per_second']) / len(results['tokens_per_second']),
        'memory_allocated': sum(results['memory_allocated']) / len(results['memory_allocated']),
        'memory_reserved': sum(results['memory_reserved']) / len(results['memory_reserved'])
    }
    
    avg_timings = profiler.get_average_timings()
    
    print("\n" + "-"*80)
    print("FFTCHAIN MODEL RESULTS (Average)")
    print("-"*80)
    print(f"Total Time:           {avg_results['total_time']:.4f} s")
    print(f"Tokens Generated:     {avg_results['tokens_generated']:.1f}")
    print(f"Tokens/Second:        {avg_results['tokens_per_second']:.2f}")
    print(f"Memory Allocated:     {avg_results['memory_allocated']:.3f} GB")
    print(f"Memory Reserved:      {avg_results['memory_reserved']:.3f} GB")
    print("-"*80)
    print("\nFFTCHAIN KERNEL BREAKDOWN (Average per Forward Pass)")
    print("-"*80)
    print(f"Input Padding:        {avg_timings['input_padding']:.4f} ms")
    print(f"Input Reshape:        {avg_timings['input_reshape']:.4f} ms")
    print(f"Dtype Convert (in):   {avg_timings['dtype_convert_to_fft']:.4f} ms")
    print(f"FFT Forward:          {avg_timings['fft_forward']:.4f} ms")
    print(f"FFT Unsqueeze:        {avg_timings['fft_unsqueeze']:.4f} ms")
    print(f"Circulant FFT:        {avg_timings['circulant_fft']:.4f} ms")
    print(f"Frequency Multiply:   {avg_timings['freq_multiply']:.4f} ms")
    print(f"Frequency Sum:        {avg_timings['freq_sum']:.4f} ms")
    print(f"Channel Weighting:    {avg_timings['channel_weight']:.4f} ms")
    print(f"IFFT Forward:         {avg_timings['ifft_forward']:.4f} ms")
    print(f"Output Reshape:       {avg_timings['output_reshape']:.4f} ms")
    print(f"Output Unpadding:     {avg_timings['output_unpad']:.4f} ms")
    print(f"Dtype Convert (out):  {avg_timings['dtype_convert_back']:.4f} ms")
    print(f"{'─'*80}")
    print(f"Total Forward:        {avg_timings['total_forward']:.4f} ms")
    print("-"*80 + "\n")
    
    del model, tokenizer, profiler
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_results, avg_timings

def compare_results(original_results, fftchain_results):
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    speedup = original_results['tokens_per_second'] / fftchain_results['tokens_per_second']
    time_ratio = fftchain_results['total_time'] / original_results['total_time']
    memory_ratio = fftchain_results['memory_allocated'] / original_results['memory_allocated']
    
    print(f"\nTokens/Second:")
    print(f"  Original:    {original_results['tokens_per_second']:.2f}")
    print(f"  FFTChain:    {fftchain_results['tokens_per_second']:.2f}")
    print(f"  Speedup:     {speedup:.2f}x")
    
    print(f"\nTotal Time:")
    print(f"  Original:    {original_results['total_time']:.4f} s")
    print(f"  FFTChain:    {fftchain_results['total_time']:.4f} s")
    print(f"  Ratio:       {time_ratio:.2f}x")
    
    print(f"\nMemory Allocated:")
    print(f"  Original:    {original_results['memory_allocated']:.3f} GB")
    print(f"  FFTChain:    {fftchain_results['memory_allocated']:.3f} GB")
    print(f"  Ratio:       {memory_ratio:.2f}x")
    
    print("="*80 + "\n")

def get_benchmark_args():
    parser = argparse.ArgumentParser(description='FFTChain Benchmark')
    
    parser.add_argument('--model_path', type=str, 
                       default='/home/zx/Proj/llama-2-13b-chat-hf_debug_FFTChain/model')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda:2')
    
    parser.add_argument('--layer_start', type=int, default=35)
    parser.add_argument('--layer_end', type=int, default=39)
    parser.add_argument('--target_matrix', type=str, default='down_proj',
                       choices=['down_proj', 'up_proj', 'gate_proj'])
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--num_fft_matrices', type=int, default=4)
    
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--warmup_rounds', type=int, default=10)
    parser.add_argument('--test_rounds', type=int, default=10)
    
    parser.add_argument('--skip_original', action='store_true',
                       help='Skip benchmarking original model')
    parser.add_argument('--skip_fftchain', action='store_true',
                       help='Skip benchmarking FFTChain model')
    
    return parser.parse_args()

def main():
    args = get_benchmark_args()
    
    print("\n" + "="*80)
    print("FFTChain Benchmark Configuration")
    print("="*80)
    print(f"Model Path:       {args.model_path}")
    print(f"Device:           {args.device}")
    print(f"Layers:           {args.layer_start}-{args.layer_end}")
    print(f"Target Matrix:    {args.target_matrix}")
    print(f"Block Size:       {args.block_size}")
    print(f"Num FFT Matrices: {args.num_fft_matrices}")
    print(f"Max Length:       {args.max_length}")
    print(f"Warmup Rounds:    {args.warmup_rounds}")
    print(f"Test Rounds:      {args.test_rounds}")
    print("="*80)
    
    original_results = None
    fftchain_results = None
    
    if not args.skip_original:
        original_results = benchmark_original_model(args)
    
    if not args.skip_fftchain:
        fftchain_results, fftchain_timings = benchmark_fftchain_model(args)
    
    if original_results and fftchain_results:
        compare_results(original_results, fftchain_results)
    
    print("Benchmark completed!")

if __name__ == '__main__':
    main()