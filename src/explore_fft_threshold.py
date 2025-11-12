import torch
import torch.nn as nn
import time
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from fftchain import FFTChainMatrix
from config_fftchain import get_args
import gc

class LinearLayerBenchmark:
    def __init__(self, model_path, device='cuda:0'):
        self.device = device
        self.model_path = model_path
        
    def get_layer_shapes(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map={"": 'cpu'}
        )
        
        shapes = {}
        num_layers = len(model.model.layers)
        
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            shapes[layer_idx] = {
                'down_proj': (layer.mlp.down_proj.in_features, layer.mlp.down_proj.out_features),
                'up_proj': (layer.mlp.up_proj.in_features, layer.mlp.up_proj.out_features),
                'gate_proj': (layer.mlp.gate_proj.in_features, layer.mlp.gate_proj.out_features),
            }
        
        del model
        gc.collect()
        
        return shapes

    def benchmark_linear_vs_fft(self, in_features, out_features, block_size, num_fft_matrices, 
                                batch_size=1, seq_len=128, num_runs=50, warmup=10):
        
        torch.cuda.empty_cache()
        
        linear = nn.Linear(in_features, out_features, bias=False).to(self.device).to(torch.float16)
        
        fft_layer = FFTChainMatrix(
            in_features=in_features,
            out_features=out_features,
            block_size=block_size,
            num_fft_matrices=num_fft_matrices,
            dtype=torch.float16
        ).to(self.device).to(torch.float32)
        
        x = torch.randn(batch_size, seq_len, in_features, device=self.device, dtype=torch.float16)
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = linear(x)
                torch.cuda.synchronize()
        
        linear_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = linear(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                linear_times.append((end - start) * 1000)
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = fft_layer(x)
                torch.cuda.synchronize()
        
        fft_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = fft_layer(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                fft_times.append((end - start) * 1000)
        
        linear_params = in_features * out_features
        fft_params = num_fft_matrices * ((in_features + block_size - 1) // block_size) * \
                     ((out_features + block_size - 1) // block_size) * block_size + num_fft_matrices
        
        del linear, fft_layer, x
        torch.cuda.empty_cache()
        
        return {
            'linear_mean_ms': np.mean(linear_times),
            'linear_std_ms': np.std(linear_times),
            'fft_mean_ms': np.mean(fft_times),
            'fft_std_ms': np.std(fft_times),
            'speedup': np.mean(linear_times) / np.mean(fft_times),
            'linear_params': linear_params,
            'fft_params': fft_params,
            'compression_ratio': linear_params / fft_params
        }

    def explore_block_k_combinations(self, in_features, out_features, 
                                     block_sizes=[128, 256, 512, 1024, 2048],
                                     k_values=[2, 4, 8, 16],
                                     batch_size=1, seq_len=128):
        
        results = []
        total_configs = len(block_sizes) * len(k_values)
        current = 0
        
        print(f"\nExploring {in_features}×{out_features} matrix")
        print(f"Testing {total_configs} configurations...")
        print("-" * 80)
        
        for block_size in block_sizes:
            for k in k_values:
                current += 1
                
                try:
                    result = self.benchmark_linear_vs_fft(
                        in_features, out_features, block_size, k, 
                        batch_size, seq_len, num_runs=30, warmup=5
                    )
                    
                    result['block_size'] = block_size
                    result['k'] = k
                    result['in_features'] = in_features
                    result['out_features'] = out_features
                    
                    speedup_symbol = '✓' if result['speedup'] > 1.0 else '✗'
                    print(f"[{current:2d}/{total_configs}] B={block_size:4d} K={k:2d} | "
                          f"Linear: {result['linear_mean_ms']:6.2f}ms | "
                          f"FFT: {result['fft_mean_ms']:6.2f}ms | "
                          f"Speedup: {result['speedup']:.2f}x {speedup_symbol} | "
                          f"Compress: {result['compression_ratio']:.0f}x")
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"[{current:2d}/{total_configs}] B={block_size:4d} K={k:2d} | Failed: {str(e)}")
                    
                torch.cuda.empty_cache()
        
        return results

def analyze_full_model_layers(args, test_configs):
    print("\n" + "="*80)
    print("FULL MODEL LAYER-BY-LAYER FFT THRESHOLD ANALYSIS")
    print("="*80)
    
    benchmark = LinearLayerBenchmark(args.model_path, args.device)
    
    print("\nStep 1: Getting layer shapes...")
    shapes = benchmark.get_layer_shapes()
    print(f"Found {len(shapes)} layers")
    
    for layer_idx, layer_shapes in list(shapes.items())[:3]:
        print(f"  Layer {layer_idx}: down_proj {layer_shapes['down_proj']}, "
              f"up_proj {layer_shapes['up_proj']}, gate_proj {layer_shapes['gate_proj']}")
    
    print("\nStep 2: Testing different layer types...")
    
    all_results = {}
    
    test_layers = [0, 10, 19, 30, 39]
    test_matrices = ['down_proj', 'up_proj', 'gate_proj']
    
    for layer_idx in test_layers:
        if layer_idx not in shapes:
            continue
            
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx}")
        print('='*80)
        
        for matrix_name in test_matrices:
            in_feat, out_feat = shapes[layer_idx][matrix_name]
            
            print(f"\n[Layer {layer_idx} - {matrix_name}]: {in_feat} → {out_feat}")
            
            results = benchmark.explore_block_k_combinations(
                in_feat, out_feat,
                block_sizes=test_configs['block_sizes'],
                k_values=test_configs['k_values'],
                batch_size=test_configs['batch_size'],
                seq_len=test_configs['seq_len']
            )
            
            key = f"layer_{layer_idx}_{matrix_name}"
            all_results[key] = results
    
    return all_results

def find_optimal_configs(results):
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION ANALYSIS")
    print("="*80)
    
    optimal_configs = {}
    
    for key, layer_results in results.items():
        if not layer_results:
            continue
        
        best_speedup = max(layer_results, key=lambda x: x['speedup'])
        best_compression = max(layer_results, key=lambda x: x['compression_ratio'])
        
        speedup_configs = [r for r in layer_results if r['speedup'] > 1.0]
        
        if speedup_configs:
            best_balance = max(speedup_configs, 
                             key=lambda x: x['speedup'] * np.log(x['compression_ratio']))
        else:
            best_balance = None
        
        optimal_configs[key] = {
            'best_speedup': best_speedup,
            'best_compression': best_compression,
            'best_balance': best_balance,
            'num_speedup_configs': len(speedup_configs),
            'total_configs': len(layer_results)
        }
    
    print("\nSummary by Layer:")
    print("-" * 80)
    for key, config in optimal_configs.items():
        layer_info = key.replace('_', ' ').title()
        print(f"\n{layer_info}:")
        
        if config['best_balance']:
            print(f"  Best Balance: B={config['best_balance']['block_size']}, "
                  f"K={config['best_balance']['k']} | "
                  f"Speedup: {config['best_balance']['speedup']:.2f}x | "
                  f"Compress: {config['best_balance']['compression_ratio']:.0f}x")
        
        print(f"  Max Speedup: B={config['best_speedup']['block_size']}, "
              f"K={config['best_speedup']['k']} | "
              f"{config['best_speedup']['speedup']:.2f}x")
        
        print(f"  Configs with speedup: {config['num_speedup_configs']}/{config['total_configs']}")
    
    return optimal_configs

def analyze_threshold_patterns(results):
    print("\n" + "="*80)
    print("FFT THRESHOLD PATTERN ANALYSIS")
    print("="*80)
    
    all_data = []
    for key, layer_results in results.items():
        all_data.extend(layer_results)
    
    if not all_data:
        return
    
    print("\n1. Block Size Analysis:")
    print("-" * 80)
    block_groups = {}
    for d in all_data:
        b = d['block_size']
        if b not in block_groups:
            block_groups[b] = []
        block_groups[b].append(d['speedup'])
    
    for block_size in sorted(block_groups.keys()):
        speedups = block_groups[block_size]
        avg_speedup = np.mean(speedups)
        num_faster = sum(1 for s in speedups if s > 1.0)
        print(f"  Block {block_size:4d}: Avg speedup {avg_speedup:.2f}x | "
              f"{num_faster}/{len(speedups)} configs faster")
    
    print("\n2. K Value Analysis:")
    print("-" * 80)
    k_groups = {}
    for d in all_data:
        k = d['k']
        if k not in k_groups:
            k_groups[k] = []
        k_groups[k].append(d['speedup'])
    
    for k_val in sorted(k_groups.keys()):
        speedups = k_groups[k_val]
        avg_speedup = np.mean(speedups)
        num_faster = sum(1 for s in speedups if s > 1.0)
        print(f"  K={k_val:2d}: Avg speedup {avg_speedup:.2f}x | "
              f"{num_faster}/{len(speedups)} configs faster")
    
    print("\n3. Matrix Size vs Speedup:")
    print("-" * 80)
    size_groups = {}
    for d in all_data:
        size = d['in_features'] * d['out_features']
        size_category = f"{size/1e6:.0f}M"
        if size_category not in size_groups:
            size_groups[size_category] = []
        size_groups[size_category].append(d['speedup'])
    
    for size_cat in sorted(size_groups.keys(), key=lambda x: float(x[:-1])):
        speedups = size_groups[size_cat]
        avg_speedup = np.mean(speedups)
        print(f"  {size_cat:>6s} params: Avg speedup {avg_speedup:.2f}x")
    
    print("\n4. Compression vs Speedup Correlation:")
    print("-" * 80)
    compressions = [d['compression_ratio'] for d in all_data]
    speedups = [d['speedup'] for d in all_data]
    
    if len(compressions) > 1:
        correlation = np.corrcoef(compressions, speedups)[0, 1]
        print(f"  Correlation coefficient: {correlation:.3f}")
        
        high_compress = [d for d in all_data if d['compression_ratio'] > 100]
        if high_compress:
            avg_speedup_high = np.mean([d['speedup'] for d in high_compress])
            print(f"  High compression (>100x): Avg speedup {avg_speedup_high:.2f}x")
        
        low_compress = [d for d in all_data if d['compression_ratio'] <= 100]
        if low_compress:
            avg_speedup_low = np.mean([d['speedup'] for d in low_compress])
            print(f"  Low compression (≤100x): Avg speedup {avg_speedup_low:.2f}x")

def generate_recommendations(results, optimal_configs):
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR DEPLOYMENT")
    print("="*80)
    
    all_balanced = []
    for key, config in optimal_configs.items():
        if config['best_balance']:
            all_balanced.append(config['best_balance'])
    
    if not all_balanced:
        print("\nNo configurations achieved speedup!")
        return
    
    block_sizes = [c['block_size'] for c in all_balanced]
    k_values = [c['k'] for c in all_balanced]
    
    from collections import Counter
    block_counter = Counter(block_sizes)
    k_counter = Counter(k_values)
    
    print("\n1. Most Common Successful Configurations:")
    print("-" * 80)
    print(f"  Block sizes: {dict(block_counter)}")
    print(f"  K values: {dict(k_counter)}")
    
    most_common_block = block_counter.most_common(1)[0][0]
    most_common_k = k_counter.most_common(1)[0][0]
    
    print(f"\n  Recommended default: block_size={most_common_block}, K={most_common_k}")
    
    print("\n2. Layer-Specific Recommendations:")
    print("-" * 80)
    
    for key, config in optimal_configs.items():
        if config['best_balance']:
            layer_name = key.replace('_', ' ').replace('layer ', 'Layer ')
            b = config['best_balance']
            print(f"  {layer_name:30s}: B={b['block_size']:4d}, K={b['k']:2d} "
                  f"({b['speedup']:.2f}x speedup, {b['compression_ratio']:.0f}x compression)")
    
    print("\n3. Training Strategy:")
    print("-" * 80)
    
    upper_layers = [k for k in optimal_configs.keys() if 'layer_3' in k or 'layer_2' in k]
    lower_layers = [k for k in optimal_configs.keys() if 'layer_0' in k or 'layer_1' in k]
    
    if upper_layers:
        upper_configs = [optimal_configs[k]['best_balance'] for k in upper_layers 
                        if optimal_configs[k]['best_balance']]
        if upper_configs:
            avg_upper_speedup = np.mean([c['speedup'] for c in upper_configs])
            print(f"  Upper layers (30-39): Avg speedup {avg_upper_speedup:.2f}x")
            print(f"    → Priority for replacement")
    
    if lower_layers:
        lower_configs = [optimal_configs[k]['best_balance'] for k in lower_layers 
                        if optimal_configs[k]['best_balance']]
        if lower_configs:
            avg_lower_speedup = np.mean([c['speedup'] for c in lower_configs])
            print(f"  Lower layers (0-10): Avg speedup {avg_lower_speedup:.2f}x")
            if avg_lower_speedup < 1.0:
                print(f"    → Not recommended for replacement")

def main():
    args = get_args()
    
    test_configs = {
        'block_sizes': [256, 512, 1024, 2048],
        'k_values': [2, 4, 8, 16],
        'batch_size': 1,
        'seq_len': 128
    }
    
    print("="*80)
    print("FFT ACCELERATION THRESHOLD EXPLORATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Block sizes to test: {test_configs['block_sizes']}")
    print(f"  K values to test: {test_configs['k_values']}")
    print(f"  Batch size: {test_configs['batch_size']}")
    print(f"  Sequence length: {test_configs['seq_len']}")
    
    results = analyze_full_model_layers(args, test_configs)
    
    optimal_configs = find_optimal_configs(results)
    
    analyze_threshold_patterns(results)
    
    generate_recommendations(results, optimal_configs)
    
    output_file = 'fft_threshold_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()