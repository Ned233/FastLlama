import torch
import torch.nn as nn
import time
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from fftchain import FFTChainMatrix
from config_fftchain import get_args
import gc

def benchmark_single_config(linear_layer, in_features, out_features, block_size, k, 
                            batch_size, seq_len, device, num_runs=30):
    
    fft_layer = FFTChainMatrix(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        num_fft_matrices=k,
        dtype=torch.float16
    ).to(device).to(torch.float32)
    
    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.float16)
    
    with torch.no_grad():
        for _ in range(5):
            _ = linear_layer(x)
            torch.cuda.synchronize()
    
    linear_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = linear_layer(x)
            torch.cuda.synchronize()
            linear_times.append((time.perf_counter() - start) * 1000)
    
    with torch.no_grad():
        for _ in range(5):
            _ = fft_layer(x)
            torch.cuda.synchronize()
    
    fft_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = fft_layer(x)
            torch.cuda.synchronize()
            fft_times.append((time.perf_counter() - start) * 1000)
    
    linear_params = in_features * out_features
    n_blocks = ((in_features + block_size - 1) // block_size)
    m_blocks = ((out_features + block_size - 1) // block_size)
    fft_params = k * n_blocks * m_blocks * block_size + k
    
    del fft_layer, x
    torch.cuda.empty_cache()
    
    return {
        'linear_mean': np.mean(linear_times),
        'fft_mean': np.mean(fft_times),
        'speedup': np.mean(linear_times) / np.mean(fft_times),
        'linear_params': linear_params,
        'fft_params': fft_params,
        'compression': linear_params / fft_params
    }

def quick_explore(args):
    print("="*80)
    print("QUICK FFT THRESHOLD EXPLORATION")
    print("="*80)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map={"": args.device}
    )
    
    test_layers = [0, 10, 19, 25, 30, 35, 39]
    configs_to_test = [
        {'block_size': 256, 'k': 2},
        {'block_size': 512, 'k': 2},
        {'block_size': 512, 'k': 4},
        {'block_size': 1024, 'k': 2},
        {'block_size': 1024, 'k': 4},
        {'block_size': 1024, 'k': 8},
        {'block_size': 2048, 'k': 2},
        {'block_size': 2048, 'k': 4},
    ]
    
    results = []
    total_tests = len(test_layers) * len(configs_to_test)
    current = 0
    
    for layer_idx in test_layers:
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx}")
        print('='*80)
        
        layer = model.model.layers[layer_idx]
        linear_down = layer.mlp.down_proj
        in_feat = linear_down.in_features
        out_feat = linear_down.out_features
        
        print(f"down_proj: {in_feat} → {out_feat}")
        print("-"*80)
        
        for config in configs_to_test:
            current += 1
            block_size = config['block_size']
            k = config['k']
            
            try:
                result = benchmark_single_config(
                    linear_down, in_feat, out_feat, block_size, k,
                    batch_size=1, seq_len=128, device=args.device, num_runs=20
                )
                
                result['layer'] = layer_idx
                result['matrix'] = 'down_proj'
                result['block_size'] = block_size
                result['k'] = k
                result['shape'] = (in_feat, out_feat)
                
                status = '✓' if result['speedup'] > 1.0 else '✗'
                print(f"[{current:2d}/{total_tests}] B={block_size:4d} K={k:2d} | "
                      f"L={result['linear_mean']:5.2f}ms F={result['fft_mean']:5.2f}ms | "
                      f"Speedup={result['speedup']:4.2f}x {status} | "
                      f"Comp={result['compression']:4.0f}x")
                
                results.append(result)
                
            except Exception as e:
                print(f"[{current:2d}/{total_tests}] B={block_size:4d} K={k:2d} | Error: {e}")
            
            torch.cuda.empty_cache()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def analyze_results(results):
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    faster_configs = [r for r in results if r['speedup'] > 1.0]
    
    print(f"\nOverall: {len(faster_configs)}/{len(results)} configs achieved speedup")
    
    print("\n1. Best Configs by Layer:")
    print("-"*80)
    layers = sorted(set(r['layer'] for r in results))
    for layer in layers:
        layer_results = [r for r in results if r['layer'] == layer]
        faster = [r for r in layer_results if r['speedup'] > 1.0]
        
        if faster:
            best = max(faster, key=lambda x: x['speedup'])
            print(f"Layer {layer:2d}: B={best['block_size']:4d} K={best['k']:2d} | "
                  f"Speedup={best['speedup']:.2f}x | Compress={best['compression']:.0f}x")
        else:
            print(f"Layer {layer:2d}: No speedup achieved")
    
    print("\n2. Block Size Performance:")
    print("-"*80)
    from collections import defaultdict
    block_speedups = defaultdict(list)
    for r in results:
        block_speedups[r['block_size']].append(r['speedup'])
    
    for block in sorted(block_speedups.keys()):
        speedups = block_speedups[block]
        avg = np.mean(speedups)
        num_faster = sum(1 for s in speedups if s > 1.0)
        print(f"Block {block:4d}: Avg={avg:4.2f}x | {num_faster}/{len(speedups)} faster")
    
    print("\n3. K Value Performance:")
    print("-"*80)
    k_speedups = defaultdict(list)
    for r in results:
        k_speedups[r['k']].append(r['speedup'])
    
    for k in sorted(k_speedups.keys()):
        speedups = k_speedups[k]
        avg = np.mean(speedups)
        num_faster = sum(1 for s in speedups if s > 1.0)
        print(f"K={k:2d}: Avg={avg:4.2f}x | {num_faster}/{len(speedups)} faster")
    
    print("\n4. Layer Depth vs Speedup:")
    print("-"*80)
    layer_groups = {
        'Early (0-10)': [r for r in results if r['layer'] <= 10],
        'Middle (11-25)': [r for r in results if 11 <= r['layer'] <= 25],
        'Late (26-39)': [r for r in results if r['layer'] >= 26]
    }
    
    for group_name, group_results in layer_groups.items():
        if group_results:
            speedups = [r['speedup'] for r in group_results]
            avg = np.mean(speedups)
            num_faster = sum(1 for s in speedups if s > 1.0)
            print(f"{group_name:20s}: Avg={avg:4.2f}x | {num_faster}/{len(group_results)} faster")
    
    if faster_configs:
        print("\n5. Recommended Configurations:")
        print("-"*80)
        
        from collections import Counter
        block_counter = Counter(r['block_size'] for r in faster_configs)
        k_counter = Counter(r['k'] for r in faster_configs)
        
        best_block = block_counter.most_common(1)[0][0]
        best_k = k_counter.most_common(1)[0][0]
        
        print(f"Most successful: block_size={best_block}, K={best_k}")
        
        best_overall = max(faster_configs, key=lambda x: x['speedup'])
        print(f"Best speedup: Layer {best_overall['layer']}, "
              f"B={best_overall['block_size']}, K={best_overall['k']}, "
              f"{best_overall['speedup']:.2f}x")
        
        best_balance = max(faster_configs, 
                          key=lambda x: x['speedup'] * np.log(x['compression']))
        print(f"Best balance: Layer {best_balance['layer']}, "
              f"B={best_balance['block_size']}, K={best_balance['k']}, "
              f"{best_balance['speedup']:.2f}x speedup, {best_balance['compression']:.0f}x compression")

def main():
    args = get_args()
    
    print(f"\nDevice: {args.device}")
    print(f"Model: {args.model_path}")
    
    results = quick_explore(args)
    
    analyze_results(results)
    
    output_file = 'quick_threshold_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()