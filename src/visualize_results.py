import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_threshold_results(json_file):
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("No results to visualize")
        return
    
    fig = plt.figure(figsize=(20, 12))
    
    layers = sorted(set(r['layer'] for r in results))
    block_sizes = sorted(set(r['block_size'] for r in results))
    k_values = sorted(set(r['k'] for r in results))
    
    speedup_matrix = np.zeros((len(block_sizes), len(k_values)))
    speedup_counts = np.zeros((len(block_sizes), len(k_values)))
    
    for r in results:
        b_idx = block_sizes.index(r['block_size'])
        k_idx = k_values.index(r['k'])
        speedup_matrix[b_idx, k_idx] += r['speedup']
        speedup_counts[b_idx, k_idx] += 1
    
    speedup_matrix = np.divide(speedup_matrix, speedup_counts, 
                               where=speedup_counts!=0, 
                               out=np.zeros_like(speedup_matrix))
    
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(speedup_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                xticklabels=k_values, yticklabels=block_sizes, ax=ax1,
                cbar_kws={'label': 'Avg Speedup'})
    ax1.set_xlabel('K Value', fontweight='bold')
    ax1.set_ylabel('Block Size', fontweight='bold')
    ax1.set_title('Average Speedup Heatmap\n(across all layers)', fontweight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    for layer in layers:
        layer_results = [r for r in results if r['layer'] == layer]
        speedups = [r['speedup'] for r in layer_results]
        ax2.plot(range(len(speedups)), sorted(speedups, reverse=True), 
                marker='o', label=f'Layer {layer}', alpha=0.7)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('Configuration Rank', fontweight='bold')
    ax2.set_ylabel('Speedup', fontweight='bold')
    ax2.set_title('Speedup Distribution by Layer', fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    block_avg = {}
    for b in block_sizes:
        b_results = [r['speedup'] for r in results if r['block_size'] == b]
        block_avg[b] = np.mean(b_results)
    
    colors = ['red' if v < 1.0 else 'green' for v in block_avg.values()]
    bars = ax3.bar(range(len(block_avg)), list(block_avg.values()), 
                   color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(block_avg)))
    ax3.set_xticklabels([str(k) for k in block_avg.keys()])
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Block Size', fontweight='bold')
    ax3.set_ylabel('Average Speedup', fontweight='bold')
    ax3.set_title('Block Size Performance', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    ax4 = plt.subplot(2, 3, 4)
    k_avg = {}
    for k in k_values:
        k_results = [r['speedup'] for r in results if r['k'] == k]
        k_avg[k] = np.mean(k_results)
    
    colors = ['red' if v < 1.0 else 'green' for v in k_avg.values()]
    bars = ax4.bar(range(len(k_avg)), list(k_avg.values()), 
                   color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(k_avg)))
    ax4.set_xticklabels([str(k) for k in k_avg.keys()])
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('K Value', fontweight='bold')
    ax4.set_ylabel('Average Speedup', fontweight='bold')
    ax4.set_title('K Value Performance', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    ax5 = plt.subplot(2, 3, 5)
    compressions = [r['compression'] for r in results]
    speedups = [r['speedup'] for r in results]
    colors = ['green' if s > 1.0 else 'red' for s in speedups]
    
    ax5.scatter(compressions, speedups, c=colors, alpha=0.6, s=50, edgecolors='black')
    ax5.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax5.set_xlabel('Compression Ratio', fontweight='bold')
    ax5.set_ylabel('Speedup', fontweight='bold')
    ax5.set_title('Compression vs Speedup', fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    layer_avg = {}
    for layer in layers:
        layer_results = [r['speedup'] for r in results if r['layer'] == layer]
        layer_avg[layer] = np.mean(layer_results)
    
    colors = ['red' if v < 1.0 else 'green' for v in layer_avg.values()]
    bars = ax6.bar(range(len(layer_avg)), list(layer_avg.values()), 
                   color=colors, alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(layer_avg)))
    ax6.set_xticklabels([str(k) for k in layer_avg.keys()])
    ax6.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax6.set_xlabel('Layer Index', fontweight='bold')
    ax6.set_ylabel('Average Speedup', fontweight='bold')
    ax6.set_title('Layer-wise Performance', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = json_file.replace('.json', '_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()
    
    fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    configs = [(b, k) for b in block_sizes for k in k_values]
    top_configs = sorted(configs, key=lambda c: np.mean([r['speedup'] for r in results 
                        if r['block_size']==c[0] and r['k']==c[1]]), reverse=True)[:8]
    
    for idx, (block, k) in enumerate(top_configs):
        ax = axes[idx]
        config_results = [r for r in results if r['block_size']==block and r['k']==k]
        
        if config_results:
            layer_speedups = {r['layer']: r['speedup'] for r in config_results}
            layers_plot = sorted(layer_speedups.keys())
            speedups_plot = [layer_speedups[l] for l in layers_plot]
            
            colors = ['green' if s > 1.0 else 'red' for s in speedups_plot]
            bars = ax.bar(range(len(layers_plot)), speedups_plot, color=colors, 
                         alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(layers_plot)))
            ax.set_xticklabels(layers_plot)
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Layer', fontweight='bold')
            ax.set_ylabel('Speedup', fontweight='bold')
            ax.set_title(f'B={block}, K={k}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            avg_speedup = np.mean(speedups_plot)
            ax.text(0.95, 0.95, f'Avg: {avg_speedup:.2f}x', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontweight='bold')
    
    plt.tight_layout()
    output_file2 = json_file.replace('.json', '_detailed.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Detailed visualization saved to: {output_file2}")
    plt.close()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'quick_threshold_results.json'
    
    print(f"Visualizing results from: {json_file}")
    visualize_threshold_results(json_file)
    print("Done!")