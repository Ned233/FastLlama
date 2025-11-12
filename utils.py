import torch
import torch.nn as nn
from fftchain import FFTChainMatrix

def replace_linear_with_fftchain(model, layer_indices, target_matrix, block_size, num_fft_matrices):
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers (indices 0-{num_layers-1})")
    
    replaced_modules = []
    
    for layer_idx in layer_indices:
        if layer_idx >= num_layers:
            print(f"layer {layer_idx} exceeds model layers, skipping")
            continue
        
        layer = model.model.layers[layer_idx]
        
        if target_matrix == 'down_proj':
            original = layer.mlp.down_proj
        elif target_matrix == 'up_proj':
            original = layer.mlp.up_proj
        elif target_matrix == 'gate_proj':
            original = layer.mlp.gate_proj
        else:
            raise ValueError(f"Unknown target_matrix: {target_matrix}")
        
        in_features = original.in_features
        out_features = original.out_features
        
        fft_module = FFTChainMatrix(
            in_features=in_features,
            out_features=out_features,
            block_size=block_size,
            num_fft_matrices=num_fft_matrices
        )
        
        if target_matrix == 'down_proj':
            layer.mlp.down_proj = fft_module
        elif target_matrix == 'up_proj':
            layer.mlp.up_proj = fft_module
        elif target_matrix == 'gate_proj':
            layer.mlp.gate_proj = fft_module
        
        replaced_modules.append(fft_module)
        print(f"[Replace] Layer {layer_idx} {target_matrix}: {in_features} -> {out_features}")
    
    print(f"[Replace] Replaced {len(replaced_modules)} modules")
    return replaced_modules

def save_checkpoint(model, optimizer, epoch, path, config=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"[Save] Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path, device='cuda'):
    """加载checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型参数
    model_state = checkpoint['model_state_dict']
    model.load_state_dict(model_state, strict=False)
    
    # 加载优化器参数（如果提供）
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 统计加载的参数
    loaded_params = sum(1 for k in model_state.keys() if any(x in k for x in ['circulant', 'diagonal', 'channel_weights']))
    total_params = len(model_state)
    
    print(f"[Load] Loaded {loaded_params} FFTChain parameters (total: {total_params})")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        if 'trained_epochs' in config and 'best_loss' in config:
            print(f"[Load] Config: layers {config['layer_start']}-{config['layer_end']}, "
                  f"trained {config['trained_epochs']} epochs, best loss: {config['best_loss']:.4f}")
    
    return checkpoint