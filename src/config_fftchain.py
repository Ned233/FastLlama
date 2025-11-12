import argparse

def get_args():
    parser = argparse.ArgumentParser(description='FFTChain Layer-wise Training')
    
    parser.add_argument('--model_path', type=str, default='../model')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda:2')
    
    parser.add_argument('--layer_start', type=int, default=29,)
    parser.add_argument('--layer_end', type=int, default=39,)
    parser.add_argument('--target_matrix', type=str, default='down_proj',choices=['down_proj', 'up_proj', 'gate_proj'])
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--num_fft_matrices', type=int, default=16)
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=500,)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--min_epochs', type=int, default=5,)
    parser.add_argument('--patience', type=int, default=3,)
    
    return parser.parse_args()