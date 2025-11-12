import torch
import torch.nn as nn
import math

class FFTChainMatrix(nn.Module):
    def __init__(self, in_features, out_features, block_size, num_fft_matrices, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_fft_matrices = num_fft_matrices
        self.dtype = dtype
        
        # 自动计算padding后的维度
        self.padded_in_features = ((in_features + block_size - 1) // block_size) * block_size
        self.padded_out_features = ((out_features + block_size - 1) // block_size) * block_size
        
        self.in_padding = self.padded_in_features - in_features
        self.out_padding = self.padded_out_features - out_features
        
        self.num_in_blocks = self.padded_in_features // block_size
        self.num_out_blocks = self.padded_out_features // block_size
        
        scale = 1.0 / math.sqrt(block_size * self.num_in_blocks)
        
        self.circulant_params = nn.Parameter(
            (torch.randn(num_fft_matrices, self.num_out_blocks, self.num_in_blocks, block_size) * scale).to(dtype)
        )
        
        self.channel_weights = nn.Parameter(
            (torch.ones(num_fft_matrices) / num_fft_matrices).to(dtype)
        )
        
        if self.in_padding > 0 or self.out_padding > 0:
            print(f"[FFTChain] Padding: in {in_features}->{self.padded_in_features}, "
                  f"out {out_features}->{self.padded_out_features}")
        
    def forward(self, x):
        batch_size, seq_len, in_feat = x.shape
        assert in_feat == self.in_features
        
        original_dtype = x.dtype
        
        # Pad input if needed
        if self.in_padding > 0:
            x = torch.nn.functional.pad(x, (0, self.in_padding), mode='constant', value=0)
        
        x_blocks = x.view(batch_size, seq_len, self.num_in_blocks, self.block_size).to(self.dtype)
        
        x_fft = torch.fft.rfft(x_blocks, dim=-1)
        x_fft = x_fft.unsqueeze(2).unsqueeze(2)
        
        circ_fft = torch.fft.rfft(self.circulant_params, dim=-1)
        circ_fft = circ_fft.unsqueeze(0).unsqueeze(0)
        
        result_fft = x_fft * circ_fft
        
        result_fft = result_fft.sum(dim=4)
        
        result_fft_weighted = (result_fft * self.channel_weights.view(1, 1, -1, 1, 1)).sum(dim=2)
        
        output = torch.fft.irfft(result_fft_weighted, n=self.block_size, dim=-1)
        
        output = output.view(batch_size, seq_len, self.padded_out_features)
        
        # Remove padding from output
        if self.out_padding > 0:
            output = output[..., :-self.out_padding]
        
        return output.to(original_dtype)