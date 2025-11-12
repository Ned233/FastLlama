### 1. quick_explore_threshold.py (推荐先用)

**快速版本**，测试关键配置组合，约15-20分钟完成。

#### 运行命令

python quick_explore_threshold.py --device cuda:2


#### 测试配置
- 测试层: [0, 10, 19, 25, 30, 35, 39] - 覆盖前中后层
- Block sizes: [256, 512, 1024, 2048]
- K values: [2, 4, 8, 16]
- 8个精选配置组合
- 每个配置20次运行取平均

#### 输出示例
================================================================================
LAYER 19
================================================================================
down_proj: 13824 → 5120
--------------------------------------------------------------------------------
[ 1/56] B=256  K=2  | L=30.45ms F=35.21ms | Speedup=0.86x ✗ | Comp=543x
[ 2/56] B=512  K=2  | L=30.12ms F=32.10ms | Speedup=0.94x ✗ | Comp=271x
[ 3/56] B=512  K=4  | L=30.34ms F=28.67ms | Speedup=1.06x ✓ | Comp=136x
[ 4/56] B=1024 K=2  | L=30.89ms F=26.43ms | Speedup=1.17x ✓ | Comp=135x
[ 5/56] B=1024 K=4  | L=31.02ms F=27.11ms | Speedup=1.14x ✓ | Comp=68x

### 2. explore_fft_threshold.py

**完整版本**，测试所有配置，约1-2小时完成。

#### 运行命令

python explore_fft_threshold.py --device cuda:2

#### 测试配置
- 测试层: [0, 10, 19, 30, 39]
- 测试矩阵: down_proj, up_proj, gate_proj
- Block sizes: [256, 512, 1024, 2048]
- K values: [2, 4, 8, 16]
- 完整的16种组合
- 每个配置30次运行

## 关键分析维度

### 1. Block Size分析
探索不同block size对加速的影响：
- 小block (256, 512): FFT优势小，kernel开销占比高
- 中block (1024): 平衡点
- 大block (2048): FFT优势大，但padding开销增加

### 2. K值分析
探索多通道数量的影响：
- 小K (2, 4): 参数少，压缩比高，但表达能力弱
- 大K (8, 16): 表达能力强，但参数增加，可能慢

### 3. 层深度分析
不同层的最优配置可能不同：
- Early layers (0-10): 可能需要更复杂配置
- Middle layers (11-25): 过渡层
- Late layers (26-39): 可能用简单配置即可

### 4. 矩阵类型分析
- down_proj: 13824 → 5120 (压缩)
- up_proj: 5120 → 13824 (扩展)
- gate_proj: 5120 → 13824 (扩展)

## 预期发现

### 假设1: 存在加速阈值

加速条件: (block_size × K) 需要满足某个范围
- 太小: FFT优势不明显
- 太大: 参数过多，反而变慢


### 假设2: 层特异性

不同层最优配置不同：
- 浅层: 需要更多参数 (大K)
- 深层: 简单配置即可 (小K)


### 假设3: 矩阵形状影响

- 压缩矩阵 (m < n): block_size应该基于较小维度
- 扩展矩阵 (m > n): 可能需要不同策略


## 结果解读

### 输出文件
- quick_threshold_results.json: 详细数据
- 控制台输出: 分析报告

### 关键指标

#### 1. Speedup
- \> 1.0: FFT更快 ✓
- < 1.0: Linear更快 ✗
- 1.15x: 你当前的结果 (很好!)

#### 2. Compression Ratio

compression = linear_params / fft_params

高压缩 (>100x): 显存节省多
低压缩 (<50x): 可能不值得


#### 3. Success Rate

某配置在多少层上实现加速

例: block_size=1024, K=4 在 15/21 层加速
→ 这是一个稳健的配置


## 使用场景

### Scenario 1: 快速验证

python quick_explore_threshold.py --device cuda:2


### Scenario 2: 深入研究

python explore_fft_threshold.py --device cuda:2


### Scenario 3: 自定义测试
修改代码中的配置：

python
test_configs = {
    'block_sizes': [512, 1024, 1536],  # 自定义
    'k_values': [3, 5, 7],              # 自定义
    'batch_size': 4,                    # 测试更大batch
    'seq_len': 256                      # 测试更长序列
}


## 实际应用

基于探索结果，你可以：

### 1. 为每层选择最优配置
python
layer_configs = {
    'layer_0_10': {'block_size': 512, 'k': 4},
    'layer_11_20': {'block_size': 1024, 'k': 4},
    'layer_21_39': {'block_size': 1024, 'k': 2}
}


### 2. 分层训练策略
```bash
# 上层用简单配置
python train_fftchain.py --layer_start 30 --layer_end 39 \
  --block_size 1024 --num_fft_matrices 2

# 中层用平衡配置
python train_fftchain.py --layer_start 15 --layer_end 29 \
  --block_size 1024 --num_fft_matrices 4

# 下层用复杂配置(如果需要)
python train_fftchain.py --layer_start 0 --layer_end 14 \
  --block_size 512 --num_fft_matrices 8
```

### 3. 选择性替换
只替换有加速效果的层：
```python
# 基于探索结果
speedup_layers = [19, 20, 25, 30, 35, 39]  # 只替换这些层
```

## 故障排除

### 问题1: CUDA OOM
```bash
# 减小测试范围
test_layers = [19, 30, 39]  # 只测试3层

# 或减少runs
num_runs = 10  # 降到10次
```

### 问题2: 结果不稳定
```bash
# 增加warmup
warmup = 10

# 增加runs
num_runs = 50
```

### 问题3: 所有配置都变慢
可能原因:
1. GPU频率被锁定在低频
2. 其他进程占用GPU
3. 模型太小，kernel开销占主导

## 理论背景

### FFT加速的数学条件

对于 m×n 矩阵，FFT加速条件:
```
T_FFT < T_GEMM

其中:
T_FFT = T_kernel_launch × N_kernels + 
        T_FFT_compute + T_memory

T_GEMM = T_GEMM_compute + T_memory_GEMM

关键: 当 (FFT计算减少) > (多kernel开销 + 额外内存) 时加速
```

### Block Size影响
```
FFT: O(B log B)
Dense: O(B²)

临界点约在 B ≈ 256-512 之间
```

### 参数量权衡
```
Linear params: m × n
FFT params: K × (m/B) × (n/B) × B

当 K × (m/B) × (n/B) × B << m × n 时，
内存带宽节省 >> kernel开销
```

## 下一步

1. 运行快速探索 → 了解总体趋势
2. 如果发现有趣模式 → 运行完整探索
3. 根据结果调整训练策略
4. 在实际模型上验证最优配置