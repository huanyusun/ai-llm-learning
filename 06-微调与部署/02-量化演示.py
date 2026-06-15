"""
微调与部署 第2课：量化原理演示 INT8 / INT4（纯 numpy）
========================================================
参考论文：
  QLoRA: Efficient Finetuning of Quantized LLMs
  Dettmers et al., 2023, arXiv:2305.14314
  LLM.int8(): 8-bit Matrix Multiplication (arXiv:2208.07339)
  AWQ: Activation-aware Weight Quantization (arXiv:2306.00978)
  GPTQ: Accurate PTQ (arXiv:2210.17323)

核心一句话：量化 = 把 32/16-bit 浮点权重，映射到 8/4-bit 整数，
  【存储】直接按比特数缩小（FP32→INT8 省 4×，→INT4 省 8×），
  【精度】代价是 round-to-nearest 引入的小误差。

本文件做什么（全部 numpy 可跑）：
  【1】FP32 → INT8 对称量化 + 反量化，看误差
  【2】FP32 → INT4 量化，看 16 个桶的「粒度」有多粗
  【3】outlier（离群值）问题：朴素量化会被极端值带偏
  【4】per-channel vs per-tensor：分组粒度如何降误差
  【5】可视化：量化前后分布 + 误差

运行：uv run python "06-微调与部署/02-量化演示.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(7)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 【0】核心函数：对称量化 / 反量化
# ============================================================
def quantize_symmetric(W, bits=8):
    """
    对称量化：把浮点权重映射到 [-2^(b-1)+1, 2^(b-1)-1] 整数。
    步骤（arXiv:2305.14314 §2 朴素量化）：
      scale = max(|W|) / qmax          # 一个浮点值代表多少「整数刻度」
      W_int = round(W / scale)         # 量化到整数
      W_int = clip(W_int, -qmax, qmax)
    返回：W_int(整数), scale
    """
    qmax = (1 << (bits - 1)) - 1       # 8-bit→127, 4-bit→7
    scale = np.max(np.abs(W)) / qmax
    if scale == 0:
        scale = 1e-12
    W_int = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int32)
    return W_int, scale

def dequantize(W_int, scale):
    """反量化：W_approx = W_int * scale。"""
    return W_int.astype(np.float32) * scale

def quant_dequant(W, bits=8):
    """量化再反量化（模拟推理时的实际权重），返回近似权重。"""
    W_int, scale = quantize_symmetric(W, bits)
    return dequantize(W_int, scale)

def rel_error(W, W_approx):
    """相对误差：||W-W_approx|| / ||W||。"""
    return np.linalg.norm(W - W_approx) / (np.linalg.norm(W) + 1e-12)


# ============================================================
# 【1】INT8 量化：存储省 4×，精度几乎无损
# ============================================================
print("=" * 60)
print("【1】FP32 → INT8：存储省 4×，精度几乎无损")
print("=" * 60)

# 模拟一个 LLM 线性层的权重：零均值正态分布（论文里权重的典型分布）
d, k = 512, 512
W = rng.standard_normal((d, k)).astype(np.float32) * 0.05

W_int8, scale8 = quantize_symmetric(W, bits=8)
W_approx8 = dequantize(W_int8, scale8)

size_fp32 = W.nbytes                       # float32 每元素 4 字节
size_int8 = W_int8.size * 1                # int8 每元素 1 字节
print(f"  权重形状 {W.shape}，FP32 占 {size_fp32/1024:.1f} KB")
print(f"  INT8 占 {size_int8/1024:.1f} KB  → 存储降到 {size_int8/size_fp32*100:.0f}%（省 {1-size_int8/size_fp32:.0%}）")
print(f"  量化误差 ||Δ||/||W|| = {rel_error(W, W_approx8):.4e}  （≈ 浮点噪声级）")
print(f"  INT8 取值范围：[{W_int8.min()}, {W_int8.max()}]，scale = {scale8:.6f}")


# ============================================================
# 【2】INT4 量化：存储再省 2×，但只有 15 个刻度
# ============================================================
print()
print("=" * 60)
print("【2】FP32 → INT4：存储省 8×，但粒度变粗")
print("=" * 60)

W_int4, scale4 = quantize_symmetric(W, bits=4)
W_approx4 = dequantize(W_int4, scale4)

size_int4 = W_int4.size * 0.5              # int4 每元素 0.5 字节
print(f"  INT4 占 {size_int4/1024:.1f} KB  → 存储降到 FP32 的 {size_int4/size_fp32*100:.0f}%（省 {1-size_int4/size_fp32:.0%}）")
print(f"  INT4 取值范围：[{W_int4.min()}, {W_int4.max()}]（只有 {len(np.unique(W_int4))} 个不同刻度）")
print(f"  量化误差 ||Δ||/||W|| = {rel_error(W, W_approx4):.4e}  （比 INT8 大，但仍很小）")
print()
print("  关键直觉：权重本身是「平滑」的正态分布，")
print("  用很少的刻度就能很好地近似 → 这正是 LLM 量化的可行性根基。")


# ============================================================
# 【3】离群值（outlier）问题：朴素量化的头号杀手
# ============================================================
print()
print("=" * 60)
print("【3】离群值陷阱：一个极端值带偏整层")
print("=" * 60)

# 给 W 注入几个「离群值」（模拟 LLM.int8() 发现的极端激活/权重列）
W_outlier = W.copy()
W_outlier[0, 0] = 2.0                      # 一个比正常值大 ~40× 的离群点

W8_no = quant_dequant(W, bits=8)
W8_out = quant_dequant(W_outlier, bits=8)
print(f"  无离群值时 INT8 误差 = {rel_error(W, W8_no):.4e}")
print(f"  有 1 个离群值时 INT8 误差 = {rel_error(W_outlier, W8_out):.4e}  （显著放大！）")
print()
print("  原因：scale = max|W|/127，离群值把 scale 撑大，")
print("        导致其余正常值都被压缩到几个刻度上 → 精度崩塌。")
print("  LLM.int8() 的解法：离群列保持 FP16，其余 INT8（混合精度）。")


# ============================================================
# 【4】per-channel vs per-tensor：分组粒度
# ============================================================
print()
print("=" * 60)
print("【4】per-tensor vs per-channel：把 scale 拆细，误差骤降")
print("=" * 60)

def quant_dequant_per_channel(W, bits=8, axis=0):
    """按列（axis=0）各自算 scale：每列有自己的动态范围。"""
    qmax = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(W), axis=axis, keepdims=True) / qmax
    scale = np.where(scale == 0, 1e-12, scale)
    W_int = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int32)
    return W_int * scale

W4_tensor = quant_dequant(W, bits=4)
W4_channel = quant_dequant_per_channel(W, bits=4, axis=0)
print(f"  INT4 per-tensor  误差 = {rel_error(W, W4_tensor):.4e}")
print(f"  INT4 per-channel 误差 = {rel_error(W, W4_channel):.4e}  （降了好几倍）")
print()
print("  AWQ/GPTQ 本质都在这之上：per-channel + 保护「显著通道」，")
print("  在 4-bit 下做到几乎无损。")


# ============================================================
# 【5】大模型层面估算：7B 模型量化的存储收益
# ============================================================
print()
print("=" * 60)
print("【5】真实尺度：7B 模型各精度的显存")
print("=" * 60)
params_7b = 7e9
for name, bits_per in [("FP32", 32), ("FP16/BF16", 16), ("INT8", 8), ("INT4", 4)]:
    gb = params_7b * bits_per / 8 / 1024**3
    print(f"  {name:<10} → {gb:6.2f} GB")
print()
print("  这就是为什么 QLoRA 能在单 48GB 卡微调 65B：4-bit 把基座压到 ~1/4。")


# ============================================================
# 【6】可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# 左：权重分布直方图 —— FP32 vs INT8 vs INT4
ax = axes[0]
sample = W.flatten()
ax.hist(sample, bins=80, alpha=0.5, density=True, label='FP32 原始')
ax.hist(W_approx8.flatten(), bins=80, alpha=0.5, density=True, label='INT8 反量化')
ax.hist(W_approx4.flatten(), bins=30, alpha=0.5, density=True, label='INT4 反量化')
ax.set_title('权重分布：量化前后\n(INT4 刻度少 → 分布更"离散")')
ax.set_xlabel('权重值'); ax.set_ylabel('密度'); ax.legend(fontsize=9)

# 中：量化误差 vs 比特数
ax = axes[1]
bits_list = [2, 3, 4, 5, 6, 8, 16]
errs = [rel_error(W, quant_dequant(W, b)) for b in bits_list]
ax.semilogy(bits_list, errs, 'o-', color='#d62728')
ax.set_xlabel('量化比特数'); ax.set_ylabel('相对误差 ||Δ||/||W||')
ax.set_title('量化误差随比特数下降\n(比特越多 → 误差指数下降)')
ax.grid(alpha=0.3)

# 右：离群值破坏 vs per-channel 修复
ax = axes[2]
labels = ['无离群\nINT8', '有离群\nINT8\n(朴素)', '有离群\nINT8\n(per-channel)']
W_oc = quant_dequant_per_channel(W_outlier, bits=8, axis=0)
vals = [rel_error(W, W8_no),
        rel_error(W_outlier, W8_out),
        rel_error(W_outlier, W_oc)]
bars = ax.bar(labels, vals, color=['#2ca02c', '#d62728', '#1f77b4'])
ax.set_ylabel('相对误差')
ax.set_title('离群值的破坏 + per-channel 的修复')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1e}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out = os.path.join(OUT_DIR, "02-量化演示.png")
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"\n【6】图已保存：{out}")


# ============================================================
# 要点小结
# ============================================================
print()
print("=" * 60)
print("✅ 量化要点（连回面试）")
print("=" * 60)
print("  • 量化 = 浮点→整数：scale = max|W|/qmax，W_int = round(W/scale)")
print("  • 收益：FP32→INT8 省 4×，→INT4 省 8× 存储/显存")
print("  • 代价：round 引入小误差，权重平滑分布时几乎无损")
print("  • 离群值是头号杀手 → LLM.int8() 混合精度保留离群列")
print("  • per-channel 比 per-tensor 误差小很多（每列独立 scale）")
print("  • NF4(QLoRA)：对正态分布信息论最优的 4-bit 数据类型")
print("  • GPTQ：逐列量化 + Hessian 二阶补偿，一次性 PTQ")
print("  • AWQ：保护 ~1% 按激活幅度判定的显著权重，最硬件友好")
print("  • GGUF k-quants：分块混合精度，CPU/Mac 端侧主流")
print("=" * 60)
