"""
第03课：位置编码（Positional Encoding）—— 绝对正余弦 + RoPE 旋转
================================================================
参考：
  - 原论文《Attention Is All You Need》§3.5（绝对正余弦 PE）arXiv:1706.03762
  - RoPE 论文《RoFormer》§3.1-3.3（旋转位置编码）arXiv:2104.09864
  - 3Blue1Brown《Transformer 中的位置编码》可视化讲解

【为什么需要位置编码？】原论文 §3.5 原话：
  "Since our model contains no recurrence and no convolution, in order for the model
   to make use of the order of the sequence, we must inject some information about
   the relative or absolute position."
  即：自注意力本身是『位置无关』的（打乱输入顺序，输出只是对应打乱），
  所以必须显式把『谁先谁后』的信息塞进去。否则 "狗咬人" 和 "人咬狗" 对模型一样。

【两种主流方案对比】
  1. 绝对正余弦 PE（原论文）：把一组 sin/cos 加到 embedding 上（加性）
  2. RoPE 旋转位置编码（现代 LLM 默认）：把 Q/K 旋转一个角度（乘性）
     —— LLaMA / Qwen / DeepSeek 等都用 RoPE

【面试高频考点】
  1. 为什么用 sin/cos 而不学一组位置向量？→ 可外推到更长序列 + 相对位置可线性表示
  2. RoPE 的核心思想？→ 用旋转矩阵编码绝对位置，内积中自然出现相对位置依赖
  3. RoPE 只乘 Q/K，不乘 V！（高频易错点）
  4. RoPE 在每个 2D 子空间用不同频率 θ_i，不是整向量同一角度

运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "01-Transformer/03-位置编码.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 【第一部分】绝对正余弦位置编码（原论文 §3.5）
# ============================================================
def sinusoidal_pe(seq_len, d_model):
    """原论文 §3.5 的正余弦绝对位置编码。

        PE(pos, 2i)   = sin( pos / 10000^{2i/d_model} )
        PE(pos, 2i+1) = cos( pos / 10000^{2i/d_model} )

    返回 (seq_len, d_model) 的 PE 矩阵，加到 embedding 上即可。
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]              # (seq_len, 1)
    # 偶数索引 2i 对应的『频率分母指数项』：10000^{2i/d_model}
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model/2,)
    pe[:, 0::2] = np.sin(position * div_term)           # 偶数维用 sin
    pe[:, 1::2] = np.cos(position * div_term)           # 奇数维用 cos
    return pe


print("=" * 64)
print("【1】绝对正余弦位置编码（原论文 §3.5）")
print("=" * 64)

d_model = 16
seq_len = 6
pe = sinusoidal_pe(seq_len, d_model)
print(f"  PE 矩阵形状: {pe.shape}（{seq_len} 个位置，每个 {d_model} 维）")
print(f"  前 3 个位置的前 6 维：\n{pe[:3, :6]}")
print("  观察规律：低维（左）频率高、变化快；高维（右）频率低、变化慢")
print("  → 像一个多频率的『二进制时钟』，每维记一个粒度的位置信号\n")

# 【1.1】为什么 PE 能让模型『学到相对位置』？—— 原论文 §3.5 的关键性质
print("=" * 64)
print("【1.1】关键性质：PE(pos+δ) 是 PE(pos) 的【线性函数】（可外推+可学相对位置）")
print("=" * 64)
# 对任意固定偏移 δ，PE(pos+δ) = M_δ · PE(pos)（M_δ 是只依赖 δ 的矩阵）
# 直观：sin/cos 的相位平移可用旋转矩阵表示。下面验证一个例子。
d_small = 4
pe_small = sinusoidal_pe(10, d_small)
delta = 3
# 用 PE(pos) 和 PE(pos+delta) 解出线性映射 M（最小二乘），看残差是否≈0
A = pe_small[:-delta]                       # PE(pos)
B = pe_small[delta:]                        # PE(pos+delta)
M, res, *_ = np.linalg.lstsq(A, B, rcond=None)   # B ≈ A · M
recon_err = np.abs(A @ M - B).max()
print(f"  d_model={d_small}, 偏移 δ={delta}")
print(f"  求 M 使 PE(pos+δ) ≈ PE(pos)·M，最大重构误差 = {recon_err:.2e}")
print("  → 误差≈0，证明 PE(pos+δ) 确实可由 PE(pos) 线性表出")
print("  → 这就是原论文『让模型容易学到按相对位置 attend』的数学根基\n")


# ============================================================
# 【第二部分】RoPE 旋转位置编码（现代 LLM 默认，RoPE 论文 §3.2-3.3）
# ============================================================
def precompute_rope_freqs(d_head, max_seq_len, base=10000.0):
    """预计算 RoPE 的频率 θ_i = 10000^{-2i/d}（与原论文 PE 同源）。

    RoPE 把 d 维向量看成 d/2 个 2D 子空间，每子空间按 θ_i 旋转。
    返回 (max_seq_len, d_head/2) 的角度矩阵：angle[m, i] = m · θ_i
    """
    half = d_head // 2
    # θ_i = base^{-2i/d} = base^{-i/half}（i = 0..half-1）
    freqs = 1.0 / (base ** (np.arange(0, half) / half))   # (half,)
    positions = np.arange(max_seq_len)                     # (max_seq_len,)
    angles = np.outer(positions, freqs)                    # (max_seq_len, half)
    return angles


def apply_rope(x, angles):
    """对 x 应用 RoPE 旋转。x: (seq_len, d_head)，angles: (seq_len, d_head/2)。

    实现：把 x 相邻两两分组 → 每组 (x_{2i}, x_{2i+1}) 按 angle 旋转 θ。
    注意：这只乘 Q/K（位置信息要进 QK^T 内积），V 不加位置！（高频易错点）

    论文 §3.2：rotate the affine-transformed word embedding vector by m·θ。
    """
    seq_len, d = x.shape
    half = d // 2
    # 拆成 (x_even, x_odd) 两路，对应 2D 子空间的两个坐标
    x1 = x[:, 0::2]      # (seq_len, half)
    x2 = x[:, 1::2]      # (seq_len, half)
    cos = np.cos(angles)  # (seq_len, half)
    sin = np.sin(angles)
    # 二维旋转：[cos -sin; sin cos] @ [x1; x2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # 交错写回（与拆分对应）
    out = np.empty_like(x)
    out[:, 0::2] = rot1
    out[:, 1::2] = rot2
    return out


print("=" * 64)
print("【2】RoPE 旋转位置编码（RoPE 论文 §3.2-3.3）")
print("=" * 64)

d_head = 16                                 # 单头维度（论文 d_k=64，这里取 16 便于打印）
max_len = 16
angles = precompute_rope_freqs(d_head, max_len)
print(f"  频率 θ_i = 10000^(-2i/d)，d_head={d_head}")
print(f"  θ = {1.0/(10000**(np.arange(0,d_head//2)/(d_head//2)))}")
print(f"  角度矩阵形状: {angles.shape}（angle[m,i] = m·θ_i）")

# 【2.1】RoPE 的核心性质：q_m · k_n 的内积【只依赖相对位置 m−n】
print("-" * 64)
print("【2.1】核心性质验证：RoPE 后 ⟨q_m, k_n⟩ 只依赖相对位置 m−n")
print("-" * 64)
np.random.seed(1)
q0 = np.random.randn(d_head)                # 一个 query 的内容（与位置无关）
k0 = np.random.randn(d_head)                # 一个 key 的内容
# 构造 q 在位置 m、k 在位置 n，内积应只随 (m-n) 变
pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (5, 5)]   # 相对距离都是 0
pairs_diff3 = [(3, 0), (5, 2), (8, 5), (10, 7)]   # 相对距离都是 3

def rope_dot(q_vec, k_vec, m, n):
    """q 在位置 m、k 在位置 n 的内积（应用 RoPE 后）。"""
    q_rot = apply_rope(q_vec[None, :], angles[m:m+1])[0]
    k_rot = apply_rope(k_vec[None, :], angles[n:n+1])[0]
    return float(q_rot @ k_rot)

d0 = [rope_dot(q0, k0, m, n) for m, n in pairs]
d3 = [rope_dot(q0, k0, m, n) for m, n in pairs_diff3]
print(f"  相对距离=0 的几组 (m,n) 内积: {[f'{v:.3f}' for v in d0]}")
print(f"  相对距离=3 的几组 (m,n) 内积: {[f'{v:.3f}' for v in d3]}")
print(f"  距离0组 标准差={np.std(d0):.2e}  距离3组 标准差={np.std(d3):.2e}")
print("  → 相同相对距离的内积几乎相同！证明 RoPE 把『绝对位置』编码后，")
print("    内积里自然只剩下『相对位置 m−n』。这正是 RoPE 论文摘要的核心卖点\n")

# 【2.2】长期衰减：内积随相对距离增大而衰减（RoPE §3.3）
# 说明：单条曲线在小维度下会振荡，RoPE 的『长期衰减』是统计意义（对随机 q/k 平均）下的性质。
# 这里用 Monte-Carlo：对很多组随机 q/k 求平均，让衰减趋势清晰呈现（论文 §3.3 也是统计性质）。
print("-" * 64)
print("【2.2】长期衰减：相对距离越大，注意力内积越小（RoPE §3.3，统计意义）")
print("-" * 64)
n_trials = 2000
rng_decay = np.random.default_rng(7)
decay_curve = []
for delta in range(0, max_len):
    vals = []
    for _ in range(n_trials):
        qv = rng_decay.standard_normal(d_head)
        kv = rng_decay.standard_normal(d_head)
        vals.append(rope_dot(qv, kv, delta, 0))
    decay_curve.append(np.mean(np.abs(vals)))   # 用 |内积| 的均值，衰减更直观
decay_curve = np.array(decay_curve) / decay_curve[0]
print("  相对距离 → 平均|内积|（对随机 q/k 2000 次平均，归一化）:")
for delta, val in enumerate(decay_curve):
    bar = '█' * int(val * 30)
    print(f"    m-n={delta:>2}: {val:.3f}  {bar}")
print("  → 统计平均后，内积随相对距离单调衰减（远处 token 联系更弱），符合 RoPE §3.3\n")


# ============================================================
# 【3】可视化 1：正余弦 PE 矩阵（多频率时钟）
# ============================================================
pe_vis = sinusoidal_pe(50, d_model)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

ax = axes[0]
im = ax.imshow(pe_vis, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_xlabel('维度索引（频率从高到低）')
ax.set_ylabel('位置 pos')
ax.set_title('绝对正余弦位置编码 PE 矩阵\n（低维高频、高维低频，像多频率时钟）',
             fontsize=11, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1]
pos = np.arange(50)
for dim in [0, 2, 6, 14]:
    ax.plot(pos, pe_vis[:, dim], label=f'维度{dim}（频率={1/(10000**(dim/d_model)):.4f}）', lw=1.8)
ax.set_xlabel('位置 pos')
ax.set_ylabel('PE 值')
ax.set_title('不同维度的 sin/cos 波形\n低维=高频（细粒度定位），高维=低频（粗粒度定位）',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "03-正余弦PE.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=110, bbox_inches='tight')
plt.close()
print(f"【3】图1 已保存：{out1}")


# ============================================================
# 【4】可视化 2：RoPE 的相对位置内积衰减曲线（统计平均）
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 4.5))
dists_full = np.arange(0, max_len)
ax.plot(dists_full, decay_curve, 'o-', color='#C44E52', lw=2.2, markersize=6,
        label='平均 |⟨RoPE(q_m), RoPE(k_0)⟩|（2000 次随机 q/k）')
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('相对位置 m − n')
ax.set_ylabel('平均 |内积|（归一化）')
ax.set_title('RoPE 的『长期衰减』性质：相对距离越大，token 间联系越弱\n（统计平均；依据 RoPE 论文 §3.3）',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3); ax.legend()
ax.annotate('近处强相关', xy=(0, decay_curve[0]), xytext=(2, decay_curve[0]*0.9),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
plt.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "03-RoPE衰减.png")
plt.savefig(out2, dpi=110, bbox_inches='tight')
plt.close()
print(f"【4】图2 已保存：{out2}\n")


# ============================================================
# 【5】连回 AI + 要点总结
# ============================================================
print("=" * 64)
print("✅ 第03课要点：位置编码")
print("=" * 64)
print("【为什么需要】自注意力本身位置无关（permutation equivariant），必须注入位置信息")
print()
print("【绝对正余弦 PE（原论文）】")
print("  • PE(pos,2i)=sin(pos/10000^{2i/d}), PE(pos,2i+1)=cos(...)")
print("  • 加在 embedding 上（加性！RoPE 才是乘性）")
print("  • 选 sin/cos 的理由：可外推到更长序列 + PE(pos+δ) 可由 PE(pos) 线性表出")
print("    （消融：学习式 PE 结果几乎相同，但外推差，所以选 sin/cos）")
print()
print("【RoPE（现代 LLM 默认）】")
print("  • 用旋转矩阵编码绝对位置，内积中自然出现相对位置依赖 m−n")
print("  • 把 d 维拆成 d/2 个 2D 子空间，每子空间按 θ_i=10000^{-2i/d} 旋转")
print("  • 三个性质：可外推、长期衰减、乘性注入")
print("  • ⚠️ 只乘 Q/K，不乘 V！只乘 Q/K，不乘 V！（高频易错）")
print()
print("🎯 AI 里的应用：")
print("  • 原论文/早期模型（BERT/GPT-2）：正余弦或学习式 PE")
print("  • 现代主流 LLM（LLaMA/Qwen/DeepSeek/Mistral）：清一色 RoPE")
print("  • 长上下文外推：NTK-aware / YaRN 都是对 RoPE 频率 base 的改造")
print("=" * 64)
