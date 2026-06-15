"""
第04课：Mini Transformer Decoder Block（从零手写）
===================================================
参考：
  - 原论文《Attention Is All You Need》§3.1（残差+LayerNorm）、§3.2.3（causal mask）、§3.3（FFN）
  - 3Blue1Brown《But what is a GPT?》《Transformer 注意力机制》可视化
  - 本模块前 3 课：02-多头注意力.py、03-位置编码.py

【这节课做什么】把前几课的零件拼成一个完整的 Transformer Decoder Block：
    输入 x → [+ 位置编码]
          → 子层1：Masked Multi-Head Self-Attention（带因果掩码）
                    + 残差连接 + LayerNorm
          → 子层2：Position-wise FFN（两层线性 + ReLU）
                    + 残差连接 + LayerNorm
          → 输出（形状与输入相同，可堆叠下一层）

    公式（Post-LN，原论文 §3.1）：
        z1 = LayerNorm( x + MaskedMHA(x) )
        z2 = LayerNorm( z1 + FFN(z1) )

【为什么是 Decoder Block？】现代主流大模型（GPT/LLaMA/Qwen）都是 Decoder-only，
    核心就是『带因果掩码的自注意力』——位置 i 只能看 ≤i 的位置，保证自回归生成。
    本课实现的就是一个最小的 Decoder-only block（不含 cross-attention）。

【面试高频考点】
  1. causal mask 怎么实现？→ softmax 输入中未来位置填 -∞（这里用 -1e9 近似）
  2. 残差连接为什么需要？→ 缓解深层梯度消失，给信息一条『捷径』
  3. LayerNorm 归一化的是【特征维】（每个 token 的向量），不是 batch 维！
  4. Post-LN vs Pre-LN：原论文是 Post-LN（需 warmup），现代大模型多用 Pre-LN（更稳）
  5. FFN 为什么升维 4×？→ 先投影到高维引入非线性，再压回 d_model

运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "01-Transformer/04-mini_transformer_block.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 基础工具：softmax（数值稳定）
# ============================================================
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================
# 零件 1：Layer Normalization（对【特征维】归一化，原论文 §3.1）
# ============================================================
class LayerNorm:
    """对每个 token 的特征向量做归一化（跨 d_model 维，不跨 batch/序列维）。

    LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β
    - μ, σ² 沿特征维（最后一维）计算
    - γ, β 是可学习的缩放和偏移（这里用随机初始化演示）
    注意：BatchNorm 是跨 batch 归一化，LayerNorm 是跨特征——面试常考区别！
    """

    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)   # 缩放（论文里是可学习参数）
        self.beta = np.zeros(d_model)   # 偏移
        self.eps = eps

    def forward(self, x):
        # x: (seq_len, d_model)，沿最后一维（特征维）算均值方差
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ============================================================
# 零件 2：Masked Multi-Head Self-Attention（02 课的实现 + causal mask）
# ============================================================
class MaskedMultiHeadAttention:
    """带因果掩码的多头自注意力（Decoder 核心组件）。

    与 02-多头注意力.py 的区别：传一个下三角 mask，让位置 i 只能 attend ≤i 的位置。
    原论文 §3.2.3：masking out (setting to −∞) all values in the input of the softmax
    which correspond to illegal connections.
    """

    def __init__(self, d_model, h, seed=0):
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        rng = np.random.default_rng(seed)
        # 把 h 个头的投影矩阵拼成一个大矩阵，更接近工程实现
        # 但这里仍按『逐头』写，便于和 02 课对照理解
        self.W_Q = [rng.standard_normal((d_model, self.d_k)) * 0.3 for _ in range(h)]
        self.W_K = [rng.standard_normal((d_model, self.d_k)) * 0.3 for _ in range(h)]
        self.W_V = [rng.standard_normal((d_model, self.d_k)) * 0.3 for _ in range(h)]
        self.W_O = rng.standard_normal((h * self.d_k, d_model)) * 0.3
        self.attn_last = None  # 存最后一头的注意力矩阵供可视化

    def forward(self, x, mask):
        """x: (seq_len, d_model)，mask: (seq_len, seq_len) bool（True=允许 attend）。"""
        seq_len = x.shape[0]
        head_outs = []
        attns = []
        for i in range(self.h):
            Q = x @ self.W_Q[i]
            K = x @ self.W_K[i]
            V = x @ self.W_V[i]
            scores = Q @ K.T / np.sqrt(self.d_k)
            scores = np.where(mask, scores, -1e9)      # ⚠️ 未来位置填 -∞（关键！）
            attn = softmax(scores, axis=-1)
            head_outs.append(attn @ V)
            attns.append(attn)
        self.attn_last = attns[0]                       # 记录一头作可视化
        concat = np.concatenate(head_outs, axis=-1)
        return concat @ self.W_O


# ============================================================
# 零件 3：Position-wise FFN（两层线性 + ReLU，原论文 §3.3）
# ============================================================
class FeedForward:
    """逐位置前馈网络。原论文 §3.3 式2：
        FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
    内层维度 d_ff 通常是 d_model 的 4 倍（论文 d_model=512→d_ff=2048）。
    各位置共享参数，但层与层参数不同；等价于两个 kernel=1 的卷积。
    """

    def __init__(self, d_model, d_ff, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_model, d_ff)) * 0.3
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * 0.3
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # x: (seq_len, d_model) → (seq_len, d_ff) → (seq_len, d_model)
        h = np.maximum(0, x @ self.W1 + self.b1)   # ReLU 激活
        return h @ self.W2 + self.b2


# ============================================================
# 零件 4：绝对正余弦位置编码（03 课）
# ============================================================
def sinusoidal_pe(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# ============================================================
# 主结构：一个 Transformer Decoder Block（Post-LN，原论文 §3.1）
# ============================================================
class TransformerDecoderBlock:
    """一个 Decoder Block = MaskedMHA + 残差 + LayerNorm + FFN + 残差 + LayerNorm。

    Post-LN（原论文）：output = LayerNorm( x + Sublayer(x) )
    （现代大模型常改 Pre-LN：output = x + Sublayer(LayerNorm(x))，训练更稳）
    """

    def __init__(self, d_model, h, d_ff, seed=0):
        self.mha = MaskedMultiHeadAttention(d_model, h, seed=seed)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 100)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask):
        # 子层1：masked self-attention + 残差 + LayerNorm
        attn_out = self.mha.forward(x, mask)
        z1 = self.ln1.forward(x + attn_out)          # Post-LN

        # 子层2：FFN + 残差 + LayerNorm
        ffn_out = self.ffn.forward(z1)
        z2 = self.ln2.forward(z1 + ffn_out)          # Post-LN
        return z2


# ============================================================
# 【1】构造输入 + 因果掩码
# ============================================================
print("=" * 64)
print("【1】构造输入与因果掩码（causal mask）")
print("=" * 64)

d_model = 16
seq_len = 6
h = 4
d_ff = d_model * 4    # FFN 内层 4 倍（论文 2048 = 512×4）

np.random.seed(0)
# 模拟 6 个 token 的 embedding（实际场景是词表查表得到）
x = np.random.randn(seq_len, d_model) * 0.5
# 加位置编码（正余弦 PE，03 课）
x_pe = x + sinusoidal_pe(seq_len, d_model)

# 因果掩码：下三角为 True（位置 i 可看 0..i），上三角（未来）为 False
causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
print(f"  输入 x: {x.shape}（{seq_len} 个 token，每个 {d_model} 维）")
print(f"  加位置编码后: {x_pe.shape}")
print(f"  因果掩码 causal_mask（1=允许看，0=禁止看）：")
print(f"  {causal_mask.astype(int)}")
print("  解读：第 0 行只能看 [1,0,0,0,0,0]（只看自己）；最后一行能看全部\n")


# ============================================================
# 【2】跑通一个 Decoder Block
# ============================================================
print("=" * 64)
print("【2】跑通一个 Transformer Decoder Block")
print("=" * 64)

block = TransformerDecoderBlock(d_model=d_model, h=h, d_ff=d_ff)
out = block.forward(x_pe, causal_mask)

print(f"  输入 x_pe: {x_pe.shape}")
print(f"  Block 输出: {out.shape}")
assert out.shape == x_pe.shape, "输出形状必须与输入一致（这样才能堆叠/加残差）"
print(f"  ✅ 形状校验通过：输出 {out.shape} == 输入 {x_pe.shape}（可堆叠下一层）")
print(f"  Block 内部：")
print(f"    - MaskedMHA: {h} 头，每头 d_k={d_model//h}")
print(f"    - FFN: {d_model} → {d_ff}（4×）→ {d_model}，ReLU 激活")
print(f"    - 2 个 LayerNorm + 2 个残差连接（Post-LN）\n")

# 校验：因果掩码真的生效了吗？（把未来 token 改掉，输出不应变）
x_future_changed = x_pe.copy()
x_future_changed[0] += 100.0    # 篡改位置 0（是位置 2 的『未来』吗？不是）
# 篡改位置 3（对位置 1 而言是未来），位置 1 的输出应不变
x_future2 = x_pe.copy()
x_future2[3] += 100.0
out_future2 = block.forward(x_future2, causal_mask)
diff_pos1 = np.abs(out_future2[1] - out[1]).max()
print(f"  因果掩码校验：篡改位置3（位置1的『未来』）后，位置1输出最大变化 = {diff_pos1:.2e}")
print(f"  → 变化≈0，证明位置1确实看不到未来 ✅（causal mask 生效）\n")


# ============================================================
# 【3】堆叠多层：把多个 Block 串起来（Transformer 的『深度』）
# ============================================================
print("=" * 64)
print("【3】堆叠 4 个 Block（模拟原论文 N=6 层，这里用 4 层演示）")
print("=" * 64)

n_layers = 4
hidden = x_pe
for layer in range(n_layers):
    blk = TransformerDecoderBlock(d_model, h, d_ff, seed=layer)
    hidden = blk.forward(hidden, causal_mask)
print(f"  堆叠 {n_layers} 层后输出: {hidden.shape}")
print(f"  输出向量范数稳定（不爆炸/不消失）: ||hidden||={np.linalg.norm(hidden):.3f}")
print("  → 残差连接 + LayerNorm 的功劳：让深层网络也能稳定前向传播\n")


# ============================================================
# 【4】可视化 1：因果掩码下的注意力矩阵（下三角）
# ============================================================
attn = block.mha.attn_last   # 最后一头注意力
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
masked_show = np.where(causal_mask, attn, np.nan)
im = ax.imshow(masked_show, cmap='viridis', aspect='auto', vmin=0, vmax=attn.max())
ax.set_xlabel('key（被关注的 token）'); ax.set_ylabel('query（关注者）')
ax.set_title('Decoder 因果注意力矩阵\n（上三角=未来，被 mask 成空白）',
             fontsize=11, fontweight='bold')
words = [f'tok{i}' for i in range(seq_len)]
ax.set_xticks(range(seq_len)); ax.set_yticks(range(seq_len))
ax.set_xticklabels(words); ax.set_yticklabels(words)
for r in range(seq_len):
    for c in range(seq_len):
        if causal_mask[r, c]:
            ax.text(c, r, f'{attn[r,c]:2f}'[:4], ha='center', va='center',
                    color='white' if attn[r,c] < attn.max()*0.6 else 'black', fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)

# ============================================================
# 【5】可视化 2：Block 数据流（每一步的向量范数变化）
# ============================================================
ax = axes[1]
# 手动记录每一步的范数
step_names = []
step_norms = []
step_names.append('输入 x'); step_norms.append(np.linalg.norm(x_pe, axis=-1).mean())
attn_out = block.mha.forward(x_pe, causal_mask)
z1_pre = x_pe + attn_out
step_names.append('x+MHA'); step_norms.append(np.linalg.norm(z1_pre, axis=-1).mean())
z1 = block.ln1.forward(z1_pre)
step_names.append('LN1'); step_norms.append(np.linalg.norm(z1, axis=-1).mean())
ffn_out = block.ffn.forward(z1)
z2_pre = z1 + ffn_out
step_names.append('z1+FFN'); step_norms.append(np.linalg.norm(z2_pre, axis=-1).mean())
z2 = block.ln2.forward(z2_pre)
step_names.append('LN2'); step_norms.append(np.linalg.norm(z2, axis=-1).mean())

colors = ['#4C72B0', '#DD8452', '#55A467', '#C44E52', '#8172B3']
bars = ax.bar(step_names, step_norms, color=colors, edgecolor='black', alpha=0.85)
for b, v in zip(bars, step_norms):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}', ha='center', fontsize=9)
ax.set_ylabel('向量 L2 范数（跨 token 平均）')
ax.set_title('Block 内部数据流：每一步的范数变化\n残差防消失，LayerNorm 把范数拉回稳定区间',
             fontsize=11, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_fig = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "04-DecoderBlock.png")
os.makedirs(os.path.dirname(out_fig), exist_ok=True)
plt.savefig(out_fig, dpi=110, bbox_inches='tight')
plt.close()
print(f"【4】【5】可视化已保存：{out_fig}\n")


# ============================================================
# 【6】连回 AI + 要点总结
# ============================================================
print("=" * 64)
print("✅ 第04课要点：Transformer Decoder Block")
print("=" * 64)
print("【结构】一个 Decoder Block（Post-LN，原论文 §3.1）")
print("  z1 = LayerNorm( x + MaskedMHA(x) )      # 子层1：masked 自注意力 + 残差 + LN")
print("  z2 = LayerNorm( z1 + FFN(z1) )          # 子层2：FFN + 残差 + LN")
print()
print("【四个关键零件】")
print("  1. Masked Multi-Head Self-Attention：causal mask 保证位置 i 只看 ≤i（自回归）")
print("     mask 实现：softmax 输入的未来位置填 -∞（这里用 -1e9 近似）")
print("  2. FFN：两层线性 + ReLU，d_model → 4·d_model → d_model（逐位置，共享参数）")
print("  3. 残差连接：x + Sublayer(x)，给信息一条『捷径』，缓解深层梯度消失")
print("  4. LayerNorm：对【特征维】归一化（不是 batch 维！），稳定每层输入分布")
print()
print("【面试高频追问】")
print("  • Post-LN vs Pre-LN：原论文 Post-LN 需 warmup；现代大模型多 Pre-LN 更稳")
print("  • LayerNorm vs BatchNorm：LN 跨特征维（适合变长序列），BN 跨 batch 维")
print("  • 为什么 FFN 升 4×：高维空间引入非线性，是模型容量的主要来源")
print("  • 残差为什么能加：所有子层输出维度都是 d_model，形状对齐才能相加")
print()
print("🎯 AI 里的应用：")
print("  • GPT/LLaMA/Qwen 都是 Decoder-only，本质就是 N 个这样的 Block 堆叠")
print("  • 加上 embedding + RoPE + 输出投影到词表 = 一个完整的生成式 LLM")
print("  • 推理时配合 KV Cache，避免每生成一个 token 重算所有注意力")
print("=" * 64)
