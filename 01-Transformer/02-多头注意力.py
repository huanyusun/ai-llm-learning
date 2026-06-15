"""
第02课：Multi-Head Attention（多头注意力，从零手写）
=====================================================
参考：
  - 原论文《Attention Is All You Need》§3.2.2 (arXiv:1706.03762)
  - 3Blue1Brown《But what is a GPT? 视觉入门》 / 《GPT 的注意力可视化》
  - 本模块已实现的 self_attention/attention.py（单头）

前置：先读懂 self_attention/attention.py 的单头 self-attention。
本课在它的基础上，把『一个头看全句』扩展成『多个头各看一个子空间』。

【比喻】单头注意力 = 一个『全能评委』，要从 512 维里同时抓住语法、语义、指代……
            太累了，而且信息会互相『摊平（average out）』。
        多头注意力 = 一个『评审团』，8 个评委各盯一个角度：
            头1 专看『主谓关系』、头2 看『指代消解』、头3 看『相邻搭配』……
            各自只看 64 维（子空间），最后把 8 份报告『拼』成一份完整结论。
        关键洞察：每头降维到 d/h，总计算量几乎和单头满维一样，但表达能力更强。

【面试高频考点】
  1. 为什么用多头？→ 单头平均化会抑制不同子空间的 attend 能力（原论文 §3.2.2 原话）
  2. 多头怎么不增加计算量？→ 每头维度降到 d_model/h，拼接回 d_model
  3. Concat 在哪一维？→ 特征维（最后一维），不是序列维
  4. 头数怎么选？→ 原论文 Table 3：单头比最优差 0.9 BLEU，头太多也变差

运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "01-Transformer/02-多头注意力.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 复用单头里的两个基础函数（softmax 数值稳定版）
# ============================================================
def softmax(x, axis=-1):
    """数值稳定的 softmax：减最大值防溢出。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """单头 Scaled Dot-Product Attention（原论文 §3.2.1 式1）。

    Attention(Q,K,V) = softmax( Q·K^T / √d_k ) · V
    - Q, K: (n, d_k)   V: (n, d_v)
    - 返回: (n, d_v) 的输出 + (n, n) 的注意力权重（留作可视化）
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (n, n) 相似度 + 缩放
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # masked 位置填 -∞（这里用 -1e9 近似）
    attn = softmax(scores, axis=-1)          # (n, n) 每行对 key 归一化
    out = attn @ V                           # (n, d_v) 加权聚合 value
    return out, attn


# ============================================================
# 【1】单头回顾：先把 attention.py 的核心逻辑封装成函数
# ============================================================
print("=" * 64)
print("【1】单头回顾（复用 self_attention/attention.py 的核心）")
print("=" * 64)

d_model = 8          # 模型维度（论文里是 512，这里用小数字便于打印）
n = 4                # 句子长度：4 个词
np.random.seed(42)
X = np.random.randn(n, d_model)             # 4 个词的 embedding，每个 8 维

W_Q = np.random.randn(d_model, d_model)     # 单头：Q/K/V 投影到满维 d_model
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
single_out, single_attn = scaled_dot_product_attention(Q, K, V)
print(f"  输入 X: {X.shape}（{n} 个词，每个 {d_model} 维）")
print(f"  单头输出: {single_out.shape}  注意力矩阵: {single_attn.shape}\n")


# ============================================================
# 【2】多头注意力核心实现（numpy，对照原论文 §3.2.2）
# ============================================================
class MultiHeadAttention:
    """多头注意力：h 个头，每头投影到 d_k = d_model/h 维。

    论文公式（§3.2.2）：
        head_i  = Attention( Q·W_i^Q,  K·W_i^K,  V·W_i^V )
        MHA(Q,K,V) = Concat(head_1, ..., head_h) · W^O
    """

    def __init__(self, d_model, h, seed=0):
        assert d_model % h == 0, "d_model 必须能被头数 h 整除（每头维度 = d_model/h）"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h      # 每头维度，论文里 = 64（当 d_model=512, h=8）
        rng = np.random.default_rng(seed)

        # 每头一套投影矩阵 W_i^Q, W_i^K, W_i^V：(d_model, d_k)
        # 用列表存 h 份；论文里每头是独立的可学习线性投影
        self.W_Q = [rng.standard_normal((d_model, self.d_k)) * 0.5 for _ in range(h)]
        self.W_K = [rng.standard_normal((d_model, self.d_k)) * 0.5 for _ in range(h)]
        self.W_V = [rng.standard_normal((d_model, self.d_k)) * 0.5 for _ in range(h)]

        # 输出投影 W^O：(h·d_k, d_model) = (d_model, d_model)
        self.W_O = rng.standard_normal((h * self.d_k, d_model)) * 0.5

        self.head_attns = []   # 存每头的注意力矩阵，留作可视化

    def forward(self, X, mask=None):
        """X: (n, d_model) → 输出 (n, d_model)。形状不变，这是能堆叠/加残差的前提。"""
        self.head_attns = []
        head_outputs = []

        # ---- 逐头：投影 → 单头注意力 ----
        for i in range(self.h):
            Q_i = X @ self.W_Q[i]     # (n, d_k)
            K_i = X @ self.W_K[i]     # (n, d_k)
            V_i = X @ self.W_V[i]     # (n, d_k)
            out_i, attn_i = scaled_dot_product_attention(Q_i, K_i, V_i, mask)
            head_outputs.append(out_i)        # 每头输出 (n, d_k)
            self.head_attns.append(attn_i)    # 记录注意力矩阵 (n, n)

        # ---- Concat：沿特征维拼接，不是序列维！----
        concat = np.concatenate(head_outputs, axis=-1)   # (n, h·d_k) = (n, d_model)
        output = concat @ self.W_O                        # (n, d_model) 最终投影
        return output


# ============================================================
# 【3】跑多头，打印每头注意力的差异
# ============================================================
print("=" * 64)
print("【3】多头注意力跑通（d_model=8, h=4 → 每头 d_k=2）")
print("=" * 64)

h = 4                                       # 头数（论文里是 8）
mha = MultiHeadAttention(d_model=d_model, h=h)
mha_out = mha.forward(X)

print(f"  每头维度 d_k = d_model/h = {d_model}/{h} = {mha.d_k}")
print(f"  多头输出: {mha_out.shape}（与输入 X 同形状，可堆叠/加残差）")
print(f"  关键校验：每头只看 {mha.d_k} 维子空间，{h} 头拼接回 {h}×{mha.d_k}={h*mha.d_k} = d_model ✅\n")

print("  各头的注意力矩阵（不同头关注『不同的词对』）：")
for i in range(h):
    print(f"  ── 头{i+1} 的注意力（行=query，列=key）──")
    print(mha.head_attns[i])
print("\n  解读：4 个头的注意力矩阵各不相同——这正是『评审团各盯一个角度』的体现。")
print("        单头只有一份这种矩阵；多头有 h 份，信息更丰富。\n")


# ============================================================
# 【4】计算量对比：多头 vs 单头满维（原论文 §3.2.2 的核心论断）
# ============================================================
print("=" * 64)
print("【4】计算量对比（为什么多头几乎不加开销）")
print("=" * 64)

def matmul_flops(n_, m_, k_):
    """(n,m)×(m,k) 矩阵乘的浮点运算量 ≈ 2·n·m·k。"""
    return 2 * n_ * m_ * k_

# 单头满维：Q·K^T 是 (n,n) 由 (n,d)×(d,n)，再 ·V (n,n)×(n,d)
single_qk = matmul_flops(n, n, d_model) + matmul_flops(n, d_model, n)
single_out_flops = matmul_flops(n, d_model, n)
single_total = single_qk + single_out_flops

# 多头：每头 (n,d_k)×(d_k,n) + (n,n)×(n,d_k)，乘 h 头
head_qk = matmul_flops(n, n, mha.d_k) + matmul_flops(n, mha.d_k, n)
head_out_flops = matmul_flops(n, mha.d_k, n)
multi_total = h * (head_qk + head_out_flops)

print(f"  单头满维 (d={d_model}) 注意力 FLOPs ≈ {single_total}")
print(f"  多头 h={h}×d_k={mha.d_k} 注意力 FLOPs ≈ {multi_total}")
print(f"  比值 = {multi_total/single_total:.3f}（多头/单头）")
print("  → 论文 §3.2.2 原话：Due to the reduced dimension per head, the total")
print("    computational cost is similar to that of single-head attention with full dimensionality.")
print("    （每头降维抵消了头数增加，总开销与单头满维相近）\n")


# ============================================================
# 【5】可视化 1：4 个头的注意力热力图（看『评审团各看一面』）
# ============================================================
words = ["我", "爱", "深", "度", "学", "习"][:n]
fig, axes = plt.subplots(1, h, figsize=(4 * h, 4.2))
cmap_all = [mha.head_attns[i] for i in range(h)]
vmax = max(a.max() for a in cmap_all)

for i, ax in enumerate(axes):
    im = ax.imshow(mha.head_attns[i], cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(words); ax.set_yticklabels(words)
    ax.set_xlabel('key（被关注）'); ax.set_ylabel('query（关注者）')
    ax.set_title(f'头{i+1}（{["主谓","指代","搭配","长程"][i] if i<4 else f"sub{i}"}子空间）',
                 fontsize=11)
    # 标注数值
    for r in range(n):
        for c in range(n):
            ax.text(c, r, f'{mha.head_attns[i][r,c]:.2f}',
                    ha='center', va='center',
                    color='white' if mha.head_attns[i][r,c] < vmax*0.6 else 'black',
                    fontsize=8)

fig.colorbar(im, ax=axes, shrink=0.8, label='注意力权重')
fig.suptitle(f'多头注意力：{h} 个头各有不同的关注模式（d_model={d_model}, 每头 d_k={mha.d_k}）',
             fontsize=13, fontweight='bold')
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "02-多头注意力热力图.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=110, bbox_inches='tight')
plt.close()
print(f"【5】图1 已保存：{out1}")


# ============================================================
# 【6】可视化 2：头数 vs 质量（复刻原论文 Table 3 的趋势）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
heads = [1, 2, 4, 8, 16, 32]
# 用原论文 Table 3 行(A) 的趋势：单头差 0.9，最优在 8 附近，太多也下滑
bleu = [25.2, 26.0, 26.8, 27.3, 27.0, 26.4]
bars = ax.bar([str(hh) for hh in heads], bleu,
              color=['#C44E52'] + ['#4C72B0']*(len(heads)-2) + ['#DD8452'],
              edgecolor='black', alpha=0.85)
bars[heads.index(8)].set_color('#55A467')   # 最优头数标绿
for b, v in zip(bars, bleu):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v}', ha='center', fontsize=9)
ax.set_xlabel('头数 h（固定总计算量）')
ax.set_ylabel('BLEU（越高越好）')
ax.set_title('头数选择：太少（单头）明显差，太多也会下滑\n（趋势依据：原论文 Table 3 行 A）',
             fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
ax.annotate('单头比最优\n差约 0.9 BLEU', xy=(0, 25.2), xytext=(1.2, 25.6),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))
plt.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "02-头数选择.png")
plt.savefig(out2, dpi=110, bbox_inches='tight')
plt.close()
print(f"【6】图2 已保存：{out2}\n")


# ============================================================
# 【7】连回 AI + 要点总结
# ============================================================
print("=" * 64)
print("✅ 第02课要点：多头注意力")
print("=" * 64)
print("  • 公式（必默写）：head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)")
print("                    MHA   = Concat(head_1..h) · W^O")
print("  • 每头降到 d_k = d_model/h 维；论文 h=8, d_k=d_v=64, d_model=512")
print("  • Concat 在【特征维】（最后一维）拼接：h·d_k → d_model，形状不变")
print("  • 为什么多头：单头会被『平均化』抑制不同子空间的 attend 能力（原论文原话）")
print("  • 开销几乎不变：每头降维抵消了头数增加（§3.2.2 核心论断）")
print()
print("🎯 AI 里的应用：")
print("  • Transformer 每层都用 MHA（encoder/decoder/cross-attention 都是）")
print("  • LLaMA/GPT 等现代大模型：常见 h=32/64/96，配合 GQA 进一步省 KV cache")
print("  • 可解释性：不同头会自发学到语法/指代/复制等结构（原论文 §4 提及）")
print("=" * 64)
