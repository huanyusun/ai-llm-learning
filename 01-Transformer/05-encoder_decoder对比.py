"""
05 · Encoder-Decoder 架构对比 + BERT vs GPT
==============================================
面试考点：三大架构的区别、BERT 和 GPT 各自适合什么任务
运行：uv run python "01-Transformer/05-encoder_decoder对比.py"
"""

import numpy as np

print("=" * 60)
print("Transformer 三大架构对比")
print("=" * 60)

# ── 1. 三大架构 ──────────────────────────────────────────
print("""
┌─────────────────────────────────────────────────────────────┐
│  1. Encoder-Only（BERT 系列）                                │
│                                                             │
│  输入 → [Encoder × N] → 双向表示                              │
│                                                             │
│  注意力：双向（每个 token 能看到所有其他 token）                 │
│  预训练：MLM（掩码语言模型）+ NSP                               │
│  适合：理解类任务（分类、NER、问答、语义相似度）                  │
│  代表：BERT, RoBERTa, ALBERT, DeBERTa                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  2. Decoder-Only（GPT 系列）                                 │
│                                                             │
│  输入 → [Decoder × N] → 自回归生成                            │
│                                                             │
│  注意力：单向/因果（每个 token 只能看到之前的 token）             │
│  预训练：CLM（因果语言模型，Next Token Prediction）              │
│  适合：生成类任务（对话、写作、代码生成）                        │
│  代表：GPT-2/3/4, LLaMA, Qwen, DeepSeek                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  3. Encoder-Decoder（T5, BART）                              │
│                                                             │
│  输入 → [Encoder × N] → 表示 → [Cross-Attn + Decoder × M]   │
│                                                             │
│  Encoder：双向注意力，理解输入                                 │
│  Decoder：因果注意力 + Cross-Attention（关注 Encoder 输出）     │
│  适合：序列到序列任务（翻译、摘要、问答生成）                    │
│  代表：T5, BART, mBART, Flan-T5                              │
└─────────────────────────────────────────────────────────────┘
""")

# ── 2. 注意力掩码对比 ──────────────────────────────────────
print("=" * 60)
print("注意力掩码对比（4 个 token 的例子）")
print("=" * 60)

# Encoder: 双向注意力（全连接）
encoder_mask = np.ones((4, 4), dtype=int)
print("\nEncoder（双向）— 每个 token 都能看到所有 token:")
print(encoder_mask)

# Decoder: 因果注意力（下三角）
decoder_mask = np.tril(np.ones((4, 4), dtype=int))
print("\nDecoder（因果）— 每个 token 只能看到自己和之前的:")
print(decoder_mask)

# Cross-Attention: Decoder query 关注 Encoder 的所有输出
print("\nCross-Attention — Decoder 的每个位置都能关注 Encoder 的所有输出:")
cross_mask = np.ones((4, 4), dtype=int)
print(cross_mask)
print("（和 Encoder 的掩码一样是全连接，但 Q 来自 Decoder，K/V 来自 Encoder）")

# ── 3. BERT vs GPT 详细对比 ──────────────────────────────
print("\n" + "=" * 60)
print("BERT vs GPT 详细对比")
print("=" * 60)
print("""
| 维度         | BERT                        | GPT                          |
|-------------|-----------------------------|-----------------------------|
| 架构         | Encoder-Only                | Decoder-Only                 |
| 注意力方向    | 双向（看前看后）               | 单向（只看前面）               |
| 预训练任务    | MLM + NSP                   | CLM (Next Token Prediction)  |
| 输出         | 每个 token 的上下文表示        | 下一个 token 的概率分布        |
| 擅长         | 理解（分类、匹配、抽取）        | 生成（对话、写作、推理）        |
| 参数规模      | 110M ~ 340M                 | 117M ~ 1.8T                  |
| 微调方式      | 加分类头，全参数微调            | Prompt / ICL / LoRA          |
| 代表应用      | 搜索排序、情感分析、NER         | ChatGPT、代码生成、Agent      |

为什么 GPT 架构"赢了"？
1. 生成能力：Decoder-Only 天然支持自回归生成
2. Scaling：GPT 架构在大规模下涌现能力更强
3. 统一范式：所有任务都可以转化为"生成"（包括分类）
4. ICL 能力：大规模 GPT 能做 In-Context Learning，无需微调

BERT 仍然有用的场景：
1. 嵌入模型（如 BGE、E5）— 用于 RAG 的向量检索
2. Rerank 交叉编码器 — 用于重排序
3. 轻量级分类任务 — 部署成本低
""")

# ── 4. 代码演示：因果掩码 vs 双向 ──────────────────────────
print("=" * 60)
print("代码演示：因果掩码的效果")
print("=" * 60)

np.random.seed(42)
seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

# 计算注意力分数
scores = Q @ K.T / np.sqrt(d_k)

# 双向注意力
weights_bi = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
out_bi = weights_bi @ V

# 因果注意力
causal_mask = np.tril(np.ones((seq_len, seq_len)))
scores_causal = np.where(causal_mask == 1, scores, -1e9)
weights_causal = np.exp(scores_causal) / np.exp(scores_causal).sum(axis=-1, keepdims=True)
out_causal = weights_causal @ V

print("\n双向注意力权重（Encoder）:")
print(np.round(weights_bi, 3))
print("\n因果注意力权重（Decoder）:")
print(np.round(weights_causal, 3))
print("\n→ 因果掩码让上三角变成 0，每个 token 只关注自己和之前的 token")
print('→ 这就是 GPT 能做自回归生成的关键：预测时不能"偷看"未来')
