"""
22 · 信息论基础 — KL散度、互信息与交叉熵的关系
==============================================
面试考点：KL散度的含义和非对称性、互信息、与交叉熵/信息熵的关系
运行：uv run python "00-数学基础/22-信息论基础.py"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 1. 信息熵回顾 ──────────────────────────────────────────
print("=" * 60)
print("1. 信息熵 H(P) — 不确定性的度量")
print("=" * 60)

def entropy(p):
    """计算离散分布的信息熵"""
    p = np.array(p, dtype=float)
    p = p[p > 0]  # 去掉 0（0*log0 = 0）
    return -np.sum(p * np.log2(p))

p_uniform = [0.25, 0.25, 0.25, 0.25]
p_peaked = [0.9, 0.05, 0.03, 0.02]
p_certain = [1.0, 0.0, 0.0, 0.0]

print(f"均匀分布 H = {entropy(p_uniform):.4f} bits  (最大不确定性)")
print(f"尖峰分布 H = {entropy(p_peaked):.4f} bits  (较确定)")
print(f"确定分布 H = {entropy(p_certain):.4f} bits  (完全确定)")

# ── 2. KL 散度 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. KL 散度 D_KL(P||Q) — 分布差异的度量")
print("=" * 60)

def kl_divergence(p, q):
    """KL(P||Q) = Σ P(x) * log(P(x)/Q(x))"""
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    # 只在 p>0 的位置计算
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

P = [0.4, 0.3, 0.2, 0.1]
Q = [0.25, 0.25, 0.25, 0.25]

kl_pq = kl_divergence(P, Q)
kl_qp = kl_divergence(Q, P)

print(f"P = {P}")
print(f"Q = {Q}")
print(f"KL(P||Q) = {kl_pq:.4f} bits")
print(f"KL(Q||P) = {kl_qp:.4f} bits")
print(f"KL(P||Q) ≠ KL(Q||P) → KL 散度是【非对称】的！")
print()
print("直觉：KL(P||Q) 衡量'用 Q 近似 P 时损失了多少信息'")
print("  - KL ≥ 0（Gibbs 不等式）")
print("  - KL = 0 当且仅当 P = Q")
print("  - 不是距离（不对称、不满足三角不等式）")

# ── 3. 交叉熵 = 熵 + KL 散度 ──────────────────────────────
print("\n" + "=" * 60)
print("3. 三者关系：H(P,Q) = H(P) + KL(P||Q)")
print("=" * 60)

def cross_entropy(p, q):
    """H(P,Q) = -Σ P(x) * log Q(x)"""
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    mask = p > 0
    return -np.sum(p[mask] * np.log2(q[mask]))

h_p = entropy(P)
ce_pq = cross_entropy(P, Q)

print(f"H(P)      = {h_p:.4f} bits    (P 的信息熵)")
print(f"KL(P||Q)  = {kl_pq:.4f} bits  (P 和 Q 的差异)")
print(f"H(P) + KL = {h_p + kl_pq:.4f} bits")
print(f"H(P,Q)    = {ce_pq:.4f} bits  (交叉熵)")
print()
print("→ 交叉熵 = 信息熵 + KL散度")
print("→ 训练时最小化交叉熵 ≡ 最小化 KL 散度（因为 H(P) 是常数）")
print("→ 这就是为什么 LLM 训练用交叉熵 loss！")

# ── 4. 互信息 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 互信息 I(X;Y) — 两个变量共享多少信息")
print("=" * 60)

# 构造联合分布
# X: 天气 (晴/雨)  Y: 带伞 (是/否)
joint = np.array([
    [0.3, 0.2],   # 晴: [不带伞, 带伞]
    [0.05, 0.45],  # 雨: [不带伞, 带伞]
])

p_x = joint.sum(axis=1)  # 边缘分布 P(X)
p_y = joint.sum(axis=0)  # 边缘分布 P(Y)

# I(X;Y) = Σ P(x,y) * log(P(x,y) / (P(x)*P(y)))
mi = 0
for i in range(2):
    for j in range(2):
        if joint[i, j] > 0:
            mi += joint[i, j] * np.log2(joint[i, j] / (p_x[i] * p_y[j]))

print(f"联合分布 P(X,Y):")
print(f"         不带伞  带伞")
print(f"  晴:    {joint[0,0]:.2f}   {joint[0,1]:.2f}")
print(f"  雨:    {joint[1,0]:.2f}   {joint[1,1]:.2f}")
print(f"\nI(X;Y) = {mi:.4f} bits")
print(f"H(X)   = {entropy(p_x):.4f} bits")
print(f"H(Y)   = {entropy(p_y):.4f} bits")
print()
print("互信息的等价表达：")
print("  I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)")
print("  I(X;Y) = H(X) + H(Y) - H(X,Y)")
print("  I(X;Y) = KL(P(X,Y) || P(X)P(Y))")
print()
print("→ 互信息 = 联合分布与独立分布的 KL 散度")
print("→ I=0 当且仅当 X,Y 独立")

# ── 5. 可视化 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 5a: KL 非对称性
alphas = np.linspace(0.01, 0.99, 100)
P_binary = np.column_stack([alphas, 1 - alphas])
Q_fixed = np.array([0.5, 0.5])
kl_forward = [kl_divergence(p, Q_fixed) for p in P_binary]
kl_reverse = [kl_divergence(Q_fixed, p) for p in P_binary]

axes[0].plot(alphas, kl_forward, label="KL(P||Q)", linewidth=2)
axes[0].plot(alphas, kl_reverse, label="KL(Q||P)", linewidth=2, linestyle="--")
axes[0].set_xlabel("P(x=1)")
axes[0].set_ylabel("KL (bits)")
axes[0].set_title("KL Divergence Asymmetry")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 5b: 信息论三者关系（韦恩图用文字）
axes[1].text(0.5, 0.8, "H(P,Q) = H(P) + KL(P||Q)", ha="center", fontsize=14, fontweight="bold")
axes[1].text(0.5, 0.55, "Cross-Entropy = Entropy + KL Divergence", ha="center", fontsize=11)
axes[1].text(0.5, 0.35, "Minimize CE ≡ Minimize KL", ha="center", fontsize=11, color="red")
axes[1].text(0.5, 0.15, "(because H(P) is constant w.r.t. model)", ha="center", fontsize=9, color="gray")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].axis("off")
axes[1].set_title("Key Relationship")

# 5c: 互信息
labels = ["H(X)", "H(Y)", "I(X;Y)"]
values = [entropy(p_x), entropy(p_y), mi]
colors = ["#4CAF50", "#2196F3", "#FF9800"]
axes[2].bar(labels, values, color=colors, width=0.5)
axes[2].set_ylabel("bits")
axes[2].set_title("Mutual Information")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("00-数学基础/img/22-信息论基础.png", dpi=120)
print("\n✅ 可视化已保存 → 00-数学基础/img/22-信息论基础.png")

# ── 6. 面试速查 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 面试速查：信息论在 AI 中的应用")
print("=" * 60)
print("""
| 概念      | 公式                          | AI 中的应用                    |
|----------|-------------------------------|-------------------------------|
| 信息熵 H  | -Σ P log P                    | 衡量模型输出的不确定性            |
| 交叉熵 CE | -Σ P log Q                    | LLM 训练的 loss 函数            |
| KL 散度   | Σ P log(P/Q)                  | VAE 正则项、RLHF 中约束策略偏移   |
| 互信息 MI | KL(P(X,Y) || P(X)P(Y))       | 特征选择、表示学习               |

关键面试点：
1. KL 散度非对称 → 不是距离
2. 交叉熵 = 熵 + KL → 最小化 CE 等价于最小化 KL
3. RLHF/DPO 中用 KL 散度约束新策略不要偏离参考策略太远
4. VAE 的 ELBO 中有 KL(q(z|x) || p(z)) 项
5. 互信息 I(X;Y) = 0 ⟺ X,Y 独立
""")
