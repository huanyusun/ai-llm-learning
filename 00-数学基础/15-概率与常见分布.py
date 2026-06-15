"""
数学基础 第15课：概率与常见分布（从零开始）
==========================================
参考：概率论基础、3Blue1Brown《概率》、动手学深度学习 https://zh.d2l.ai

概率 = 「某件事发生的可能性」，0 到 1 之间的小数。
分布 = 「所有可能结果，各自有多大概率」的规律。AI 里最常用的两种：
- 正态分布（高斯）：钟形曲线，自然界随处可见（身高、噪声、权重初始化…）
- 均匀分布：每个结果概率相等（掷骰子、随机采样）

本课用 numpy 随机采样 + matplotlib 画出不同参数下的分布形状。
运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "00-数学基础/15-概率与常见分布.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体（开头模板，固定写法）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子，保证每次跑结果一样（可复现）
np.random.seed(42)

# ============================================================
# 【1】概率基础：0~1 之间的数，加起来（互斥事件）= 1
# ============================================================
print("【1】概率基础")
print("    掷一颗骰子：1~6 点，每点概率 = 1/6 ≈ 0.167")
print("    所有结果概率加起来 = 6 × (1/6) = 1（必然事件的概率是 1）")
# 用 numpy 模拟掷 6 万次骰子，看每个面出现的频率
rolls = np.random.randint(1, 7, size=60000)
freq = np.bincount(rolls)[1:] / 60000   # 去掉 0 那一档
print(f"    实测：掷 60000 次的频率 = {np.round(freq, 4)}（都接近 1/6）\n")

# ============================================================
# 【2】均匀分布：每个值概率相等
# ============================================================
# 在区间 [0,1) 上均匀采样 100000 个数
uni = np.random.uniform(0, 1, size=100000)
print("【2】均匀分布 Uniform(0,1)")
print(f"    理论均值 = 0.5，实测均值 = {uni.mean():.4f}")
print(f"    理论方差 = 1/12 ≈ {1/12:.4f}，实测方差 = {uni.var():.4f}")
print("    用途：随机初始化、dropout 采样、随机打乱数据\n")

# ============================================================
# 【3】正态分布（高斯）：钟形，中间多、两头少
# ============================================================
# 概率密度函数 PDF：  f(x) = (1/(σ√2π)) · exp( -(x-μ)² / (2σ²) )
# μ(mu) = 均值：钟的中心位置；σ(sigma) = 标准差：钟的胖瘦
def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

print("【3】正态分布 Normal(μ, σ²)")
print("    公式：f(x) = (1/(σ√2π)) · exp( -(x-μ)² / (2σ²) )")
print("    μ = 均值 → 决定钟形「中心」在哪")
print("    σ = 标准差 → 决定钟形「胖瘦」（σ 大 = 更分散、更扁）\n")

# 从标准正态 N(0,1) 采样
std_normal = np.random.normal(0, 1, size=100000)
print(f"    N(0,1) 采样：实测均值={std_normal.mean():.4f}，标准差={std_normal.std():.4f}")
print(f"    约 68% 的点落在 [μ-σ, μ+σ] = [-1, 1] 内")
in_1sigma = np.mean(np.abs(std_normal) <= 1)
print(f"    实测 |x|≤1 的比例 = {in_1sigma:.4f}（应≈0.68）\n")

# ============================================================
# 【4】均值与方差如何影响分布形状
# ============================================================
print("【4】参数怎么影响形状")
print("    μ 变大  → 整条钟形曲线「向右平移」")
print("    σ 变大  → 钟形「变矮变胖」(更分散)；σ 变小 → 变高变瘦(更集中)\n")

# ============================================================
# 【5】可视化：均匀 vs 正态，以及不同 μ、σ 的正态
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# 图1：均匀分布 直方图
ax = axes[0]
ax.hist(uni, bins=50, density=True, color='#1f77b4', alpha=0.7, edgecolor='white')
ax.axhline(1.0, color='red', ls='--', lw=2, label='理论 PDF = 1.0')
ax.set_title("① 均匀分布 Uniform(0,1)：每个值概率相等", fontsize=11)
ax.set_xlabel("值"); ax.set_ylabel("概率密度")
ax.set_ylim(0, 1.5); ax.grid(True, alpha=0.3); ax.legend()

# 图2：正态分布 直方图 vs 理论 PDF 曲线
ax = axes[1]
ax.hist(std_normal, bins=60, density=True, color='#ff7f0e', alpha=0.6, edgecolor='white', label='采样直方图')
xs = np.linspace(-4, 4, 300)
ax.plot(xs, normal_pdf(xs, 0, 1), color='black', lw=2.5, label='理论 PDF N(0,1)')
ax.axvline(-1, color='gray', ls=':', lw=1); ax.axvline(1, color='gray', ls=':', lw=1)
ax.set_title("② 正态分布：钟形，中间高两头低", fontsize=11)
ax.set_xlabel("值"); ax.set_ylabel("概率密度")
ax.set_xlim(-4, 4); ax.grid(True, alpha=0.3); ax.legend()

# 图3：不同 μ、σ 的正态曲线（看形状变化）
ax = axes[2]
xs = np.linspace(-6, 10, 400)
configs = [
    (0, 1, 'N(0,1)  标准', '#1f77b4'),
    (0, 2, 'N(0,2²) σ大→矮胖', '#d62728'),
    (4, 0.7, 'N(4,0.7²) μ右移、σ小→高瘦', '#2ca02c'),
]
for mu, sigma, label, color in configs:
    ax.plot(xs, normal_pdf(xs, mu, sigma), lw=2.5, color=color,
            label=f'{label}\n  μ={mu}, σ={sigma}')
ax.set_title("③ 改 μ、σ：μ 平移钟，σ 改胖瘦", fontsize=11)
ax.set_xlabel("值"); ax.set_ylabel("概率密度")
ax.set_xlim(-6, 10); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "15-概率与常见分布.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")

# ============================================================
# 【6】连回 AI：权重初始化、噪声、中心极限定理
# ============================================================
print("【6】连回 AI")
print("  • 权重初始化常用正态分布 N(0, σ²)：让初始网络既不太大也不全 0")
print("  • VAE / 扩散模型：用正态分布建模「潜在变量」和「噪声」")
print("  • Dropout：用均匀/伯努利分布随机「关掉」一些神经元")
print("  • 中心极限定理：很多独立小随机量加起来 → 趋近正态分布")
print("    （这就是为什么正态分布在自然界和 AI 里无处不在）\n")

print("=" * 60)
print("✅ 第15课要点")
print("=" * 60)
print("  • 概率：0~1 之间；所有互斥结果加起来 = 1")
print("  • 均匀分布：每个值概率相等（掷骰子、随机采样）")
print("  • 正态分布 N(μ,σ²)：钟形；μ=中心位置，σ=胖瘦程度")
print("  • 68% 的数据落在 [μ-σ, μ+σ] 内（经验法则）")
print("  • numpy 采样：np.random.normal / uniform / randint")
print()
print("🎯 AI 里的应用：")
print("  • 权重初始化、加噪、潜在变量都用正态分布")
print("  • 随机采样、数据打乱、dropout 用均匀/伯努利分布")
print("  • 数据本身往往就近似正态（中心极限定理）")
print("=" * 60)
