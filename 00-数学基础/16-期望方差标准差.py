"""
数学基础 第16课：期望、方差、标准差（从零开始）
================================================
参考：概率论基础、动手学深度学习 https://zh.d2l.ai、《Attention Is All You Need》

三个描述「一坨数字」的核心量：
- 期望 E[X]：平均值，代表「中心」在哪
- 方差 Var(X)：数据离均值有多远（的平方的平均）→ 描述「分散程度」
- 标准差 σ = √方差：和原数据同一个量纲，更好理解

本课重点解答一个高频面试题：
  为什么 attention 里的 QKᵀ 要除以 √d（d 是维度）？
答案：高维点积的方差会随维度 d 线性增大 → 数值会变得很大 →
     softmax 后梯度几乎为 0（梯度消失）→ 训练不动。除以 √d 把方差缩回 1。
运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "00-数学基础/16-期望方差标准差.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体（开头模板，固定写法）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(0)

# ============================================================
# 【1】期望 = 长期平均值（「加权平均」）
# ============================================================
# 离散情形：E[X] = Σ xᵢ · pᵢ   （每个值 × 它的概率，加起来）
# 连续情形：E[X] = ∫ x · f(x) dx
# 经验上：用大量样本的平均值来近似期望
data = np.random.normal(3.0, 1.0, size=100000)   # 真实均值 μ=3
print("【1】期望 E[X] = 平均值")
print(f"    真实均值 μ = 3.0")
print(f"    样本均值（近似期望）= {data.mean():.4f}")
print("    样本越多，样本均值越接近真实期望（大数定律）\n")

# ============================================================
# 【2】方差 = 「分散程度」：每个点离均值有多远（取平方后求平均）
# ============================================================
# 公式：Var(X) = E[(X - μ)²]  → 离均差的平方的期望
# 样本方差：var = (1/n) · Σ (xᵢ - x̄)²
mean = data.mean()
var = np.mean((data - mean) ** 2)          # 手算方差
print("【2】方差 Var(X) = 离均差的平方的平均")
print(f"    手算方差 = {var:.4f}")
print(f"    numpy var = {data.var():.4f}   （两者一致）")
print(f"    真实方差 σ² = 1.0（因为 σ=1）\n")

# ============================================================
# 【3】标准差 σ = √方差：和原数据同量纲，可直接解读
# ============================================================
std = np.sqrt(var)
print("【3】标准差 σ = √方差")
print(f"    标准差 = √{var:.4f} = {std:.4f}")
print(f"    numpy std = {data.std():.4f}")
print("    为什么开根号？方差是「平方」量纲（比如 米²），")
print("    开根号变回「米」，才能和原始数据直接比较\n")

# ============================================================
# 【4】关键引理：两个独立随机变量之和的方差 = 方差之和
# ============================================================
# 若 X,Y 独立：Var(X+Y) = Var(X) + Var(Y)
# 推论：n 个独立同分布(方差σ²)变量之和，方差 = n·σ²
#       它们的「平均」方差 = σ²；它们的「和」标准差 = σ·√n
print("【4】独立变量之和的方差 = 方差之和（重要引理）")
print("    若 X,Y 独立：Var(X+Y) = Var(X) + Var(Y)")
print("    n 个独立(方差σ²)变量之和 → 方差 = n·σ²，标准差 = σ·√n")
print("    这就是 attention 里除以 √d 的数学根源！\n")

# ============================================================
# 【5】模拟：点积方差随维度 d 线性增大（除以 √d 前后对比）
# ============================================================
# attention 的 Q·Kᵀ 中，q_i·k_i = Σ_{j=1}^{d} q_ij · k_ij
# 假设每个 q_ij, k_ij 都是 N(0,1) 的独立随机变量：
#   每项 q_ij·k_ij 的方差 = Var(q)·Var(k) + ... ≈ 1
#   d 项相加 → 总方差 ≈ d，标准差 ≈ √d → 数值随 d 增大而膨胀
dims = [1, 4, 16, 64, 256, 1024]
n_trials = 20000
print("【5】模拟：维度 d 越大，点积数值越分散（方差≈d）")
print(f"    {'维度d':>6} | {'未缩放方差(≈d)':>16} | {'未缩放std(≈√d)':>16} | {'除√d后方差(≈1)':>16}")
print("    " + "-" * 66)
unscaled_vars, scaled_vars, unscaled_stds = [], [], []
for d in dims:
    q = np.random.normal(0, 1, size=(n_trials, d))
    k = np.random.normal(0, 1, size=(n_trials, d))
    dot = np.sum(q * k, axis=1)            # 点积 Q·Kᵀ（每个 trial 一个值）
    dot_scaled = dot / np.sqrt(d)         # attention 的缩放
    uv, sv = dot.var(), dot_scaled.var()
    unscaled_vars.append(uv); scaled_vars.append(sv); unscaled_stds.append(dot.std())
    print(f"    {d:>6} | {uv:>16.2f} | {dot.std():>16.2f} | {sv:>16.2f}")
print("\n    结论：未缩放方差≈d（随维度爆炸）；除以√d后方差≈1（稳定）\n")

# ============================================================
# 【6】可视化：① 点积方差 vs 维度  ② softmax 梯度消失现象
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 图1：方差/标准差 随维度 d 的变化
ax = axes[0]
d_fine = np.array(dims)
ax.plot(d_fine, unscaled_vars, 'o-', color='#d62728', lw=2.5, label='未缩放方差 Var(点积)')
ax.plot(d_fine, d_fine, '--', color='#d62728', alpha=0.4, label='理论 y = d')
ax.plot(d_fine, unscaled_stds, 's-', color='#ff7f0e', lw=2.5, label='未缩放标准差 std(点积)')
ax.plot(d_fine, np.sqrt(d_fine), '--', color='#ff7f0e', alpha=0.4, label='理论 y = √d')
ax.plot(d_fine, scaled_vars, '^--', color='#2ca02c', lw=2.5, label='除√d 后方差 ≈ 1')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("维度 d (对数轴)"); ax.set_ylabel("方差 / 标准差 (对数轴)")
ax.set_title("① 点积方差随维度 d 增大：未缩放≈d，除√d后≈1", fontsize=11)
ax.grid(True, alpha=0.3, which='both'); ax.legend(fontsize=9)

# 图2：softmax 输出随点积数值变大而「变尖」→ 梯度消失
ax = axes[1]
def softmax(z):
    e = np.exp(z - z.max())   # 减最大值，数值稳定
    return e / e.sum()

scales = [1, 4, 16, 64]        # 对应不同 d 量级的点积分散程度
xs = np.arange(5)
for s in scales:
    z = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * s   # 模拟点积（数值随 s 放大）
    p = softmax(z)
    ax.plot(xs, p, 'o-', lw=2, label=f'点积×{s}（数值分散 std≈{z.std():.1f}）')
ax.set_xticks(xs)
ax.set_title("② 点积越大 → softmax 越尖 → 其余项梯度≈0（消失）", fontsize=11)
ax.set_xlabel("类别"); ax.set_ylabel("softmax 概率")
ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "16-期望方差标准差.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【6】图已保存：{out}\n")

# ============================================================
# 【7】连回 AI：attention 为什么除以 √d（完整推导）
# ============================================================
print("【7】连回 AI：Scaled Dot-Product Attention 为什么要除以 √d")
print("    假设 Q、K 的每个元素都是独立 N(0,1)：")
print("      点积 = Σ_{i=1}^{d} q_i · k_i")
print("      每项 q_i·k_i 期望=0、方差≈1")
print("      ⇒ 点积的方差 = d，标准差 = √d")
print("    d 很大时（如 d=512），点积数值可达 ±几十，softmax 输出接近 one-hot：")
print("      · 最大项概率→1，其余→0")
print("      · 其余项的梯度 ∝ p_i·(…) ≈ 0  → 梯度消失，学不动")
print("    除以 √d 把方差缩回 ≈1 → 数值稳定 → softmax 平滑 → 梯度健康")
print("    公式：Attention(Q,K,V) = softmax( QKᵀ / √d ) · V\n")

print("=" * 62)
print("✅ 第16课要点")
print("=" * 62)
print("  • 期望 E[X]：平均值，代表分布的「中心」")
print("  • 方差 Var(X)：离均差的平方的平均，描述「分散程度」")
print("  • 标准差 σ = √方差：和原数据同量纲，更直观")
print("  • 独立变量之和：Var(X+Y)=Var(X)+Var(Y)，n 个相加方差×n")
print()
print("🎯 AI 里的应用：")
print("  • Attention 缩放：softmax(QKᵀ/√d) 防止高维点积方差过大")
print("    （方差≈d → 数值爆炸 → softmax 变尖 → 梯度消失）")
print("  • LayerNorm/BatchNorm：用均值和方差做归一化，稳定训练")
print("  • 损失函数、不确定度估计都依赖方差/标准差")
print("=" * 62)
