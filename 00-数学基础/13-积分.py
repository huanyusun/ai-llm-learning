"""
数学基础 第13课：积分 Integral（从零开始）
=========================================================================
参考：3Blue1Brown《微积分的本质》、《动手学深度学习》https://zh.d2l.ai

导数研究「变化快慢」，积分研究「累积总量」。它俩其实是一对逆运算。

核心思想（3 句话）：
1. 定积分 = 曲线下面的「面积」= 无数个细长小矩形面积之和（当宽度→0）。
2. 不定积分 = 导数的逆：找一个函数 F，使得 F' = f。
3. 微积分基本定理：定积分 = F(b) − F(a)，把「算面积」变成「代入原函数」。

AI 角度：概率密度曲线下面的面积 = 概率。积分是理解概率分布的钥匙。

本课用 matplotlib 把「曲线下面积」填出来，并验证基本定理。
运行：uv run python "00-数学基础/13-积分.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】定积分 = 曲线下面积（用小矩形近似）
# ============================================================
# 拿一个简单的函数：f(x) = x²。求它在 [0, 3] 这段下面的面积。
# 思路（3Blue1Brown 的讲法）：把 [0,3] 切成很多小段，
# 每段用一个细长矩形（高=f(x)，宽=dx）去近似，再把所有矩形面积加起来。
# 矩形越细（dx→0），总和就越接近真实面积 —— 这就是「积分」。

def f(x):
    return x ** 2

a, b = 0.0, 3.0
print("【1】定积分 ≈ 曲线下面积")
print(f"    f(x) = x²，区间 [{a}, {b}]")
print("    把区间切成 N 个小矩形，高=f(x)，宽=dx，求和：\n")

# 用不同数量的矩形看『越切越准』
for N in [5, 50, 500]:
    x = np.linspace(a, b, N + 1)     # N+1 个点 → N 个矩形
    y = f(x)
    dx = (b - a) / N
    area = np.sum(y[:-1] * dx)        # 左端点黎曼和
    print(f"    N={N:>4} 个矩形，dx={dx:.4f}，面积 ≈ {area:.5f}")

# 理论值（用基本定理算，见下面第3节）：F(3)-F(0) = 9
print(f"    → 矩形越细，越逼近真实面积 9.00000\n")

# ============================================================
# 【2】不定积分 = 导数的逆运算
# ============================================================
# 已知 f(x) 的导数，反过来求 f(x) 本身。
# 例：d/dx(x³) = 3x²  ⟹  ∫x² dx = x³/3 + C
#   （C 是任意常数，因为常数求导变 0，所以原函数可以整体平移）
print("【2】不定积分（导数的逆）")
print("    已知 d/dx(x³) = 3x²，反过来：")
print("    ∫ x² dx = x³/3 + C    （C 是任意常数）")
print("    所以 f(x)=x² 的一个『原函数』是 F(x) = x³/3\n")

def F(x):
    return x ** 3 / 3   # f(x)=x² 的原函数

# ============================================================
# 【3】微积分基本定理：定积分 = F(b) − F(a)
# ============================================================
# 这是微积分最美的一个结论：
#   「算面积」这件听起来要切成无数小矩形的事，
#    居然只要找到原函数 F，代入端点相减就完成了！
exact = F(b) - F(a)
print("【3】微积分基本定理")
print(f"    ∫₀³ x² dx = F(3) − F(0) = 3³/3 − 0 = {F(b)} − {F(a)} = {exact}")
print(f"    这就是第1节里矩形们『越切越逼近』的那个真实面积。\n")

# ============================================================
# 【4】AI 应用：概率密度曲线下的面积 = 概率
# ============================================================
# 概率密度函数 PDF（比如正态分布那条钟形曲线）本身不是概率，
# 『它下方的面积』才是概率。整条曲线下总面积 = 1（全部概率之和）。
#
# 例：标准正态分布 N(0,1) 在 [−1, 1] 之间的面积 ≈ 0.683，
# 也就是『数据落在均值 ±1 个标准差内』的概率 ≈ 68.3%（著名的 68-95-99.7 法则）。
def normal_pdf(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

xs = np.linspace(-4, 4, 1000)
ys = normal_pdf(xs)
# 用梯形法算 [-1, 1] 的面积
mask = (xs >= -1) & (xs <= 1)
prob_68 = np.trapezoid(ys[mask], xs[mask])
total_area = np.trapezoid(ys, xs)
print("【4】概率分布的面积 = 概率")
print(f"    标准正态分布整条曲线下面积 = {total_area:.4f}（应该 = 1）")
print(f"    [−1, 1] 这段的面积 = {prob_68:.4f}（即 ±1σ 内的概率 ≈ 68%）")
print("    → 这就是为什么分类模型/采样的『概率』本质是积分。\n")

# ============================================================
# 【5】可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 图1：矩形近似（黎曼和）+ 真实曲线
ax = axes[0]
N = 12
x_rect = np.linspace(a, b, N + 1)
y_rect = f(x_rect)
dx = (b - a) / N
ax.bar(x_rect[:-1], y_rect[:-1], width=dx, align='edge',
       color='orange', alpha=0.6, edgecolor='black', label=f'{N} 个矩形（近似）')
xs_fine = np.linspace(a, b, 200)
ax.plot(xs_fine, f(xs_fine), 'r-', lw=2.5, label='f(x)=x²（真实）')
ax.set_title("积分 = 曲线下面积\n（矩形越多越准）", fontsize=12)
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.legend(); ax.grid(True, alpha=0.3)

# 图2：定积分 = 填充面积，标注 F(b)-F(a)
ax2 = axes[1]
ax2.plot(xs_fine, f(xs_fine), 'r-', lw=2.5)
ax2.fill_between(xs_fine, 0, f(xs_fine), alpha=0.3, color='red',
                 label=f'面积 = ∫₀³ x² dx = {exact:.0f}')
ax2.set_title("微积分基本定理\n∫₀³ x² dx = F(3)−F(0) = 9", fontsize=12)
ax2.set_xlabel("x"); ax2.set_ylabel("f(x)")
ax2.legend(); ax2.grid(True, alpha=0.3)

# 图3：正态分布，[-1,1] 那段填充 = 概率
ax3 = axes[2]
ax3.plot(xs, ys, 'b-', lw=2.5, label='正态分布 PDF')
ax3.fill_between(xs[mask], 0, ys[mask], alpha=0.4, color='blue',
                 label=f'[−1,1] 面积 = {prob_68:.3f}\n(=68% 概率)')
ax3.set_title("概率密度曲线下面积 = 概率\n(±1σ 内约占 68%)", fontsize=12)
ax3.set_xlabel("x"); ax3.set_ylabel("概率密度")
ax3.legend(); ax3.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "13-积分.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")

print("=" * 60)
print("✅ 第13课要点")
print("=" * 60)
print("  • 定积分 = 曲线下面积（小矩形之和，dx→0 的极限）")
print("  • 不定积分 = 导数的逆（找一个 F 使 F'=f）")
print("  • 微积分基本定理：∫ₐᵇ f dx = F(b) − F(a)，算面积变代入")
print("  • 概率密度曲线下的面积 = 概率（整条曲线面积 = 1）")
print()
print("🎯 AI 里的应用：")
print("  • 连续型概率分布（正态/均匀）的概率 = 积分面积")
print("  • 采样、KL 散度、期望值都要用到积分思想")
print("  • 连续型损失（如变分推断 ELBO）也靠积分累积")
print("=" * 60)
