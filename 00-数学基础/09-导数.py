"""
数学基础 第9课：导数 Derivative（从零开始）
==========================================
参考：3Blue1Brown《微积分的本质》、
      《动手学深度学习》https://zh.d2l.ai

导数是整个深度学习的「发动机」：梯度下降就是顺着导数（梯度）找最低点。

本课讲四件事：
  1. 导数的直觉 = 「瞬间变化率」 = 切线的斜率
  2. 数值导数 = 用很小的差分去「逼近」导数（代码最常用）
  3. 常见导数公式（x²→2x，eˣ→eˣ …）
  4. 导数法则（加法、乘法、链式 —— 后两个为下一课铺垫）

用 matplotlib 把「函数曲线 + 切线」画出来，让你直观看见斜率。
运行：uv run python "00-数学基础/09-导数.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)


# ============================================================
# 【1】导数的直觉：瞬间变化率 = 切线斜率
# ============================================================
# 想象你在开车，问「这一刻速度多快」？
#   速度 = 位置变化量 / 时间变化量。
# 把时间间隔取得「无穷小」，这个比值就是「瞬时速度」= 导数。
#
# 数学定义：f'(x) = lim(h→0) [ f(x+h) - f(x) ] / h
#   它就是函数在 x 点的「切线斜率」。
print("【1】导数的直觉")
print("    导数 f'(x) = 函数在某点的「瞬间变化率」= 切线的斜率")
print("    定义：f'(x) = lim(h→0) [f(x+h) - f(x)] / h\n")


# ============================================================
# 【2】数值导数：用「很小的 h」近似导数（工程里最常用）
# ============================================================
def num_derivative(f, x, h=1e-5):
    """数值导数（中心差分）：比单侧差分更准，误差是 O(h²)"""
    return (f(x + h) - f(x - h)) / (2 * h)


def f_square(x):
    return x ** 2


print("【2】数值导数：用极小的 h 逼近极限（代码里就这么算）")
print("    对 f(x)=x²，理论导数 f'(x)=2x")
for x0 in [1.0, 2.0, -3.0]:
    approx = num_derivative(f_square, x0)
    exact = 2 * x0
    print(f"    x={x0:+.0f}: 数值导数≈{approx:.6f},  理论值=2x={exact:+.1f}")
print()


# ============================================================
# 【3】常见导数公式（背几个就够用很久）
# ============================================================
print("【3】常见导数公式（神经网络里反复出现）")
print("    • 常数 c        → 0")
print("    • 幂函数 xⁿ     → n·xⁿ⁻¹      （特例 x² → 2x）")
print("    • 指数 eˣ       → eˣ           （导数=自己，eˣ 的神奇性质）")
print("    • 对数 ln(x)    → 1/x")
print("    • sin(x)        → cos(x)")
print("    • sigmoid σ(x) → σ(x)·(1-σ(x))（这就是它好用的原因）\n")


# ============================================================
# 【4】导数法则：把复杂函数拆开求导
# ============================================================
print("【4】导数基本法则")
print("    • 加法法则： (f+g)' = f' + g'")
print("    • 乘法法则： (f·g)' = f'·g + f·g'")
print("    • 链式法则： (f(g(x)))' = f'(g)·g'   ← 下一课详细讲，反向传播的灵魂")
print()


# ============================================================
# 【5】画出来：函数曲线 + 切线 = 看见斜率
# ============================================================
def tangent_line(f, fprime, x0, xs):
    """在 x0 处画切线：y = f(x0) + f'(x0)·(x - x0)"""
    return fprime(x0) * (xs - x0) + f(x0)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
xs = np.linspace(-3, 3, 400)

# ---- 图① x² 与三处切线：斜率随位置变化 ----
ax = axes[0, 0]
ax.plot(xs, f_square(xs), label="y=x²", lw=2.5, color='blue')
for x0, c in [(-1.5, 'green'), (0.0, 'orange'), (1.5, 'red')]:
    slope = 2 * x0
    ax.plot(xs, tangent_line(f_square, lambda t: 2 * t, x0, xs),
            '--', lw=1.8, color=c, label=f"切线 x={x0:+.1f}, 斜率={slope:+.1f}")
ax.set_title("① y=x² 与切线：斜率(导数)=2x 随位置变化", fontsize=12)
ax.set_ylim(-1, 6); ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9)

# ---- 图② eˣ：导数=自己，每处切线斜率都=该点高度 ----
ax = axes[0, 1]
f_exp = np.exp
xs2 = np.linspace(-2, 1.2, 400)
ax.plot(xs2, f_exp(xs2), label="y=eˣ", lw=2.5, color='purple')
for x0 in [-1.0, 0.0, 0.8]:
    ax.plot(xs2, tangent_line(f_exp, lambda t: np.exp(t), x0, xs2),
            '--', lw=1.8, label=f"切线 x={x0:+.1f}, 斜率={np.exp(x0):.2f}")
ax.set_title("② y=eˣ：导数=自己（切线斜率=该点高度）", fontsize=12)
ax.set_ylim(-0.5, 4); ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9)

# ---- 图③ 损失函数的「碗」：导数告诉你往哪走能下降 ----
ax = axes[1, 0]
# 一个二次型损失 L(w) = (w-1)² + 0.5，最低点在 w=1
def loss(w):
    return (w - 1) ** 2 + 0.5
def loss_prime(w):
    return 2 * (w - 1)
ws = np.linspace(-2, 4, 400)
ax.plot(ws, loss(ws), label="损失 L(w)=(w-1)²+0.5", lw=2.5, color='teal')
ax.plot(1, loss(1), '*', ms=18, color='red', label="最低点 w=1（导数=0）")
# 在 w=-0.5 处画切线：斜率为负 → 往右走能下降
w0 = -0.5
ax.plot(ws, tangent_line(loss, loss_prime, w0, ws), '--', lw=2, color='orange',
        label=f"切线 w={w0}, 斜率={loss_prime(w0):+.1f}（负→往右下走）")
# 用箭头表示梯度下降方向（负梯度方向）
ax.annotate('', xy=(0.8, loss(0.8)), xytext=(-0.5, loss(-0.5)),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))
ax.set_title("③ 损失函数的「碗」+ 导数 = 梯度下降", fontsize=12)
ax.set_ylim(0, 6); ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9, loc='upper right')

# ---- 图④ 数值导数 vs 解析导数：几乎重合 ----
ax = axes[1, 1]
f_cube = lambda t: t ** 3            # 解析导数 3x²
analytic = 3 * xs ** 2
numeric = np.array([num_derivative(f_cube, t) for t in xs])
ax.plot(xs, analytic, lw=4, alpha=0.4, color='blue', label="解析导数 3x²")
ax.plot(xs, numeric, '--', lw=2, color='red', label="数值导数（差分近似）")
ax.set_title("④ 数值导数 ≈ 解析导数（差分法够用）", fontsize=12)
ax.set_ylim(-2, 12); ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(IMG, "09-导数.png")
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")


# ============================================================
# 【6】连回 AI：导数 = 梯度下降的「方向盘」
# ============================================================
print("=" * 55)
print("✅ 第9课要点")
print("=" * 55)
print("  • 导数 = 瞬间变化率 = 切线斜率")
print("  • 数值导数：用极小 h 做 [f(x+h)-f(x-h)]/(2h)，工程上够用")
print("  • x²→2x、eˣ→eˣ、ln(x)→1/x，记几个就够")
print("  • 法则：加法、乘法、链式（下一课重点）")
print()
print("🎯 AI 里的应用：")
print("  • 损失函数 L(w) 是个「碗」，导数告诉你坡度")
print("  • 梯度下降：w ← w - η·L'(w)   （η 是学习率）")
print("  • 顺着负导数方向走，就能走到损失最小的点 = 训练")
print("=" * 55)
