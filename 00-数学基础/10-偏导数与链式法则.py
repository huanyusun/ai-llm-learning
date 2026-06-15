"""
数学基础 第10课：偏导数与链式法则 Partial Derivatives & Chain Rule（从零开始）
=============================================================================
参考：3Blue1Brown《微积分的本质》、
      《动手学深度学习》https://zh.d2l.ai

这一课是「反向传播（backprop）」的数学基础，务必理解透。

本课讲三件事：
  1. 偏导数 = 多变量函数「只动一个变量」时的变化率
  2. 梯度 = 把所有偏导数打包成一个向量（指向上坡最快的方向）
  3. 链式法则 = 复合函数 f(g(x)) 的导数 = f'(g)·g'
                → 神经网络层层嵌套，反向传播就是反复用链式法则

可视化：3D 曲面 + 等高线 + 偏导方向箭头，让你看见多变量的「坡」。
运行：uv run python "00-数学基础/10-偏导数与链式法则.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)


# ============================================================
# 【1】偏导数：多变量函数里，只动一个变量看变化率
# ============================================================
# 一个「碗」形损失：L(w1, w2) = w1² + w2²  （两个参数的神经网络损失原型）
def loss2(w1, w2):
    return w1 ** 2 + w2 ** 2


# 偏导数 ∂L/∂w1：把 w2 当常数，只对 w1 求导 → 2·w1
# 偏导数 ∂L/∂w2：把 w1 当常数，只对 w2 求导 → 2·w2
def dL_dw1(w1, w2):
    return 2 * w1


def dL_dw2(w1, w2):
    return 2 * w2


print("【1】偏导数：多变量函数，只动一个变量")
print("    损失 L(w1,w2) = w1² + w2²")
print("    ∂L/∂w1 = 2·w1   （把 w2 当常数，对 w1 求导）")
print("    ∂L/∂w2 = 2·w2\n")

# 数值验证偏导数
def partial(f, idx, args, h=1e-5):
    """对 args 的第 idx 个变量做数值偏导（中心差分）"""
    a = list(args); b = list(args)
    a[idx] += h; b[idx] -= h
    return (f(*a) - f(*b)) / (2 * h)


w1, w2 = 1.5, -0.8
print(f"    数值验证 @ (w1={w1}, w2={w2}):")
print(f"    ∂L/∂w1 ≈ {partial(loss2, 0, (w1, w2)):.6f},  解析 2w1 = {dL_dw1(w1, w2):.6f}")
print(f"    ∂L/∂w2 ≈ {partial(loss2, 1, (w1, w2)):.6f},  解析 2w2 = {dL_dw2(w1, w2):.6f}\n")


# ============================================================
# 【2】梯度：把所有偏导数打包成一个向量 → 指向「上坡最快」方向
# ============================================================
# 梯度 ∇L = (∂L/∂w1, ∂L/∂w2) = (2w1, 2w2)
# 关键性质：梯度指向函数值「上升最快」的方向，
#          所以「负梯度」指向「下降最快」方向 → 这就是梯度下降往哪走！
grad = np.array([dL_dw1(w1, w2), dL_dw2(w1, w2)])
print("【2】梯度 = 偏导数打包成的向量")
print(f"    @ (w1={w1}, w2={w2})，∇L = {grad}")
print(f"    负梯度 -∇L = {-grad}  ← 梯度下降就走这个方向\n")


# ============================================================
# 【3】链式法则：复合函数求导（反向传播的数学灵魂）
# ============================================================
# 复合函数 y = f(g(x))，它的导数 = f'(g(x)) · g'(x)
# 直觉：「外层的变化率」乘「内层的变化率」。
#
# 例子：y = (2x+1)²
#   外层 f(u) = u²   → f'(u) = 2u
#   内层 g(x) = 2x+1 → g'(x) = 2
#   链式：dy/dx = f'(g)·g' = 2(2x+1)·2 = 4(2x+1)
def composite(x):
    return (2 * x + 1) ** 2


def composite_prime(x):
    """链式法则算出来的解析导数：4(2x+1)"""
    return 4 * (2 * x + 1)


def num_deriv(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


print("【3】链式法则：复合函数 f(g(x)) 的导数 = f'(g)·g'")
print("    例子 y=(2x+1)²，拆成 外层 u² + 内层 2x+1")
print("    dy/dx = 2(2x+1)·2 = 4(2x+1)")
for x0 in [0.0, 1.0, -0.5]:
    print(f"    x={x0:+.1f}: 数值≈{num_deriv(composite, x0):.6f}, "
          f"链式法则={composite_prime(x0):.6f}")
print()


# ============================================================
# 【4】链式法则 × 神经网络 = 反向传播（直觉版）
# ============================================================
# 一个最简两层网络前向计算：
#   z  = w·x + b        （线性层）
#   a  = relu(z)        （激活）
#   L  = (a - y)²       （损失，y 是标签）
# 想知道 ∂L/∂w（用来更新 w），要从 L 一路链式传回去：
#   ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w   ← 三段链式相乘
# 这就是「反向传播」：从损失往输入方向，逐层乘偏导。
print("【4】链式法则 → 反向传播（直觉）")
print("    两层网络：z=wx+b → a=relu(z) → L=(a-y)²")
print("    ∂L/∂w = (∂L/∂a)·(∂a/∂z)·(∂z/∂w)   ← 三段链式相乘")
print("    从损失往输入「反向」逐层乘偏导 = backpropagation\n")


# ============================================================
# 【5】画出来：3D 曲面 + 等高线 + 偏导方向 + 链式法则
# ============================================================
fig = plt.figure(figsize=(14, 10))

# 准备网格
W1 = np.linspace(-2, 2, 60)
W2 = np.linspace(-2, 2, 60)
g1, g2 = np.meshgrid(W1, W2)
Lgrid = g1 ** 2 + g2 ** 2

# ---- 图① 3D 曲面：损失函数的「碗」 ----
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(g1, g2, Lgrid, cmap=cm.viridis, alpha=0.85, edgecolor='none')
ax.set_xlabel("w1"); ax.set_ylabel("w2"); ax.set_zlabel("L")
ax.set_title("① 3D 损失曲面 L=w1²+w2²（一个碗）", fontsize=11)

# ---- 图② 等高线 + 梯度（红色，指上坡）+ 负梯度（绿色，指下坡）----
ax = fig.add_subplot(2, 2, 2)
cs = ax.contour(g1, g2, Lgrid, levels=15, cmap='viridis')
ax.clabel(cs, inline=True, fontsize=7)
# 在几个点画梯度箭头
for p in [(-1.2, 0.8), (1.0, 1.0), (-0.6, -1.2), (1.4, -0.5)]:
    gx, gy = dL_dw1(*p), dL_dw2(*p)
    gn = np.hypot(gx, gy) + 1e-9
    # 红色梯度（归一化长度）
    ax.annotate('', xy=(p[0] + gx / gn * 0.35, p[1] + gy / gn * 0.35), xytext=p,
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    # 绿色负梯度（往中心=最低点走）
    ax.annotate('', xy=(p[0] - gx / gn * 0.35, p[1] - gy / gn * 0.35), xytext=p,
                arrowprops=dict(arrowstyle='->', lw=2, color='limegreen'))
ax.plot(0, 0, '*', ms=16, color='white', markeredgecolor='black', label="最低点 (0,0)")
ax.set_title("② 等高线：红=梯度(上坡)  绿=负梯度(下坡)", fontsize=11)
ax.set_xlabel("w1"); ax.set_ylabel("w2"); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
red_patch = plt.Line2D([0], [0], color='red', lw=2, label='梯度 ∇L（上坡）')
green_patch = plt.Line2D([0], [0], color='limegreen', lw=2, label='负梯度 -∇L（下坡）')
ax.legend(handles=[red_patch, green_patch, ax.get_children()[0]] if False else [red_patch, green_patch],
          fontsize=8, loc='upper left')

# ---- 图③ 偏导数的直觉：固定 w2，只看 w1 方向的「切片」----
ax = fig.add_subplot(2, 2, 3)
w2_fixed = 1.0
w1_range = np.linspace(-2, 2, 200)
slice_curve = loss2(w1_range, w2_fixed)
ax.plot(w1_range, slice_curve, lw=2.5, color='blue',
        label=f"切片 L(w1, w2={w2_fixed})")
# 在 w1=1 处画切线 = 偏导数 ∂L/∂w1
w1_0 = 1.0
slope = dL_dw1(w1_0, w2_fixed)
ax.plot(w1_range, slope * (w1_range - w1_0) + loss2(w1_0, w2_fixed),
        '--', lw=2, color='red', label=f"切线，斜率=∂L/∂w1={slope:.0f}")
ax.plot(w1_0, loss2(w1_0, w2_fixed), 'o', ms=10, color='red')
ax.set_title("③ 偏导=切片斜率（固定 w2，只动 w1）", fontsize=11)
ax.set_xlabel("w1"); ax.set_ylabel("L"); ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ---- 图④ 链式法则：复合函数 + 切线 ----
ax = fig.add_subplot(2, 2, 4)
xc = np.linspace(-1.5, 1.5, 200)
ax.plot(xc, composite(xc), lw=2.5, color='purple', label="y=(2x+1)²")
x0 = 0.5
ax.plot(xc, composite_prime(x0) * (xc - x0) + composite(x0),
        '--', lw=2, color='orange',
        label=f"切线 x={x0}, 斜率={composite_prime(x0):.2f}\n(=f'(g)·g'=链式)")
ax.plot(x0, composite(x0), 'o', ms=10, color='orange')
ax.set_title("④ 链式法则：复合函数的切线斜率", fontsize=11)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
out = os.path.join(IMG, "10-偏导数与链式法则.png")
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")


# ============================================================
# 【6】连回 AI：偏导 + 链式 = 反向传播 = 训练神经网络
# ============================================================
print("=" * 55)
print("✅ 第10课要点")
print("=" * 55)
print("  • 偏导数 = 多变量函数里只动一个变量的变化率")
print("  • 梯度 = 所有偏导数组成的向量，指向上坡最快方向")
print("  • 梯度下降：沿「负梯度」走 → 损失下降最快")
print("  • 链式法则：f(g(x))' = f'(g)·g'")
print()
print("🎯 AI 里的应用（本系列的「高潮」）：")
print("  • 反向传播 = 反复用链式法则，逐层算 ∂L/∂w")
print("  • 没有偏导数/梯度，就没法训练任何神经网络")
print("  • PyTorch 的 autograd，做的就是自动链式法则")
print("=" * 55)
