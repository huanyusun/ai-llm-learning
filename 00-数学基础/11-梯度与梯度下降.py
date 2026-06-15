"""
数学基础 第11课：梯度与梯度下降 Gradient & Gradient Descent（从零开始）
=========================================================================
参考：3Blue1Brown《微积分的本质》、《动手学深度学习》https://zh.d2l.ai

上一课学了偏导数（每个方向上的斜率）。把所有方向的偏导数收在一起，
就得到「梯度」——这是神经网络训练里最重要的一个量。

核心思想（3 句话）：
1. 梯度 = 把函数在每个变量上的偏导数排成一个向量。
2. 梯度指向函数值「上升最快」的方向（上山）。
3. 想让损失最小（下山），就朝「负梯度」方向走一步 —— 这就是梯度下降。

本课用 numpy 实现梯度下降，求一个函数的最小值，并把「下山的轨迹」画在等高线图上。
运行：uv run python "00-数学基础/11-梯度与梯度下降.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】梯度是什么？—— 所有偏导数组成的向量
# ============================================================
# 先回忆偏导数：固定其它变量，只对某一个变量求导。
# 把函数 f(x, y) 对 x 的偏导 ∂f/∂x、对 y 的偏导 ∂f/∂y 放在一起：
#     梯度 ∇f = [∂f/∂x, ∂f/∂y]
# 这个向量指向 f 「增加最快」的方向。
#
# 例子（一个经典的「碗」形函数，开口朝上，有最小值）：
#     f(x, y) = x² + y²        （最小值在 (0,0)，f=0）
def f(x, y):
    return x**2 + y**2

# 解析梯度（手工求导）：∂f/∂x = 2x, ∂f/∂y = 2y
def grad_f(x, y):
    return np.array([2 * x, 2 * y])

x0, y0 = 3.0, 4.0
g = grad_f(x0, y0)
print("【1】梯度")
print(f"    函数 f(x,y) = x² + y²")
print(f"    在点 ({x0}, {y0}) 处，梯度 ∇f = [2x, 2y] = {g}")
print("    梯度指向 f 增大最快的方向（从碗底往外、往上爬）")
print("    想『下山』到最小值？那就朝『负梯度』方向走。\n")

# ============================================================
# 【2】梯度下降：朝负梯度方向走一步，重复
# ============================================================
# 更新公式（深度学习每天用的那个）：
#     θ_new = θ_old - 学习率 × 梯度
#                          ↑ lr 控制步子大小
#     注意是「减号」：减去梯度 = 朝上升的反方向 = 下降。

def gradient_descent(start, lr, steps):
    """从 start 出发，做梯度下降，返回走过的所有点。"""
    path = [np.array(start, dtype=float)]
    p = np.array(start, dtype=float)
    for _ in range(steps):
        g = grad_f(p[0], p[1])   # 算当前点的梯度
        p = p - lr * g            # 朝负梯度方向走一步
        path.append(p.copy())
    return np.array(path)

lr = 0.1        # 学习率（步长）
steps = 30
path = gradient_descent(start=(3.0, 4.0), lr=lr, steps=steps)
print("【2】梯度下降（学习率 lr=%.2f，迭代 %d 步）" % (lr, steps))
print(f"    起点: ({path[0,0]:.3f}, {path[0,1]:.3f})，f = {f(*path[0]):.3f}")
print(f"    终点: ({path[-1,0]:.4f}, {path[-1,1]:.4f})，f = {f(*path[-1]):.6f}")
print(f"    （理论最小值在 (0,0)，f=0。看，我们一步步走过去了！）\n")

# ============================================================
# 【3】学习率 lr 的作用：太小→慢，太大→发散
# ============================================================
# 用三种学习率各跑一遍，对比效果。
print("【3】学习率的影响")
for test_lr in [0.01, 0.1, 0.5]:
    p = gradient_descent(start=(3.0, 4.0), lr=test_lr, steps=30)
    final = f(*p[-1])
    print(f"    lr={test_lr:<4} → 30步后 f = {final:.5f}"
          f"{'  (太小，还没走到)' if test_lr < 0.05 else ''}"
          f"{'  (太大，来回震荡)' if test_lr >= 0.45 else ''}")
print()

# ============================================================
# 【4】可视化：等高线 + 下山轨迹
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 左图：等高线 + 梯度下降轨迹 ---
ax = axes[0]
xs = np.linspace(-5, 5, 200)
ys = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

# 画等高线（一圈一圈的，越靠中心 f 越小，就是碗底）
cs = ax.contour(X, Y, Z, levels=15, cmap='viridis')
ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

# 画下降轨迹
ax.plot(path[:, 0], path[:, 1], 'r.-', lw=2, ms=8, label='梯度下降轨迹')
ax.plot(path[0, 0], path[0, 1], 'go', ms=12, label='起点')
ax.plot(0, 0, 'k*', ms=18, label='最小值 (0,0)')

# 在起点处画一个梯度箭头（红色）和负梯度箭头（蓝色）
gx, gy = grad_f(*path[0])
ax.annotate('', xy=(path[0,0]+gx*0.3, path[0,1]+gy*0.3),
            xytext=tuple(path[0]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.annotate('', xy=(path[0,0]-gx*0.3, path[0,1]-gy*0.3),
            xytext=tuple(path[0]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(path[0,0]+gx*0.3+0.1, path[0,1]+gy*0.3+0.1, '梯度∇f\n(上山)', color='red', fontsize=9)
ax.text(path[0,0]-gx*0.3+0.1, path[0,1]-gy*0.3-0.4, '-∇f(下山)', color='blue', fontsize=9)

ax.set_title(f"梯度下降在等高线上的轨迹（lr={lr}）\n一步步『下山』到碗底", fontsize=12)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# --- 右图：f 值随迭代下降 ---
ax2 = axes[1]
f_values = [f(*p) for p in path]
ax2.plot(f_values, 'r.-', lw=2, ms=6)
ax2.set_title("损失 f 随迭代步数下降", fontsize=12)
ax2.set_xlabel("迭代步数"); ax2.set_ylabel("f = x² + y²（损失）")
ax2.grid(True, alpha=0.3)
ax2.annotate('起点损失很高', xy=(0, f_values[0]), xytext=(5, f_values[0]*0.9),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10)
ax2.annotate('接近 0（碗底）', xy=(len(f_values)-1, f_values[-1]),
             xytext=(10, f_values[-1]+3),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "11-梯度与梯度下降.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【4】图已保存：{out}\n")

print("=" * 60)
print("✅ 第11课要点")
print("=" * 60)
print("  • 梯度 ∇f = 所有偏导数组成的向量，指向 f 增大最快的方向")
print("  • 梯度下降：θ ← θ − lr × ∇f   （朝负梯度走，让 f 变小）")
print("  • 学习率 lr 控制步长：太小走得慢，太大会震荡甚至发散")
print("  • 最小值处梯度为 0（平地，没有方向可『下』）")
print()
print("🎯 AI 里的应用（这是神经网络训练的核心）：")
print("  • 把『损失』看成 f，把『模型参数(权重)』看成 (x, y)")
print("  • 训练 = 反复算梯度、朝负梯度走一步 → 让损失越来越小")
print("  • 学习率是最重要的超参数之一（调不好模型就学不会）")
print("  • 《动手学深度学习》3.3 节：随机梯度下降 SGD")
print("=" * 60)
