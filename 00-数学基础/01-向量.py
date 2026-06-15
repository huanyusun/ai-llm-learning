"""
数学基础 第1课：向量 Vector（从零开始）
=========================================
参考：3Blue1Brown《线性代数的本质》、《动手学深度学习》

向量是 AI 里最基本的东西：词向量、特征、梯度……都是向量。
两个视角理解它：
- 几何：一个带方向的「箭头」（有长度有方向）
- 代数：一排数字 [x, y]，代表箭头终点的坐标

本课用 matplotlib 把向量「画」出来，让你看见它。
运行：uv run python "00-数学基础/01-向量.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】向量 = 一排数字 = 一个箭头
# ============================================================
v = np.array([3, 2])   # 2维向量：向右3格、向上2格
print("【1】向量 v =", v)
print("    几何：从原点(0,0)出发的一个箭头，终点在 (3,2)")
print("    代数：一排数字 [3, 2]\n")

# ============================================================
# 【2】向量加法：对应位置相加 = 几何上「首尾相接」
# ============================================================
u = np.array([1, 3])
s = v + u
print("【2】向量加法")
print(f"    v + u = {v} + {u} = {s}   （对应位置相加：3+1, 2+3）")
print("    几何：先走 v，再从 v 的终点走 u，终点就是 v+u\n")

# ============================================================
# 【3】数乘：把向量拉长/缩短/反向
# ============================================================
k = 2
scaled = k * v
print("【3】数乘")
print(f"    2 × v = {scaled}   （每个数字乘2，向量变长一倍）")
print(f"    负数会反向：-1 × v = {-1 * v}\n")

# ============================================================
# 【4】向量的长度（范数/模长）：勾股定理！
# ============================================================
length = np.linalg.norm(v)
print("【4】向量的长度 |v|")
print(f"    |v| = √(3² + 2²) = √(9+4) = √13 ≈ {length:.3f}")
print("    （就是初中学的勾股定理：直角边平方和开根号）\n")

# ============================================================
# 【5】画出来，建立几何直觉
# ============================================================
def draw_arrow(ax, end, start=(0, 0), color='blue', label=''):
    ax.annotate('', xy=tuple(end), xytext=tuple(start),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color))
    if label:
        ax.text(end[0] + 0.15, end[1] + 0.15, label, fontsize=12,
                color=color, fontweight='bold')

def setup(ax, title, lim=5):
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
    ax.set_aspect('equal')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 图1：一个向量
setup(axes[0], "① 向量 v=[3,2] 就是一个箭头")
draw_arrow(axes[0], v, color='blue', label='v')

# 图2：向量加法（首尾相接 → 绿色和向量）
setup(axes[1], "② 加法 v+u：先走v，再走u")
draw_arrow(axes[1], v, color='blue', label='v')
draw_arrow(axes[1], s, start=v, color='orange', label='u')
draw_arrow(axes[1], s, color='green', label='v+u')

# 图3：数乘 + 长度
setup(axes[2], "③ 数乘 2v（变长）")
draw_arrow(axes[2], v, color='blue', label=f'v (|v|={length:.2f})')
draw_arrow(axes[2], scaled, color='red', label='2v')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "01-向量.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")

print("=" * 55)
print("✅ 第1课要点")
print("=" * 55)
print("  • 向量 = 一排数字 = 一个箭头（有方向有长度）")
print("  • 加法 = 对应位置相加 = 几何上首尾相接")
print("  • 数乘 = 每个数字乘一个数 = 拉长 / 缩短 / 反向")
print("  • 长度(范数) = 勾股定理 = √(各分量平方和)")
print()
print("🎯 AI 里的应用：")
print("  • 词向量 word2vec：把「词」变成一排数字（向量）")
print("  • 经典例子：king - man + woman ≈ queen（向量加减法）")
print("  • 模型的每个特征、每个梯度都是向量")
print("=" * 55)
