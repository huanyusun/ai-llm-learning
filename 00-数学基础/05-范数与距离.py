"""
数学基础 第5课：范数与距离 Norm & Distance（从零开始）
=====================================================
参考：3Blue1Brown《线性代数的本质》、《动手学深度学习》
       https://zh.d2l.ai/ （权重衰减 / L2 正则化章节）

「范数」就是衡量向量「有多长」的规则。换一套规则，
同一个向量可以量出不同的「长度」；而两个向量的「距离」
也只是把这套规则套在它们的差上。

本课画图让你看清 L1、L2 这两套规则长什么样，并解释它们
在 AI 里大名鼎鼎的两个名字：Lasso（L1）、Ridge（L2）正则化。
运行：uv run python "00-数学基础/05-范数与距离.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】范数 = 一把「量长度」的尺子
# ============================================================
# L1 范数：所有分量的「绝对值」加起来（走曼哈顿街区）
# L2 范数：所有分量平方和再开根号（直线距离 / 勾股定理）
v = np.array([3, 4])

l1 = np.sum(np.abs(v))                 # |3| + |4|
l2 = np.linalg.norm(v)                 # √(3² + 4²)
print("【1】范数：用不同尺子量同一个向量 v =", v)
print(f"    L1 范数 = |3| + |4|              = {l1:.1f}   （绝对值之和）")
print(f"    L2 范数 = √(3² + 4²) = √25       = {l2:.1f}   （勾股 / 直线长度）")
print("    同一个向量，换把尺子量出的长度不一样！\n")

# ============================================================
# 【2】距离：把范数套在「两个向量的差」上
# ============================================================
a = np.array([1, 1])
b = np.array([4, 5])
diff = a - b

euclid   = np.linalg.norm(diff)                 # L2 距离 = 欧氏距离
manhatt  = np.sum(np.abs(diff))                 # L1 距离 = 曼哈顿距离
print("【2】两个点之间的距离")
print(f"    a = {a},  b = {b},  差 a-b = {diff}")
print(f"    欧氏距离(L2) = √(3² + 4²)         ≈ {euclid:.3f}   （鸟飞过去，走直线）")
print(f"    曼哈顿距离(L1) = |3| + |4|         = {manhatt:.1f}   （只能在横竖街道上走）")
print("    → 在方格子城市（如曼哈顿）里，横走+竖走才是真实代价\n")

# ============================================================
# 【3】画「单位圆」：范数为 1 的所有点连起来长什么样
# ============================================================
# 这是理解 L1/L2 最直观的图：
#   L2 单位圆 = 真正的圆   （x² + y² = 1）
#   L1 单位圆 = 一个菱形   （|x| + |y| = 1）
theta = np.linspace(0, 2 * np.pi, 400)
circle_l2 = np.array([np.cos(theta), np.sin(theta)])          # 半径1的圆
diamond_l1 = np.array([                             # 菱形 |x|+|y|=1
    np.cos(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta))),
    np.sin(theta) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta))),
])

def setup(ax, title, lim=1.7):
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
    ax.set_aspect('equal')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# 图1：L1 菱形 vs L2 圆
setup(axes[0], "① 范数为 1 的点的集合（单位圆）")
axes[0].plot(diamond_l1[0], diamond_l1[1], color='red',   lw=2.5, label='L1: |x|+|y|=1 (菱形)')
axes[0].plot(circle_l2[0],  circle_l2[1],  color='blue',  lw=2.5, label='L2: √(x²+y²)=1 (圆)')
axes[0].legend(fontsize=9, loc='upper right')

# 图2：两种距离对比（直线 vs 折线）
setup(axes[1], "② 两种距离：欧氏(蓝直线) vs 曼哈顿(红折线)")
axes[1].annotate('', xy=b, xytext=a,
                 arrowprops=dict(arrowstyle='->', lw=2.5, color='blue'))
axes[1].plot([a[0], b[0], b[0]], [a[1], a[1], b[1]],
             color='red', lw=2.5, ls='--')
axes[1].plot([a[0], a[0], b[0]], [a[1], b[1], b[1]],
             color='red', lw=2.5, ls='--', alpha=0.4)
axes[1].plot(*a, 'ko'); axes[1].plot(*b, 'ko')
axes[1].text(a[0]-0.1, a[1]-0.25, 'a', fontsize=12)
axes[1].text(b[0]+0.05, b[1]+0.05, 'b', fontsize=12)
axes[1].text(2.0, 2.6, f'欧氏≈{euclid:.2f}', color='blue', fontsize=10)
axes[1].text(0.2, 4.3, f'曼哈顿={manhatt:.0f}', color='red', fontsize=10)

# 图3：正则化直觉 —— 等高线 vs 约束形状的切点
#   把 L1 / L2 看作「参数 w 能花的预算」，
#   损失的等高线（这里画同心圆示意）和预算形状相切的地方就是解。
setup(axes[2], "③ 正则化直觉：L1 切在角上(稀疏) vs L2 切在边上")
w = np.linspace(-1.5, 1.5, 200)
W1, W2 = np.meshgrid(w, w)
loss = (W1 - 0.9) ** 2 + (W2 - 0.9) ** 2        # 损失等高线（示意）
axes[2].contour(W1, W2, loss, levels=6, colors='gray', alpha=0.6)
axes[2].plot(diamond_l1[0], diamond_l1[1], color='red',  lw=2.5, label='L1 预算')
axes[2].plot(circle_l2[0],  circle_l2[1],  color='blue', lw=2.5, label='L2 预算')
axes[2].plot(0.5, 0.5, 'r*', markersize=18)      # L1 解：撞在角上 → 某维为0
axes[2].plot(0.64, 0.64, 'b*', markersize=18)    # L2 解：贴在圆边上 → 都不为0
axes[2].plot(0.9, 0.9, 'kx', markersize=12, mew=3)  # 没有正则化时的最优解
axes[2].text(0.52, 0.38, 'L1解\n(有维=0)', color='red', fontsize=9)
axes[2].text(0.66, 0.52, 'L2解\n(都不为0)', color='blue', fontsize=9)
axes[2].text(0.92, 0.92, '无正则化\n最优解', fontsize=9)
axes[2].legend(fontsize=9, loc='upper left')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "05-范数与距离.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【3】图已保存：{out}\n")

print("=" * 55)
print("✅ 第5课要点")
print("=" * 55)
print("  • 范数 = 量向量「长度」的尺子；换尺子，长度不同")
print("  • L1 范数 = 各分量绝对值之和；单位圆是【菱形】")
print("  • L2 范数 = 平方和开根号（勾股）；单位圆是【真圆】")
print("  • 距离 = 把范数套在「两向量之差」上")
print("    - 欧氏距离 = L2 = 直线距离")
print("    - 曼哈顿距离 = L1 = 横竖走")
print()
print("🎯 AI 里的应用：")
print("  • L2 正则化 = Ridge（岭回归）：让权重都变小、防过拟合")
print("  • L1 正则化 = Lasso：产生【稀疏】权重（很多直接为0）→ 特征选择")
print("  • 关键直觉：L1 的菱形有「尖角」，损失等高线容易切在角上，")
print("    于是某个维度直接被压成 0 → 自动挑出重要特征")
print("  • KNN、K-Means、相似度检索都靠「距离」来衡量「有多像」")
print("=" * 55)
