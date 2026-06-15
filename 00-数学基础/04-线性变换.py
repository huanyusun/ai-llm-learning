"""
数学基础 第4课：线性变换 Linear Transformation（从零开始）
=========================================================
参考：3Blue1Brown《线性代数的本质》、《动手学深度学习》https://zh.d2l.ai

前三课学了向量、矩阵乘法、点积。这节课把「矩阵乘向量」当成一个动作（变换）来看：
一个矩阵乘到向量上，等于把这个向量「扭曲/旋转/拉伸」到新位置。
理解了这个，你就懂了为什么神经网络的每一层都是「矩阵乘法 + 激活函数」。

本课目标：
  1. 「线性变换」= 把空间里的点（向量）按规则搬到新位置，且保持网格平行等距
  2. 矩阵的两列 = 变换后「单位向量 i、j」落在哪里，整个变换由此完全决定
  3. 旋转 / 缩放 / 剪切的几何直觉
  4. 画「原始网格」经矩阵变换后的「变形网格」

运行：uv run python "00-数学基础/04-线性变换.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】什么是「线性变换」？
# ============================================================
print("【1】线性变换 = 把空间『搬动』一下")
print("    『变换』= 一个函数：输入一个向量，输出一个新向量（点搬家了）")
print("    『线性』= 搬完之后，网格线依然保持『平行且等距』（没被揉皱）")
print("    形象比喻：把一张画着坐标格的橡皮膜，整体拉伸/旋转/倾斜，但格线不断不弯\n")

# ============================================================
# 【2】矩阵的两列 = 变换后 i、j 去了哪
# ============================================================
# 单位向量 i=[1,0]（指向右），j=[0,1]（指向上）
# 一个 2×2 矩阵 M = [[a, b], [c, d]]
#   → 它的第0列 [a,c] 就是『变换后 i 的位置』
#   → 它的第1列 [b,d] 就是『变换后 j 的位置』
# 只要记住 i、j 变成了什么，整个空间的每个点怎么变就都定了！
print("【2】矩阵的两列 = 变换后 i、j 的落点")
print("    M = [[a, b],")
print("         [c, d]]")
print("    第0列 [a,c] = 变换后 i(原(1,0)) 的位置")
print("    第1列 [b,d] = 变换后 j(原(0,1)) 的位置")
print("    记住 i、j 去哪了 → 整个空间的变换就完全确定\n")

# ============================================================
# 【3】三个经典变换：缩放 / 旋转 / 剪切
# ============================================================
theta = np.radians(30)  # 旋转 30 度

M_scale = np.array([[2, 0],   # x 方向拉长 2 倍，y 不变
                    [0, 1]])
M_rot = np.array([[np.cos(theta), -np.sin(theta)],   # 逆时针旋转 30°
                  [np.sin(theta),  np.cos(theta)]])
M_shear = np.array([[1, 1],   # 剪切：j 被拽歪，原来竖直的方向变倾斜
                    [0, 1]])

# 验证：取一个向量 v，看它被变换到哪
v = np.array([1.5, 1.0])
print("【3】三个经典变换（用 v=[1.5, 1.0] 验证）")
print(f"    原向量 v = {v}")
print(f"    缩放(×2,×1) 后 M·v = {M_scale @ v}")
print(f"    旋转30°   后 M·v = {np.round(M_rot @ v, 3)}")
print(f"    剪切      后 M·v = {M_shear @ v}\n")

# ============================================================
# 【4】画出来：原始网格 vs 变换后的变形网格
# ============================================================
def setup(ax, title, lim=3.2):
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)

def draw_basis(ax, M, color_i='red', color_j='green'):
    """画出变换后的 i、j（矩阵的两列）"""
    i_new = M[:, 0]
    j_new = M[:, 1]
    ax.annotate('', xy=tuple(i_new), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color_i))
    ax.annotate('', xy=tuple(j_new), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color_j))
    ax.text(i_new[0]+0.05, i_new[1]+0.05, "i'", color=color_i, fontweight='bold')
    ax.text(j_new[0]+0.05, j_new[1]+0.05, "j'", color=color_j, fontweight='bold')

def draw_grid(ax, M, color='blue', n=6):
    """画变换后的网格：把原始水平线/竖直线上的点，都用 M 变换后连起来"""
    t = np.linspace(-n, n, 2*n+1)
    # 竖直线（x 固定，y 变化）变换后
    for x in t:
        pts = np.stack([np.full_like(t, x), t], axis=1)   # (k,2)
        pts = pts @ M.T
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.35, lw=0.8)
    # 横直线（y 固定，x 变化）变换后
    for y in t:
        pts = np.stack([t, np.full_like(t, y)], axis=1)
        pts = pts @ M.T
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.35, lw=0.8)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

# 图0：原始（单位矩阵，啥也不变）
I = np.eye(2)
setup(axes[0], "① 原始空间（单位矩阵 I）")
draw_grid(axes[0], I, color='blue')
draw_basis(axes[0], I)
axes[0].plot(*v, 'ko')  # 原向量
axes[0].text(v[0]+0.05, v[1]+0.05, f"v={v}", fontsize=9)

# 图1：缩放
setup(axes[1], "② 缩放 [[2,0],[0,1]]：横向拉长2倍")
draw_grid(axes[1], M_scale, color='purple')
draw_basis(axes[1], M_scale)
axes[1].plot(*(M_scale @ v), 'ko')
axes[1].text(*(M_scale @ v + [0.1, 0.1]), f"{np.round(M_scale@v,1)}", fontsize=9)

# 图2：旋转
setup(axes[2], "③ 旋转30°：整个网格转个角度")
draw_grid(axes[2], M_rot, color='darkorange')
draw_basis(axes[2], M_rot)
axes[2].plot(*(M_rot @ v), 'ko')
axes[2].text(*(M_rot @ v + [0.1, 0.1]), f"{np.round(M_rot@v,1)}", fontsize=9)

# 图3：剪切
setup(axes[3], "④ 剪切 [[1,1],[0,1]]：竖直方向被推歪")
draw_grid(axes[3], M_shear, color='teal')
draw_basis(axes[3], M_shear)
axes[3].plot(*(M_shear @ v), 'ko')
axes[3].text(*(M_shear @ v + [0.1, 0.1]), f"{M_shear@v}", fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "04-线性变换.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【4】图已保存：{out}\n")

# ============================================================
# 【5】连回 AI：神经网络一层 = 线性变换 + 激活
# ============================================================
print("=" * 55)
print("🎯 连回 AI：神经网络的每一层都是线性变换")
print("=" * 55)
print("  • 全连接层：y = activation(W @ x + b)")
print("    其中 W @ x 就是一个『线性变换』——矩阵 W 把输入向量 x 变换到新空间")
print("  • 多层网络 = 多次线性变换（中间夹激活函数引入非线性）")
print("  • Transformer 里的 Q = X @ W_Q，K = X @ W_K，V = X @ W_V")
print("    → 本质都是用矩阵 W_Q/W_K/W_V 把输入『变换』成查询/键/值向量")
print("  • 词向量也如此：word2vec/Embedding 用一个矩阵把『词』变换成『向量』")
print("=" * 55)

print()
print("=" * 55)
print("✅ 第4课要点")
print("=" * 55)
print("  • 线性变换 = 把空间整体搬动，网格保持平行等距")
print("  • 矩阵的两列 = 变换后单位向量 i、j 的位置，决定整个变换")
print("  • 缩放=对角矩阵；旋转=cos/sin 矩阵；剪切=把某个方向推歪")
print("  • 矩阵乘向量 M·v = 把 v 这个点变换到新位置")
print("  • 神经网络一层 = 线性变换(矩阵乘) + 非线性激活；Attention 的 Q/K/V 都是线性变换")
print("=" * 55)
