"""
数学基础 第6课：特征值与特征向量 Eigenvalue & Eigenvector（从零开始）
====================================================================
参考：3Blue1Brown《线性代数的本质》第14、17集、《动手学深度学习》
       https://zh.d2l.ai/ （主成分分析 PCA 章节）

矩阵 A 作用在向量 v 上，通常会把它「又拉又转」。
但有那么一些特殊方向 v：A 作用上去后它【只伸缩、不转向】——
这就是「特征向量」，伸缩的倍数 λ 就是「特征值」。

数学一句话：A·v = λ·v
这个式子是 PCA（主成分分析）的根，也是理解协方差、
图谱、PageRank、振动模式……的万能钥匙。
运行：uv run python "00-数学基础/06-特征值与特征向量.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】定义：A·v = λ·v —— 只伸缩、不转向的方向
# ============================================================
# 造一个矩阵 A，它会把平面「沿 x 方向拉长、并稍微剪切」
A = np.array([[2.0, 0.3],
              [0.0, 1.4]])
print("【1】矩阵 A =")
print(A)
print("    普通向量被 A 作用后：方向会变（又拉又转）")
print("    但「特征向量」只会被拉长/缩短，方向不变！\n")

# ============================================================
# 【2】用 numpy 求：np.linalg.eig
# ============================================================
eigenvalues, eigenvectors = np.linalg.eig(A)
print("【2】用 np.linalg.eig 求特征值 / 特征向量")
for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f"    特征值 λ{i+1} = {lam:+.3f},  对应特征向量 v{i+1} ≈ [{vec[0]:+.3f}, {vec[1]:+.3f}]")
# numpy 返回的列向量通常会被归一化（长度=1），方向才是重点
print("    （特征向量按「方向」理解即可，长度可任意缩放）\n")

# 验证 A·v = λ·v
v1 = eigenvectors[:, 0]
print("    验证 A·v1 与 λ1·v1 是否相等：")
print(f"      A·v1   = {A @ v1}")
print(f"      λ1·v1  = {eigenvalues[0] * v1}")
print("    两者相同 → 证实：A 作用在 v1 上 = 把 v1 缩放 λ1 倍\n")

# ============================================================
# 【3】几何意义：把一堆向量画出来，变换前后对比
# ============================================================
def setup(ax, title, lim=3.2):
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
    ax.set_aspect('equal')

def draw_arrow(ax, end, start=(0, 0), color='blue', label='', lw=2.5, alpha=1.0):
    ax.annotate('', xy=tuple(end), xytext=tuple(start),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color, alpha=alpha))
    if label:
        ax.text(end[0] + 0.1, end[1] + 0.1, label, fontsize=11,
                color=color, fontweight='bold', alpha=alpha)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# 图1：普通向量 vs 特征向量，变换后
setup(axes[0], "① 变换前：特征向量(红/绿) + 一个普通向量(蓝)")
draw_arrow(axes[0], v1, color='red',  label='v1 (特征)')
v2 = eigenvectors[:, 1]
draw_arrow(axes[0], v2, color='green', label='v2 (特征)')
generic = np.array([0.6, 1.0])
draw_arrow(axes[0], generic, color='blue', label='普通向量')

# 图2：变换后 —— 特征向量方向没变，普通向量转向了
setup(axes[1], "② 经 A 变换后：红/绿只变长，蓝转向了！")
draw_arrow(axes[0], v1, color='red', lw=1, label='')      # 重复淡色作参考
g_v1 = A @ v1
g_v2 = A @ v2
g_generic = A @ generic
draw_arrow(axes[1], v1,       color='red',    lw=1,   alpha=0.4)
draw_arrow(axes[1], v2,       color='green',  lw=1,   alpha=0.4)
draw_arrow(axes[1], generic,  color='blue',   lw=1,   alpha=0.4)
draw_arrow(axes[1], g_v1,     color='red',    label='A·v1（同向变长）')
draw_arrow(axes[1], g_v2,     color='green',  label='A·v2（同向变长）')
draw_arrow(axes[1], g_generic, color='blue',  label='A·普通（转向了）')

# 图3：PCA 降维直觉 —— 二维点云，找方差最大(最长伸展)的方向
# 把数据画出来，并把「主成分方向」（协方差矩阵的特征向量）标出
rng = np.random.default_rng(0)
# 造一组沿某斜方向拉长的数据
angle = np.deg2rad(35)
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])
data = rng.standard_normal((200, 2)) @ np.diag([2.5, 0.4]) @ R
mean = data.mean(axis=0)
data_c = data - mean
cov = np.cov(data_c.T)                      # 协方差矩阵
evals, evecs = np.linalg.eigh(cov)          # 对称矩阵用 eigh
order = np.argsort(evals)[::-1]             # 从大到小
evals, evecs = evals[order], evecs[:, order]
print("【3】PCA 直觉：数据的协方差矩阵，其特征向量 = 主成分方向")
print(f"    特征值(方差大小) λ = {evals.round(3)}")
print("    最大特征值方向 = 数据铺得最开的方向 = 第一主成分")
print("    把数据投影到这个方向，就保留了最多的信息\n")

setup(axes[2], "③ PCA 直觉：最大特征值方向 = 数据最「铺开」的方向")
axes[2].scatter(data_c[:, 0], data_c[:, 1], s=10, alpha=0.5, color='steelblue',
                label='数据点')
scale = 3 * np.sqrt(evals)
axes[2].plot([0, evecs[0, 0] * scale[0]], [0, evecs[1, 0] * scale[0]],
             color='red', lw=3, label='第1主成分(方差大)')
axes[2].plot([0, evecs[0, 1] * scale[1]], [0, evecs[1, 1] * scale[1]],
             color='orange', lw=3, label='第2主成分(方差小)')
axes[2].legend(fontsize=9, loc='upper left')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "06-特征值与特征向量.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"    图已保存：{out}\n")

print("=" * 55)
print("✅ 第6课要点")
print("=" * 55)
print("  • 特征向量：被矩阵 A 作用后【只伸缩、不转向】的特殊方向")
print("  • 特征值 λ：伸缩的倍数；满足 A·v = λ·v")
print("  • 求法：np.linalg.eig(A)（对称矩阵用 eigh 更稳）")
print("  • 特征向量按「方向」理解，长度不固定")
print()
print("🎯 AI 里的应用：")
print("  • PCA 降维：协方差矩阵的最大特征值方向 = 信息最多的方向")
print("    → 把高维数据投影到前几个主成分，实现降维 / 可视化")
print("  • 特征值大小 = 该方向上的「重要性 / 方差 / 能量」")
print("  • 谱聚类、PageRank、Fisher 判别都建立在特征分解上")
print("=" * 55)
