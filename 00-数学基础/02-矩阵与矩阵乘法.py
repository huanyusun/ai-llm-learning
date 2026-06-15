"""
数学基础 第2课：矩阵与矩阵乘法 Matrix & Matrix Multiplication（从零开始）
=========================================================================
参考：3Blue1Brown《线性代数的本质》、《动手学深度学习》https://zh.d2l.ai

上节课学了「向量」（一排数字）。把很多向量摞起来，就成了「矩阵」——一张数字表。
矩阵是深度学习里数据的「标准容器」：一张图片、一批样本、注意力里的 Q/K/V 都是矩阵。

本课目标：
  1. 知道矩阵是「一张数字表」，会看它的形状 (行数, 列数)
  2. 手动验证矩阵乘法 @：用「行点乘列」算出一个元素
  3. 理解转置 .T（行列互换）和单位矩阵（乘法的「1」）
  4. 连回 Attention：Q @ K.T 为什么是「查询和键的相似度表」

运行：uv run python "00-数学基础/02-矩阵与矩阵乘法.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】矩阵 = 一张数字表（二维数组）
# ============================================================
# 一张「2行3列」的表：把它想象成 Excel 的一个区域
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("【1】矩阵 A =")
print(A)
print(f"    形状 shape = {A.shape}  → (行数, 列数) = (2行, 3列)")
print("    比喻：矩阵就是 Excel 的一个数据区域，每个格子装一个数字\n")

# ============================================================
# 【2】矩阵乘法 @ ：核心规则「行点乘列」
# ============================================================
# A 形状 (2,3)，B 形状 (3,2)。能相乘的条件：A的列数 == B的行数（这里是3==3）
# 结果形状：(A的行数, B的列数) = (2, 2)
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])
C = A @ B                 # @ 是 Python/numpy 的矩阵乘法运算符
print("【2】矩阵乘法 A @ B")
print("    A (2×3) @ B (3×2) → C (2×2)，因为 A的列数3 == B的行数3")
print("    规则：结果的第 i 行第 j 列 = A的第 i 行 与 B的第 j 列 做「点积」")
print("    （点积就是对应相乘再相加，上节课学过向量，这里把行和列当向量）\n")

# ---- 手动验证 C 的第一个元素 C[0,0] ----
# 它 = A 的第0行 [1,2,3] · B 的第0列 [7,9,11]
manual = A[0, 0]*B[0, 0] + A[0, 1]*B[1, 0] + A[0, 2]*B[2, 0]
print(f"    手算 C[0,0] = A第0行·B第0列 = 1·7 + 2·9 + 3·11 = {manual}")
print(f"    numpy 算出来 C[0,0] = {C[0, 0]}   ← 两者一致，说明规则理解对了")
print("    完整结果 C =")
print(C, "\n")

# ============================================================
# 【3】转置 .T ：行列互换（把表「翻一下」）
# ============================================================
At = A.T
print("【3】转置 A.T")
print(f"    A 形状 {A.shape} → A.T 形状 {At.shape}")
print("    原来的第0行 [1,2,3] 变成了第0列，相当于把矩阵沿对角线翻一下")
print(At, "\n")

# ============================================================
# 【4】单位矩阵 I ：矩阵乘法的「1」
# ============================================================
# 对角线全是1、其它全是0的方阵。任何矩阵乘它都等于自己（就像数 × 1 = 数）
I = np.eye(3)             # 3×3 单位矩阵
print("【4】单位矩阵 I (3×3)")
print(I)
print("    特性：A @ I = A，就像数字里 a × 1 = a")
print("    验证 A @ I =")
print(A @ I, "  ← 和 A 一模一样\n")

# ============================================================
# 【5】画出来：把矩阵乘法的「行×列」可视化
# ============================================================
def draw_matrix(ax, M, title, cmap='Blues'):
    """把矩阵画成带数字的彩色方块表"""
    ax.set_title(title, fontsize=11)
    rows, cols = M.shape
    ax.matshow(M, cmap=cmap)
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, str(M[i, j]), ha='center', va='center',
                    fontsize=12, color='black', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

draw_matrix(axes[0], A, f"A  {A.shape}", cmap='Blues')
# 中间的运算符
axes[1].text(0.5, 0.5, "@", ha='center', va='center', fontsize=36, fontweight='bold')
axes[1].axis('off')
draw_matrix(axes[2], B, f"B  {B.shape}", cmap='Oranges')
axes[3].text(0.5, 0.5, "=", ha='center', va='center', fontsize=36, fontweight='bold')
axes[3].axis('off')
draw_matrix(axes[4], C, f"C=A@B  {C.shape}", cmap='Greens')

# 在结果图上标出 C[0,0] 是怎么来的
axes[4].add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False,
                                edgecolor='red', lw=3))
fig.suptitle("矩阵乘法：A的第i行 · B的第j列 → C[i,j]   "
             "(红框 C[0,0] = 1·7+2·9+3·11 = 58)", fontsize=13)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "02-矩阵乘法.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")

# ============================================================
# 【6】连回 AI：Attention 里的 Q @ K.T
# ============================================================
print("=" * 55)
print("🎯 连回 Attention：Q @ K.T 是什么？")
print("=" * 55)
print("  • Transformer 的注意力机制核心公式：scores = Q @ K.T / √d")
print("  • Q 是「查询」矩阵，K 是「键」矩阵，每一行都是一个词的向量")
print("  • Q @ K.T 的结果：一个『打分表』，第 i 行第 j 列 = 第 i 个词 和 第 j 个词 有多像")
print("  • 形状直觉：Q(n×d) @ K.T(d×n) → 打分表(n×n)，n 是词的个数")
print("  • 所以『矩阵乘法』在 AI 里 = 批量算『两两有多相似』\n")

# 一个迷你 demo：3 个词，2 维向量
Q = np.array([[1, 0],   # 词1
              [0, 1],   # 词2
              [1, 1]])  # 词3
K = Q.copy()
scores = Q @ K.T        # (3,2) @ (2,3) → (3,3) 两两相似度表
print("  小 demo：3个词的向量，Q @ K.T 得到两两相似度表")
print("  Q ="); print(Q)
print("  scores = Q @ K.T ="); print(scores)
print("  读法：scores[i,j] 越大，词 i 和词 j 越相似（这其实就是『点积』，下节课细讲）")
print("=" * 55)

print()
print("=" * 55)
print("✅ 第2课要点")
print("=" * 55)
print("  • 矩阵 = 一张数字表，形状写成 (行数, 列数)")
print("  • 矩阵乘法 @ 规则：结果[i,j] = A第i行 · B第j列（点积）")
print("  • 乘法前提：A的列数 == B的行数；结果形状 = (A行数, B列数)")
print("  • 转置 .T：行列互换；单位矩阵 I：对角线为1，A@I=A")
print("  • Attention 用 Q @ K.T 批量算两两词的相似度打分")
print("=" * 55)
