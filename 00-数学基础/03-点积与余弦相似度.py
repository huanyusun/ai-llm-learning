"""
数学基础 第3课：点积与余弦相似度 Dot Product & Cosine Similarity（从零开始）
=============================================================================
参考：3Blue1Brown《线性代数的本质》、《动手学深度学习》https://zh.d2l.ai

上节课矩阵乘法里反复出现「行点乘列」的「点积」。本课把「点积」单独讲透，
并引出 NLP/推荐系统里最常用的「余弦相似度」——衡量两个向量方向有多一致。

本课目标：
  1. 点积 = 对应相乘再相加；它和向量「夹角」有关
  2. 余弦相似度 = 两个向量夹角的 cos 值，范围 [-1, 1]
  3. 为什么不直接用点积而要除以长度？→ 排除「长度」干扰，只看方向
  4. 连回 Attention：Q·K 衡量词与词的相似度，本质就是点积

运行：uv run python "00-数学基础/03-点积与余弦相似度.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】点积 = 对应位置相乘，再全部相加
# ============================================================
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)         # 也可以写成 a @ b（一维时等价）
print("【1】点积 a · b")
print(f"    a = {a},  b = {b}")
print(f"    点积 = 1×4 + 2×5 + 3×6 = 4+10+18 = {dot}")
print("    规则一句话：对应相乘，再相加。结果是一个「数」（标量）\n")

# ============================================================
# 【2】点积的几何含义：和夹角有关！
# ============================================================
# 公式：a · b = |a| |b| cosθ
# 所以点积的大小，既取决于向量「长度」，也取决于「夹角」
# 反过来：cosθ = (a·b) / (|a| |b|)
print("【2】点积的几何含义")
print("    a · b = |a| × |b| × cos(夹角θ)")
print("    所以点积同时受『长度』和『夹角』影响")
print("    • 夹角=0°（同向）：cos=1，点积最大且为正")
print("    • 夹角=90°（垂直）：cos=0，点积=0 → 两个方向『没关系』")
print("    • 夹角=180°（反向）：cos=-1，点积最小且为负\n")

# ============================================================
# 【3】余弦相似度：只看「方向」，忽略「长度」
# ============================================================
def cosine_similarity(x, y):
    """cosθ = (x·y) / (|x| |y|)。除以两个长度，就抹掉了大小，只剩方向。"""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

u = np.array([3, 4])       # 方向 ≈ 东北
v = np.array([6, 8])       # 方向同上，但长度是 u 的 2 倍
w = np.array([-4, 3])      # 和 u 接近垂直
print("【3】余弦相似度（范围 -1 到 1）")
print(f"    u = {u},  v = {v}（v 是 u 的 2 倍长，同向）,  w = {w}（接近垂直）")
print(f"    cos(u, v) = {cosine_similarity(u, v):.3f}   → 接近 1，说明方向几乎一样")
print(f"    cos(u, w) = {cosine_similarity(u, w):.3f}   → 接近 0，说明方向几乎无关")
print("    关键：虽然 v 比 u 长一倍，但 cos 仍是 1 —— 余弦相似度『只认方向不认大小』\n")

# 对比点积：点积会被长度带跑
print(f"    对比点积：u·v = {np.dot(u,v)}（被长度放大），但 cos(u,v) = 1.000（稳定）")
print("    → 想比较『方向像不像』时，用余弦相似度，别用点积\n")

# ============================================================
# 【4】画出来：两个向量的夹角 + 相似度
# ============================================================
def setup(ax, title, lim=9):
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
    ax.set_aspect('equal')

def draw_vec(ax, end, color, label):
    ax.annotate('', xy=tuple(end), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color))
    ax.text(end[0] + 0.2, end[1] + 0.2, label, fontsize=12,
            color=color, fontweight='bold')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 图1：同向 → cos 接近 1
setup(axes[0], f"① 同向：cos = {cosine_similarity(u, v):.2f}（很像）")
draw_vec(axes[0], u, 'blue', 'u')
draw_vec(axes[0], v, 'orange', 'v=2u')

# 图2：垂直 → cos 接近 0
setup(axes[1], f"② 垂直：cos = {cosine_similarity(u, w):.2f}（无关）")
draw_vec(axes[1], u, 'blue', 'u')
draw_vec(axes[1], w, 'green', 'w')
# 画一段弧表示直角
angle_u = np.degrees(np.arctan2(*u[::-1]))
angle_w = np.degrees(np.arctan2(*w[::-1]))
arc = np.linspace(np.radians(min(angle_u, angle_w)),
                  np.radians(max(angle_u, angle_w)), 30)
axes[1].plot(1.2*np.cos(arc), 1.2*np.sin(arc), 'r--', lw=1.5)

# 图3：反向 → cos = -1（用 -u 和 v 举例）
neg_u = -u
setup(axes[2], f"③ 反向：cos = {cosine_similarity(neg_u, u):.2f}（完全相反）")
draw_vec(axes[2], u, 'blue', 'u')
draw_vec(axes[2], neg_u, 'red', '-u')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "03-点积与余弦相似度.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【4】图已保存：{out}\n")

# ============================================================
# 【5】连回 AI：Attention 本质就是点积衡量相似度
# ============================================================
print("=" * 55)
print("🎯 连回 Attention：Q · K = 词与词的相似度")
print("=" * 55)
print("  • 注意力打分 scores = Q @ K.T，每个元素就是『一个Q向量 · 一个K向量』")
print("  • 也就是说：Attention 用『点积』衡量『查询词』和『被查词』有多相似")
print("  • 点积越大 → 两个词方向越一致 → 注意力分数越高 → 模型越『关注』它")
print("  • 除以 √d 只是为了防止数值太大导致 softmax 不稳定，方向含义不变\n")

# 迷你 demo：句子「猫 坐 垫子」，看『坐』最关注哪个词
Q = np.array([[1, 0], [1, 1], [0, 1]])   # 猫 / 坐 / 垫子
K = Q.copy()
scores = Q @ K.T
print("  小 demo：句子『猫 / 坐 / 垫子』，每个词 2 维向量")
print("  scores = Q @ K.T ="); print(scores)
print("  看第1行（『坐』这个词作为查询）：", scores[1])
print("  → 『坐』和『垫子』(索引2)的点积最大 = 1，模型会让『坐』更关注『垫子』")
print("=" * 55)

print()
print("=" * 55)
print("✅ 第3课要点")
print("=" * 55)
print("  • 点积 = 对应相乘再相加，结果是一个数")
print("  • 点积几何含义：a·b = |a||b|cosθ，同时受长度和夹角影响")
print("  • 余弦相似度 = cosθ，范围 [-1,1]，1同向 / 0垂直 / -1反向")
print("  • 比方向像不像，用余弦相似度（除掉长度干扰）；点积会被长度带跑")
print("  • Attention 用点积 Q·K 衡量词相似度，分数越大越关注")
print("=" * 55)
