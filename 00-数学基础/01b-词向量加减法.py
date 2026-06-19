"""
数学基础 第1课·补：词向量加减法 king - man + woman ≈ queen
=============================================================
用「手工设计的 2 维坐标」把 word2vec 最经典的等式画出来，让你看见：
  1. 每个词 = 向量空间里的一个点（位置 = 语义）
  2. king − man = 「王权」方向（king 比 man 多的那一段）
  3. 把这个方向「首尾相接」地加到 woman 上 → 正好落在 queen

【为什么用 2 维？】
  真实词向量是几百维的，画不出来；但 king-man+woman 这套加减法的道理，
  在 2 维里一模一样。所以我们手工设计一组坐标，让等式精确成立，方便"看见"。
  坐标轴含义：x 轴 = 性别（男 ↔ 女），y 轴 = 王权（平民 ↔ 皇室）。

参考：
  - Mikolov et al., 2013, "Distributed Representations of Words and Phrases
    and their Compositionality"（word2vec）https://arxiv.org/abs/1310.4546
  - 3Blue1Brown《线性代数的本质》

比喻：词向量空间是一张「语义地图」，加减法 = 在地图上沿某个方向走一段。

【面试高频考点】
  • 为什么 king - man + woman ≈ queen 成立？（向量加减 = 对语义关系做加减）
  • 词向量里的「方向」为什么能编码语义？（分布假说 + 训练逼出来的）

运行：uv run python "00-数学基础/01b-词向量加减法.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体（仓库统一写法）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【0】手工设计 2 维「语义地图」坐标（让等式精确成立）
# ============================================================
#   x 轴：男(+2) ←——性别——→ 女(-2)
#   y 轴：平民(0) ——王权——→ 皇室(4)
man   = np.array([ 2.0, 0.0])   # 男性、平民
woman = np.array([-2.0, 0.0])   # 女性、平民
king  = np.array([ 2.0, 4.0])   # 男性、皇室
queen = np.array([-2.0, 4.0])   # 女性、皇室

royalty = king - man            # = [0, 4]，纯「王权」方向（垂直、无性别成分）

# ============================================================
# 【1】先算一遍，确认等式成立
# ============================================================
result = king - man + woman
print("【1】算一下 king - man + woman")
print(f"    king       = {king}")
print(f"    king - man = {king} - {man} = {king - man}   ← 这就是「王权」方向")
print(f"    再 + woman = {king - man} + {woman} = {result}")
print(f"    queen      = {queen}")
print(f"    是否落在 queen？{np.allclose(result, queen)}（设计上精确相等）\n")

# ============================================================
# 【2】画图工具
# ============================================================
def draw_vec(ax, end, start=(0, 0), color='blue', lw=2.5):
    """画一条从 start 到 end 的箭头。"""
    ax.annotate('', xy=tuple(end), xytext=tuple(start),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color))

def plot_word(ax, p, name, color='black'):
    """在点 p 处画一个圆点 + 词名。"""
    ax.plot(*p, 'o', color=color, markersize=10, zorder=5)
    ax.text(p[0] + 0.18, p[1] - 0.42, name, fontsize=13,
            fontweight='bold', color=color)

def setup(ax, title):
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-1.6, 6.2)
    ax.set_xlabel("性别：男 ←—————→ 女", fontsize=10)
    ax.set_ylabel("王权：平民 → 皇室", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_aspect('equal')

BLUE, RED, GREEN = '#1f77b4', '#d62728', '#2ca02c'

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

# ----- ① 语义地图：4 个词各是空间里的一个点 -----
setup(axes[0], "① 词向量地图：词 = 空间里的一个点")
plot_word(axes[0], man,   "man",   BLUE)
plot_word(axes[0], woman, "woman", RED)
plot_word(axes[0], king,  "king",  BLUE)
plot_word(axes[0], queen, "queen", RED)
axes[0].text(-4.1, 5.7, "位置 = 含义\n越上越「皇」，越右越「男」", fontsize=10, color='gray')

# ----- ② king − man = 「王权」方向 -----
setup(axes[1], "② king − man = 「王权」方向")
plot_word(axes[1], man,  "man",  BLUE)
plot_word(axes[1], king, "king", BLUE)
draw_vec(axes[1], man,  color=BLUE)                 # man 向量（从原点）
draw_vec(axes[1], king, start=man, color=GREEN, lw=3.5)  # 绿箭头：man → king = king-man
axes[1].text(-4.1, 5.4, "绿箭头 = king − man\n= king 比 man 多的那段\n= 纯「王权」方向 [0,4]\n（垂直↑，不含性别成分）",
             fontsize=10, color=GREEN)

# ----- ③ 把王权方向首尾相接加到 woman → queen -----
setup(axes[2], "③ 王权方向 + woman → queen（首尾相接!）")
plot_word(axes[2], woman, "woman", RED)
plot_word(axes[2], queen, "queen", RED)
draw_vec(axes[2], woman, color=RED)                      # woman 向量（从原点）
draw_vec(axes[2], woman + royalty, start=woman, color=GREEN, lw=3.5)  # 同一条绿箭头，接在 woman 尖上
# 标注「首尾相接」的接缝
axes[2].annotate('接缝在这里\n(首尾相接)',
                 xy=tuple(woman), xytext=(-3.9, 2.2),
                 fontsize=9, color=GREEN,
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))
axes[2].text(-1.0, 5.6, "落脚点 = queen ✓", fontsize=12, color=RED, fontweight='bold')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "01b-词向量加减法.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')

print("=" * 55)
print("✅ 怎么看这三幅图")
print("=" * 55)
print("  ① 4 个词是地图上的 4 个点；含义 = 你站在哪")
print("  ② 绿箭头 man→king = king−man = 「王权」方向")
print("  ③ 把这条绿箭头接到 woman 的箭头尖上 → 正好到 queen")
print("     （这就是第1课学的「首尾相接」！）")
print(f"\n图已保存：{out}")
