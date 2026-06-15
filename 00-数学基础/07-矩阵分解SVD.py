"""
数学基础 第7课：矩阵分解 SVD 奇异值分解（从零开始）
====================================================
参考：3Blue1Brown《线性代数的本质》第18-22集、《动手学深度学习》
       https://zh.d2l.ai/ （主成分分析 / 降维章节）

任意一个矩阵 A（哪怕是「长方形」的、哪怕不是方阵！），
都能被拆成三块：

        A  =  U  ·  Σ  ·  Vᵀ

 - U、V 是「旋转 / 镜像」（正交矩阵，不改变长度）
 - Σ  是「对角矩阵」，对角线上排着【奇异值】（从大到小）

直觉：任何线性变换 = 旋转 → 沿坐标轴伸缩 → 再旋转。
而伸缩的「力度」就存在奇异值里：大奇异值 = 重要方向，
小奇异值 = 可以扔掉。这就是降维、压缩、推荐系统的根。
运行：uv run python "00-数学基础/07-矩阵分解SVD.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】任意矩阵都能拆：A = U·Σ·Vᵀ
# ============================================================
rng = np.random.default_rng(42)
A = rng.standard_normal((4, 6))   # 一个 4×6 的「长方形」矩阵
print("【1】任意长方形矩阵都能 SVD 分解")
print("    原矩阵 A 的形状 =", A.shape, "（行≠列，也能分解！）")

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"    U 形状 = {U.shape}   （左奇异向量，像「旋转」）")
print(f"    Σ 奇异值 = {np.round(S, 3)}   （从大到小排，全是非负数）")
print(f"    Vᵀ 形状 = {Vt.shape}  （右奇异向量，也像「旋转」）\n")

# 重建验证
A_rebuild = U @ np.diag(S) @ Vt
print(f"    重建 U·Σ·Vᵀ 与原 A 最大误差 = {np.max(np.abs(A - A_rebuild)):.2e}")
print("    （误差是浮点精度级别 → 证明 A = U·Σ·Vᵀ）\n")

# ============================================================
# 【2】降维：只保留前 k 个最大的奇异值
# ============================================================
print("【2】降维 = 只留前 k 个最大的奇异值")
energy = S ** 2
total = energy.sum()
for k in [1, 2, 3, 4]:
    keep = energy[:k].sum() / total * 100
    print(f"    保留前 {k} 个奇异值 → 捕获 {keep:5.1f}% 的「能量/信息」")
print("    → 大奇异值集中了绝大部分信息，小奇异值可以丢掉\n")

# ============================================================
# 【3】可视化：用 SVD 给「图像」降维（数据压缩）
# ============================================================
# 造一张简单的灰度「图」（带斜条纹的图案），用低秩 SVD 近似
x = np.linspace(-3, 3, 120)
y = np.linspace(-3, 3, 90)
X, Y = np.meshgrid(x, y)
img = (np.sin(X) * np.exp(-Y**2 / 6))          # 90×120 的灰度图
img += 0.3 * np.cos(X * 2 + Y)
img = (img - img.min()) / np.ptp(img)          # 归一化到 0~1

U2, S2, Vt2 = np.linalg.svd(img, full_matrices=False)

def approx_rank(k):
    """用前 k 个奇异值重建图像（截断 SVD）"""
    return U2[:, :k] @ np.diag(S2[:k]) @ Vt2[:k, :]

ks = [1, 3, 8, 20]
fig, axes = plt.subplots(1, len(ks) + 1, figsize=(16, 3.6))

# 第一张：原图
axes[0].imshow(img, cmap='gray')
axes[0].set_title("原图\n(秩很高)", fontsize=11)
axes[0].axis('off')

# 其余：用不同 k 重建
for ax, k in zip(axes[1:], ks):
    rec = approx_rank(k)
    info = (S2[:k] ** 2).sum() / (S2 ** 2).sum() * 100
    ax.imshow(rec, cmap='gray')
    ax.set_title(f"保留 k={k} 个奇异值\n信息 {info:.0f}%  | 存储↓{(1 - k*(90+120+1)/(90*120))*100:.0f}%",
                 fontsize=10)
    ax.axis('off')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "07-矩阵分解SVD.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【3】图已保存：{out}\n")

print("=" * 55)
print("✅ 第7课要点")
print("=" * 55)
print("  • 任意矩阵都能分解：A = U·Σ·Vᵀ（连长方形矩阵都行）")
print("  • U、V 是「旋转」，Σ 对角线上的【奇异值】是「伸缩力度」")
print("  • 奇异值按从大到小排：大 = 重要方向，小 = 噪声/可丢弃")
print("  • 截断 SVD：只留前 k 个奇异值 → 既降维又压缩")
print()
print("🎯 AI 里的应用：")
print("  • 数据压缩：图/视频的「低秩近似」就是丢小奇异值")
print("  • 推荐系统：把「用户×电影」打分矩阵 SVD 分解，")
print("    得到低维的「用户特征」和「物品特征」，预测你没看的电影")
print("  • 降维 / 去噪：PCA 本质就是 SVD（在协方差矩阵上）")
print("  • LLM 里：嵌入矩阵、权重矩阵的「秩」影响模型容量")
print("=" * 55)
