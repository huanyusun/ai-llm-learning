"""CLIP 式图文对齐（从零，纯 numpy）—— 多模态的核心思想

全景图块九「多模态」此前一个项目都没有，本文件补上。多模态大模型（CLIP/BLIP/GPT-4V）
的根基是：把【图像】和【文本】映射到同一个向量空间，用【对比学习】让"匹配的图文对"
靠近、"不匹配的"拉远。本文件用最小例子把这个思想跑通。

【三个核心知识点（面试能讲清）】
    1. 共享嵌入空间：图像和文本各自编码成同维向量，落在同一空间，才能直接比相似度。
    2. 对比损失（InfoNCE）：对 N 个图文对，让【对角线】（匹配对）的相似度高于其他。
       一句话：正确的图文配对应比错误的配对更相似。
    3. 零样本分类（zero-shot）：拿一张图，和一串"候选文本"算相似度，最近的那个就是分类
       ——不用为每个任务训练分类头，这正是 CLIP 的革命性。

【本文件怎么演示（不用真训练，靠"对齐 vs 未对齐"对照）】
    - 把图/文的语义直接表示在【同一组语义维度】上（红/蓝/圆/方），这就是"已对齐"的共享空间。
    - 对照：把文本过一个【随机投影】→ 空间错位 → 对比损失飙升、对角线不再最大。
    - 真实 CLIP 用亿级图文对训练两个编码器（图像编码器 + 文本编码器）+ 对比损失，
      把随机投影"学"成对齐投影。本文件展示的是【训练要达到的效果】。

【运行】uv run "07-多模态/clip图文对齐.py"   （依赖只有 numpy）
"""

from __future__ import annotations

import numpy as np

np.set_printoptions(precision=2, suppress=True)

# 语义维度 = [红, 蓝, 圆, 方]
DIMS = ["红", "蓝", "圆", "方"]
# 4 张"图像"：各自在这 4 个语义维度上的特征（有没有该属性）
IMAGES = {
    "红圆": np.array([1.0, 0.0, 1.0, 0.0]),
    "蓝方": np.array([0.0, 1.0, 0.0, 1.0]),
    "红方": np.array([1.0, 0.0, 0.0, 1.0]),
    "蓝圆": np.array([0.0, 1.0, 1.0, 0.0]),
}
# 4 句"文本"（与图像一一匹配，语义向量相同 → 这就是"对齐的共享空间"）
CAPTIONS = {
    "一张红圆":   np.array([1.0, 0.0, 1.0, 0.0]),
    "一张蓝方":   np.array([0.0, 1.0, 0.0, 1.0]),
    "一张红方":   np.array([1.0, 0.0, 0.0, 1.0]),
    "一张蓝圆":   np.array([0.0, 1.0, 1.0, 0.0]),
}
img_names = list(IMAGES)
cap_names = list(CAPTIONS)
IMG = np.stack([IMAGES[n] for n in img_names])   # [4,4]
CAP = np.stack([CAPTIONS[n] for n in cap_names])  # [4,4]


def normalize(x: np.ndarray) -> np.ndarray:
    """L2 归一化：归一化后点积 = 余弦相似度（连回 00-数学基础/03-点积与余弦相似度）。"""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


def sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a/b 已归一化时，a @ bᵀ 就是 N×N 的余弦相似度矩阵。"""
    return normalize(a) @ normalize(b).T


def contrastive_loss(sim: np.ndarray) -> float:
    """InfoNCE：对角线（匹配对）应最大。= mean( logsumexp(行) - 对角线 )，越小越好。"""
    sim = sim - sim.max(axis=1, keepdims=True)          # 数值稳定
    log_denom = np.log(np.exp(sim).sum(axis=1))
    diag = sim[np.arange(len(sim)), np.arange(len(sim))]
    return float(np.mean(log_denom - diag))


def section_aligned():
    print("\n" + "=" * 60)
    print(" ① 对齐的共享空间：图文相似度矩阵（对角线 = 匹配对）")
    print("=" * 60)
    sim = sim_matrix(IMG, CAP)
    print("  行=图像, 列=文本, 值=余弦相似度：")
    print("        " + "  ".join(f"{c:>6}" for c in cap_names))
    for i, name in enumerate(img_names):
        print(f"  {name:>4} " + "  ".join(f"{sim[i,j]:>6.2f}" for j in range(len(cap_names))))
    print(f"\n  对比损失(InfoNCE) = {contrastive_loss(sim):.3f}  （越小越好；对齐后明显低于未对齐，见②）")
    diag_max = np.all(sim[np.arange(4), np.arange(4)] >= sim.max(axis=1) - 1e-9)
    print(f"  对角线是否每行最大：{diag_max}  → 匹配图文对最相似 ✓")


def section_unaligned():
    print("\n" + "=" * 60)
    print(" ② 未对齐（随机投影）：对比损失飙升、对角线不再最大")
    print("=" * 60)
    rng = np.random.default_rng(7)
    proj = rng.normal(0, 1, (4, 4))          # 随机投影：把文本映射到错位的空间
    sim = sim_matrix(IMG, CAP @ proj)
    print("  文本过随机投影后的相似度矩阵（对角线不再突出）：")
    for i, name in enumerate(img_names):
        print(f"  {name:>4} " + "  ".join(f"{sim[i,j]:>6.2f}" for j in range(len(cap_names))))
    print(f"\n  对比损失 = {contrastive_loss(sim):.3f}  （比对齐时更高 → 投影错位）")
    print("  → CLIP 训练就是在做这件事的反面：学一个投影，把损失从高降到低。")


def section_zeroshot():
    print("\n" + "=" * 60)
    print(" ③ 零样本分类：新文本 → 最近的图像（CLIP 的革命性）")
    print("=" * 60)
    queries = {
        "蓝色的圆": np.array([0.0, 1.0, 1.0, 0.0]),
        "红色的方": np.array([1.0, 0.0, 0.0, 1.0]),
    }
    for q_text, q_vec in queries.items():
        sims = normalize(IMG) @ normalize(q_vec)
        best = img_names[int(sims.argmax())]
        print(f"  文本「{q_text}」→ 各图像相似度 "
              f"{dict(zip(img_names, np.round(sims, 2).tolist()))}  → 最像【{best}】")
    print("  → 不用专门训练分类头：靠图文共享空间的相似度就能分类，这就是 zero-shot。")


def main():
    print("=" * 60)
    print(" CLIP 式图文对齐（从零，numpy）—— 多模态核心思想")
    print(" 语义维度:", DIMS)
    print("=" * 60)
    section_aligned()
    section_unaligned()
    section_zeroshot()
    print("\n" + "=" * 60)
    print(" 跑通 ✓  → 共享空间 + 对比损失(InfoNCE) + 零样本分类")
    print(" 真实 CLIP：图像编码器(ResNet/ViT) + 文本编码器(Transformer) + 亿级图文对对比训练。")
    print("=" * 60)


if __name__ == "__main__":
    main()
