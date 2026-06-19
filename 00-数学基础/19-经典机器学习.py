"""19 - 经典机器学习三件套（从零，纯 numpy）—— 线性回归 / 逻辑回归+评估指标 / KMeans

全景图块一「数学与机器学习基础」里的【经典 ML】此前一个项目都没有，本文件补上。
三个最常考的经典算法，全部 numpy 从零实现，和 11-梯度下降、18-交叉熵 串起来：

    ① 线性回归（监督·连续）：梯度下降拟合 y = wx + b  → 连回 11-梯度与梯度下降
    ② 逻辑回归（监督·分类）：sigmoid + 交叉熵 + 梯度下降 → 连回 18-交叉熵
       附评估指标：精确率 P / 召回率 R / F1（分类模型必考）
    ③ KMeans（无监督·聚类）：迭代「分配→更新中心」

【为什么从零写而不是 sklearn？】
    面试考的是【你懂不懂原理】，不是会不会调 API。从零写一遍，梯度下降、sigmoid、
    损失怎么来的、P/R/F1 怎么算，全透了。生产里再用 sklearn 一行调。

【运行】uv run "00-数学基础/19-经典机器学习.py"   （依赖只有 numpy）
"""

from __future__ import annotations

import numpy as np

np.set_printoptions(precision=3, suppress=True)


# ============================================================================
# ① 线性回归：梯度下降拟合 y = wx + b
# ============================================================================
def demo_linear_regression():
    print("\n" + "=" * 60)
    print(" ① 线性回归（梯度下降拟合 y = wx + b）")
    print("=" * 60)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 50)
    y = 2.0 * x + 3.0 + rng.normal(0, 0.1, size=50)  # 真值 w=2, b=3
    print("  真值：w=2.0, b=3.0（加了高斯噪声）")

    w, b, lr = 0.0, 0.0, 0.1
    for _ in range(500):
        pred = w * x + b
        err = pred - y
        # 均方误差 MSE = mean(err²) 对 w/b 的梯度：2*mean(err*x), 2*mean(err)
        w -= lr * 2 * np.mean(err * x)
        b -= lr * 2 * np.mean(err)
    print(f"  学到：w={w:.3f}  b={b:.3f}  （应接近 2.0 / 3.0）")
    print("  → 梯度下降把参数拉到真值；这是所有「用梯度学参数」算法的最小形态。")


# ============================================================================
# ② 逻辑回归（分类）+ 评估指标 P/R/F1
# ============================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def precision_recall_f1(y_true, y_pred):
    """精确率/召回率/F1。y_true, y_pred 都是 0/1 数组。"""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0.0   # 预测为正里，真为正的比例
    r = tp / (tp + fn) if (tp + fn) else 0.0   # 真为正里，被找出来的比例
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def demo_logistic_regression():
    print("\n" + "=" * 60)
    print(" ② 逻辑回归（分类）+ 评估指标 P/R/F1")
    print("=" * 60)
    rng = np.random.default_rng(1)
    # 两类点：0 类在左下，1 类在右上
    n = 40
    X0 = rng.normal([1, 1], 0.6, (n, 2));   y0 = np.zeros(n)
    X1 = rng.normal([4, 4], 0.6, (n, 2));   y1 = np.ones(n)
    X = np.vstack([X0, X1]);  Y = np.concatenate([y0, y1])
    # 加偏置列
    Xb = np.hstack([X, np.ones((len(X), 1))])

    w = np.zeros(3)
    lr = 0.1
    for _ in range(500):
        p = sigmoid(Xb @ w)
        # 交叉熵损失对 w 的梯度：Xᵀ(p - y) / n   （连回 18-交叉熵）
        grad = Xb.T @ (p - Y) / len(Y)
        w -= lr * grad

    probs = sigmoid(Xb @ w)
    preds = (probs >= 0.5).astype(int)
    p, r, f1 = precision_recall_f1(Y.astype(int), preds)
    acc = float(np.mean(preds == Y.astype(int)))
    print(f"  学到权重 w={w.round(2)}（第 3 维是偏置）")
    print(f"  准确率 acc={acc:.3f}")
    print(f"  精确率 P={p:.3f}  召回率 R={r:.3f}  F1={f1:.3f}")
    print("  → P=预测为正里真为正的比例；R=真为正里被找出来的比例；F1=二者的调和平均。")
    print("    AUC：把'阈值从 1 扫到 0 画 ROC 曲线下的面积'，衡量排序能力（不依赖阈值）。")


# ============================================================================
# ③ KMeans 聚类（无监督）
# ============================================================================
def demo_kmeans():
    print("\n" + "=" * 60)
    print(" ③ KMeans 聚类（无监督：分配 → 更新中心）")
    print("=" * 60)
    rng = np.random.default_rng(2)
    pts = np.vstack([rng.normal([0, 0], 0.5, (30, 2)),
                     rng.normal([5, 5], 0.5, (30, 2)),
                     rng.normal([0, 5], 0.5, (30, 2))])
    k = 3
    # k-means++ 初始化：避免随机初始化"两个中心落进同一团"导致烂局部最优。
    # 第一个中心随机选；后续选"离已有中心越远（距离平方越大）越可能被选"的点。
    centroids = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):
        d2 = np.min(np.stack([np.sum((pts - c) ** 2, axis=1) for c in centroids]), axis=0)
        centroids.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    centroids = np.array(centroids, dtype=float)

    for it in range(10):
        # 分配：每个点归到最近中心
        d = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)  # [N,k]
        labels = d.argmin(1)
        # 更新中心：每簇点的均值
        new_c = np.array([pts[labels == j].mean(0) if (labels == j).any() else centroids[j]
                          for j in range(k)])
        if np.allclose(new_c, centroids):
            break
        centroids = new_c
    counts = [int((labels == j).sum()) for j in range(k)]
    print(f"  聚类收敛（{it+1} 次迭代），3 簇各 {counts} 个点")
    print(f"  中心 ≈ {centroids.round(1).tolist()}（应接近 [0,0]/[5,5]/[0,5]）")
    print("  → 无监督：没有标签，靠'距离最近'自动成团；EM 思想的最小形态。")


def main():
    print("=" * 60)
    print(" 经典机器学习三件套（从零，numpy）")
    print("=" * 60)
    demo_linear_regression()
    demo_logistic_regression()
    demo_kmeans()
    print("\n" + "=" * 60)
    print(" 跑通 ✓  → 线性/逻辑回归（监督）+ KMeans（无监督）+ P/R/F1")
    print(" 生产里换 sklearn 一行调，但面试考的就是这些原理。")
    print("=" * 60)


if __name__ == "__main__":
    main()
