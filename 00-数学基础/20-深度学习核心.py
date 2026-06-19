"""20 - 深度学习核心：激活函数 / 优化器 / CNN / RNN（从零，纯 numpy）

全景图块二「深度学习核心」此前缺：激活函数(ReLU/GELU/Sigmoid)、优化器(SGD/Adam)、
CNN(卷积/池化)、RNN/LSTM。本文件一次补齐，全部 numpy 从零，和已有的 12-反向传播 串成
"深度学习怎么搭怎么训"的完整图景。

    ① 激活函数：sigmoid / tanh / ReLU / GELU —— 为什么 ReLU/GELU 干掉了 sigmoid（梯度消失）
    ② 优化器：SGD vs Adam —— Adam 为什么收敛更快（动量 + 自适应学习率）
    ③ CNN：conv2d 卷积 + maxpool 池化前向 —— 卷积怎么提特征、池化怎么降维
    ④ RNN：一个 RNN cell 的前向 —— 怎么处理序列、隐状态怎么传

【为什么前向用 numpy 手写？】面试常考"卷积/池化具体怎么算""RNN cell 公式"，
从零写一遍就真懂了；反向靠 PyTorch（见 06/00-pytorch基础）。

【运行】uv run "00-数学基础/20-深度学习核心.py"   （依赖只有 numpy）
"""

from __future__ import annotations

import numpy as np

np.set_printoptions(precision=3, suppress=True)


# ============================================================================
# ① 激活函数：sigmoid / tanh / ReLU / GELU
# ============================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def relu(z):
    return np.maximum(0.0, z)


def gelu(z):
    # GELU ≈ x · Φ(x)，用 tanh 近似
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


def demo_activations():
    print("\n" + "=" * 60)
    print(" ① 激活函数：sigmoid / tanh / ReLU / GELU")
    print("=" * 60)
    xs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    tanh = np.tanh(xs)
    print(f"  输入 x      : {xs}")
    print(f"  sigmoid(x)  : {sigmoid(xs)}   ← 饱和(两端→0/1)，导数→0 → 【梯度消失】")
    print(f"  tanh(x)     : {tanh}   ← 零中心，仍会饱和")
    print(f"  ReLU(x)     : {relu(xs)}   ← 正区导数恒1(不消失)、负区置0(可能「死」)")
    print(f"  GELU(x)     : {gelu(xs).round(3)}   ← 平滑版 ReLU，现代大模型(GPT)默认")
    print("  → 现代网络默认 ReLU/GELU：避免 sigmoid 的梯度消失，深层才训得动。")


# ============================================================================
# ② 优化器：SGD vs Adam（在 f(x,y)=x² + 10y² 上看收敛速度）
# ============================================================================
def demo_optimizers():
    print("\n" + "=" * 60)
    print(" ② 优化器：SGD vs Adam")
    print("=" * 60)
    # 病态目标：min f(x,y)=x² + 50·y²（y 维比 x 维陡 50 倍，条件数大）
    # 这种"各方向陡峭程度差很多"的曲面，最能体现优化器的差别。
    def grad(p):
        return np.array([2 * p[0], 100 * p[1]])

    def run_sgd(x0, lr=0.009, steps=60):
        # SGD 用同一个 lr：为不在陡峭的 y 维上发散，lr 必须很小 → 平坦的 x 维就爬得极慢。
        p = x0.copy()
        for _ in range(steps):
            p -= lr * grad(p)
        return p

    def run_adam(x0, lr=0.3, steps=60, b1=0.9, b2=0.999, eps=1e-8):
        # Adam：每个参数有自己的有效学习率（≈lr/sqrt(历史梯度平方)），陡/平两维都能前进。
        p = x0.copy()
        m = np.zeros_like(p); v = np.zeros_like(p)
        for t in range(1, steps + 1):
            g = grad(p)
            m = b1 * m + (1 - b1) * g                 # 一阶动量（梯度的指数滑动均值）
            v = b2 * v + (1 - b2) * g * g              # 二阶动量（梯度平方的滑动均值）
            mhat = m / (1 - b1 ** t)                   # 偏差修正
            vhat = v / (1 - b2 ** t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)     # 每个参数自适应学习率
        return p

    x0 = np.array([6.0, 6.0])
    p_sgd = run_sgd(x0)
    p_adam = run_adam(x0)
    print(f"  目标 min x²+50y²，最优 (0,0)，起点 (6,6)，各跑 60 步：")
    print(f"  SGD  终点 = ({p_sgd[0]:.2f}, {p_sgd[1]:.2f})  到最优点距离 {np.linalg.norm(p_sgd):.2f}")
    print(f"        ↑ 小 lr 保 y 不发散 → x 维爬得慢，远没收敛")
    print(f"  Adam 终点 = ({p_adam[0]:.2f}, {p_adam[1]:.2f})  到最优点距离 {np.linalg.norm(p_adam):.2f}")
    print(f"        ↑ 每个参数自适应学习率 → 陡/平两维都快，更接近最优点")
    print("  → Adam = 动量(惯性) + 自适应学习率，对病态曲面/学习率不敏感 → 现代训练默认。")


# ============================================================================
# ③ CNN：conv2d 卷积 + maxpool 池化（前向，numpy）
# ============================================================================
def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """二维卷积（valid，stride=1）。对应面试"卷积怎么算"。"""
    H, W = img.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out


def maxpool2d(img: np.ndarray, size: int = 2) -> np.ndarray:
    """最大池化（size×size 窗口，stride=size）。对应面试"池化怎么算"。"""
    H, W = img.shape
    out = np.zeros((H // size, W // size))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = img[i * size:(i + 1) * size, j * size:(j + 1) * size].max()
    return out


def demo_cnn():
    print("\n" + "=" * 60)
    print(" ③ CNN：conv2d 卷积 + maxpool 池化（前向）")
    print("=" * 60)
    # 一张 6×6 灰度图：中间一个 3×3 亮块
    img = np.zeros((6, 6))
    img[1:4, 1:4] = 1.0
    print("  输入图（6×6，中间亮块）：")
    for row in img:
        print("   ", " ".join(f"{v:.0f}" for v in row))
    # 拉普拉斯边缘核：突出和邻居差异大的像素 → 边缘
    edge = np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]])
    feat = conv2d(img, edge)
    print("  卷积后（3×3 边缘核，输出特征图）：")
    for row in feat:
        print("   ", " ".join(f"{v:+.0f}" for v in row))
    print(f"  池化后（2×2 maxpool）形状 {feat.shape} → {maxpool2d(feat).shape}：")
    print("   ", maxpool2d(feat))
    print("  → 卷积=局部加权求和提特征（这里是边缘）；池化=降维+平移不变性。")


# ============================================================================
# ④ RNN：一个 RNN cell 的前向（处理序列）
# ============================================================================
def demo_rnn():
    print("\n" + "=" * 60)
    print(" ④ RNN：一个 cell 的前向（序列 → 隐状态）")
    print("=" * 60)
    # h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
    din, dhid = 2, 4
    rng = np.random.default_rng(0)
    Wx = rng.normal(0, 0.5, (dhid, din))
    Wh = rng.normal(0, 0.5, (dhid, dhid))
    b = np.zeros(dhid)
    h = np.zeros(dhid)
    # 一个长度 4 的序列（每个时刻 2 维输入）
    seq = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
           np.array([1.0, 1.0]), np.array([0.0, 0.0])]
    print("  公式：h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)")
    for t, xt in enumerate(seq):
        h = np.tanh(Wx @ xt + Wh @ h + b)
        print(f"  t={t}  x={xt.tolist()}  →  h_{t}={h.round(2).tolist()}")
    print("  → RNN 把历史压缩进隐状态 h，逐时刻更新 → 能处理变长序列（Transformer 前的标配）。")
    print("    LSTM 在此基础上加门控(遗忘/输入/输出)缓解长序列梯度消失——结构更复杂，原理同源。")


def main():
    print("=" * 60)
    print(" 深度学习核心：激活函数 / 优化器 / CNN / RNN（numpy 从零）")
    print("=" * 60)
    demo_activations()
    demo_optimizers()
    demo_cnn()
    demo_rnn()
    print("\n" + "=" * 60)
    print(" 跑通 ✓  → 激活(ReLU/GELU)/优化(Adam)/CNN(卷积池化)/RNN(cell 前向)")
    print("=" * 60)


if __name__ == "__main__":
    main()
