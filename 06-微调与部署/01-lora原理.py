"""
微调与部署 第1课：LoRA 低秩适应原理（从零手写，纯 numpy）
============================================================
参考论文：
  LoRA: Low-Rank Adaptation of Large Language Models
  Hu et al., 2021, arXiv:2106.09685
  QLoRA: Efficient Finetuning of Quantized LLMs
  Dettmers et al., 2023, arXiv:2305.14314

核心一句话：把「权重增量 ΔW」假设成【低秩矩阵】，
  W = W0 + ΔW = W0 + B·A       （B、A 是瘦长矩阵，秩 r 很小）
于是「要训练的参数」从 d×k 暴跌到 r×(d+k)，而推理时还能把 B·A
合并回 W0，【零额外延迟】。

本文件做什么（全部 numpy 可跑）：
  【1】直观对比：全参微调 vs LoRA 的「可训练参数量」
  【2】手写 LoRA 前向：h = W0·x + (α/r)·B·A·x
  【3】手写 LoRA 反向：用链式法则求 ∂L/∂A、∂L/∂B（W0 冻结不更新）
  【4】跑一个最小训练 demo：LoRA 学一个目标线性变换
  【5】可视化：不同秩 r 下，参数占比 vs 拟合误差

运行：uv run python "06-微调与部署/01-lora原理.py"
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(0)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 【1】参数量对比：全参 vs LoRA
# ============================================================
print("=" * 60)
print("【1】全参微调 vs LoRA：可训练参数量对比")
print("=" * 60)

def count_full(d, k):
    """全参微调：整个 W0 ∈ R^{d×k} 都要训练。"""
    return d * k

def count_lora(d, k, r):
    """LoRA：只训练 A∈R^{r×k} 和 B∈R^{d×r}，W0 冻结。"""
    return r * k + d * r   # = r*(d+k)

# 以 Qwen2.5-7B 里一个典型大线性层为例（d≈4096, k≈4096）
d, k = 4096, 4096
print(f"  设一个线性层权重 W0 ∈ R^({d}×{k})，共 {d*k:,} 个参数")
print()
print(f"  {'方法':<14}{'可训练参数':>16}{'占比':>12}")
print(f"  {'-'*42}")
print(f"  {'全参微调':<14}{count_full(d,k):>16,}{100:>11.2f}%")
for r in [1, 2, 4, 8, 16, 64]:
    n = count_lora(d, k, r)
    print(f"  {'LoRA r='+str(r):<14}{n:>16,}{n/count_full(d,k)*100:>11.3f}%")

# 论文原话对应：GPT-3 175B 上 LoRA 可减少 ~10000× 可训练参数
print()
print("  论文硬指标（arXiv:2106.09685 摘要）：")
print("    「LoRA can reduce the number of trainable parameters")
print("     by 10,000 times and GPU memory by 3 times.」")
# 用一个超大层算一下倍数
ratio = count_full(12288, 12288) / count_lora(12288, 12288, r=2)
print(f"    d=k=12288, r=2 时，全参/LoRA 可训练参数比 ≈ {ratio:,.0f}×")


# ============================================================
# 【2】手写 LoRA 前向：h = W0·x + (α/r)·B·A·x
# ============================================================
print()
print("=" * 60)
print("【2】LoRA 前向：W = W0 + (α/r)·B·A")
print("=" * 60)

class LoRALinear:
    """
    最小 LoRA 线性层（纯 numpy）。
    W0：冻结的预训练权重，d×k（注意这里是 y = W0·x，x 是 k 维）
    A：r×k，高斯随机初始化
    B：d×r，【零初始化】（关键！保证训练初始 ΔW=BA=0，不破坏 W0）
    alpha：缩放因子的分子，等效缩放 = alpha/r
    """
    def __init__(self, k, d, r, alpha=16.0, seed=0):
        g = np.random.default_rng(seed)
        self.k, self.d, self.r = k, d, r
        self.alpha = alpha
        self.scaling = alpha / r                 # 论文里的 α/r
        # W0 视为「预训练权重」，这里随便给一个（真实场景从基座加载）
        self.W0 = g.standard_normal((d, k)) * 0.05
        # LoRA 初始化：A 高斯，B 零  ← arXiv:2106.09685 §4.1
        self.A = g.standard_normal((r, k)) * 0.01
        self.B = np.zeros((d, r))
        # 训练时只更新 A、B；W0 冻结（requires_grad=False）

    def forward(self, x):
        """x: (k,) 或 (batch, k) -> (d,) 或 (batch, d)。"""
        # 注意 ΔW = B·A 形状 (d,k)；这里不显式组装 ΔW，而是先 A·x 再 B·(...)，更省算
        self.x = x                              # 缓存给反向用
        self.Ax = self.A @ x.T if x.ndim > 1 else self.A @ x   # (r,) 或 (r,batch)
        return self.W0 @ x.T + self.scaling * (self.B @ self.Ax) if x.ndim > 1 \
               else self.W0 @ x + self.scaling * (self.B @ self.Ax)

    def backward(self, grad_out, lr=1e-2):
        """
        grad_out: ∂L/∂y，形状同 forward 输出 (d,) 或 (d,batch)
        用链式法则求 ∂L/∂A、∂L/∂B（W0 不更新）。
        前向： y = W0·x + s·B·(A·x)，s = α/r
        反向（对 batch 求和）：
          ∂L/∂B = s · grad_out · (A·x)ᵀ               形状 (d,r)
          ∂L/∂A = s · Bᵀ · grad_out · xᵀ              形状 (r,k)
        """
        s = self.scaling
        if grad_out.ndim == 1:
            grad_out_col = grad_out.reshape(-1, 1)
            x_col = self.x.reshape(1, -1)
            Ax_col = self.Ax.reshape(-1, 1)
        else:
            grad_out_col = grad_out         # (d,batch)
            x_col = self.x.T                # (k,batch)
            Ax_col = self.Ax                # (r,batch)
        grad_B = s * (grad_out_col @ Ax_col.T)          # (d,r)
        grad_A = s * (self.B.T @ grad_out_col @ x_col.T)  # (r,k)
        # SGD 更新（真实训练用 Adam，这里演示 SGD 即可）
        self.A -= lr * grad_A
        self.B -= lr * grad_B


# ============================================================
# 【3】最小训练 demo：让 LoRA 学会一个目标线性变换
# ============================================================
print()
print("=" * 60)
print("【3】训练 demo：用 LoRA 去拟合一个目标权重 W*")
print("=" * 60)

k, d, r = 64, 64, 4                          # 小尺寸便于肉眼观察
# 经验法则 α≈2r → 缩放 α/r≈2 恒定，保证不同 r 下学习率口径一致
alpha = 2 * r
model = LoRALinear(k, d, r, alpha=alpha)

# 目标：一个与 W0「不同」的线性变换 W*。LoRA 要用 ΔW=BA 把 W0 修正成 W*。
g = np.random.default_rng(123)
W_target = g.standard_normal((d, k)) * 0.05
# 真实的 ΔW = W_target - W0，看 LoRA 能否用低秩近似它
delta_true = W_target - model.W0

# 训练数据：随机输入 x，目标是 W_target·x
N = 256
X = g.standard_normal((N, k))
Y = (W_target @ X.T).T                       # (N, d)

losses = []
for step in range(800):
    # 全 batch 前向
    pred = model.forward(X)                  # (d,N)
    pred = pred.T                            # (N,d)
    grad = 2 * (pred - Y) / N                # ∂L/∂y，(N,d)
    model.backward(grad.T, lr=3e-2)          # 传 (d,N)
    if step % 100 == 0 or step == 799:
        l = np.mean((pred - Y) ** 2)
        losses.append((step, l))
        print(f"  step {step:3d}  MSE = {l:.6f}")

# 验证 ΔW ≈ B·A
delta_lora = model.B @ model.A
err = np.linalg.norm(delta_lora - delta_true) / (np.linalg.norm(delta_true) + 1e-9)
print(f"  训练后 ||B·A − ΔW_true|| / ||ΔW_true|| = {err:.3f}")
print(f"  （ΔW_true 的有效秩较高时，低秩 r={r} 只能近似；增大 r 会更准）")


# ============================================================
# 【4】不同秩 r 下的「参数占比 vs 拟合误差」
# ============================================================
print()
print("=" * 60)
print("【4】扫秩 r：参数占比 vs 拟合误差（量化「低秩够用」）")
print("=" * 60)

def train_one(r, steps=800):
    # α=2r → 缩放 α/r=2 恒定，使不同 r 在同一学习率下公平比较
    m = LoRALinear(k, d, r, alpha=2*r, seed=0)
    for _ in range(steps):
        pred = m.forward(X).T
        grad = 2 * (pred - Y) / N
        m.backward(grad.T, lr=3e-2)
    final = np.mean((pred - Y) ** 2)
    params = count_lora(d, k, r)
    full = count_full(d, k)
    return final, params / full * 100

rs = [1, 2, 4, 8, 16, 32, 64]
results = [train_one(r) for r in rs]
for r, (mse_v, pct) in zip(rs, results):
    tag = " ← 接近全秩(可完全恢复)" if r == 64 else (" ← 满秩可精确拟合" if r >= 32 else "")
    print(f"  r={r:3d}  可训练占比 {pct:7.3f}%   最终 MSE = {mse_v:.6f}{tag}")


# ============================================================
# 【5】可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# 左：参数占比 vs 拟合误差（双 y 轴）
ax1 = axes[0]
pcts = [count_lora(d, k, r) / count_full(d, k) * 100 for r in rs]
mses = [mse_v for (_, mse_v) in [train_one(r, 200) for r in rs]]
ax1.semilogy(rs, pcts, 'o-', color='#1f77b4', label='可训练参数占比 (%)')
ax1.set_xlabel('LoRA 秩 r')
ax1.set_ylabel('可训练参数占比 (%)', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_title('LoRA：秩 r 越大，参数越多、拟合越准\n(但 r 很小时已能逼近全参)')

ax1b = ax1.twinx()
ax1b.plot(rs, mses, 's--', color='#d62728', label='最终 MSE')
ax1b.set_ylabel('最终 MSE', color='#d62728')
ax1b.tick_params(axis='y', labelcolor='#d62728')

# 右：训练 loss 曲线（来自 demo 的 losses）
ax2 = axes[1]
steps_plot, losses_plot = zip(*losses)
ax2.plot(steps_plot, losses_plot, 'o-', color='#2ca02c')
ax2.set_xlabel('训练 step')
ax2.set_ylabel('MSE')
ax2.set_title(f'训练 demo loss（r={r}，α={alpha}）\n前向 W0·x + (α/r)·B·A·x + SGD 反向')
ax2.set_yscale('log')
ax2.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, "01-lora原理.png")
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"\n【5】图已保存：{out}")


# ============================================================
# 要点小结
# ============================================================
print()
print("=" * 60)
print("✅ LoRA 要点（连回面试）")
print("=" * 60)
print("  • 公式：W = W0 + B·A，B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k)")
print("  • 可训练参数：d·k → r·(d+k)，r=8 时通常 <1%")
print("  • 初始化：A 随机高斯、B=零 → 训练初始 ΔW=0，不破坏预训练")
print("  • 缩放 α/r：固定 α≈2r 后基本不用再调学习率")
print("  • 推理零延迟：部署时合并 W=W0+B·A 当普通层用")
print("  • 多任务：一份冻结基座 + N 个 MB 级 LoRA 适配器动态切换")
print("  • QLoRA：在 4-bit(NF4) 量化基座上做 LoRA，单 48GB 卡微调 65B")
print("  • 经验：覆盖全部线性层(q/k/v/o/gate/up/down) 才能追平 16-bit 全参")
print("=" * 60)
