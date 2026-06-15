"""
数学基础 第12课：反向传播 Backpropagation（从零开始）
=========================================================================
参考：3Blue1Brown《微积分的本质》《神经网络究竟在做什么》、
     《动手学深度学习》https://zh.d2l.ai （3.6-3.7 反向传播）

上一课的梯度下降只对 2 个变量。真正的神经网络有几千几亿个参数，
怎么高效地算出『损失对每个参数的梯度』？答案就是**反向传播**。

核心思想（3 句话）：
1. 把网络拆成一张「计算图」：输入 → 乘 → 加 → 损失，一步步算（前向）。
2. 链式法则：复合函数的导数 = 一层一层导数相乘。
3. 反向传播 = 从损失开始『倒着』用链式法则，一次算出所有参数的梯度。

本课：手写一个极简网络 y = w·x，loss = (y - y_true)²，
       前向 → 反向 → 更新 w，用 numpy 实现，并画出 loss 下降。
运行：uv run python "00-数学基础/12-反向传播.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】计算图：把一次预测拆成清晰的步骤
# ============================================================
# 任务：给你输入 x，想让模型 y = w·x 去逼近真实值 y_true。
# 计算图（前向，从左到右）：
#
#     x ──┐
#         ├─► u = w·x ──► y = u ──► e = y - y_true ──► L = e²
#     w ──┘
#
# 每个箭头都很简单，但串起来就能算出损失 L。

x = 2.0        # 输入
w = 0.5        # 权重（这是我们要学的参数，初始随便给一个）
y_true = 4.0   # 真实标签（我们希望 w·x ≈ 4）

print("【1】计算图（前向）")
print(f"    输入 x = {x}, 初始权重 w = {w}, 真实值 y_true = {y_true}")
print("    计算图: x,w ─► u=w·x ─► y ─► e=y-y_true ─► L=e²\n")

# ============================================================
# 【2】前向传播：从左到右算出损失
# ============================================================
def forward(x, w, y_true):
    u = w * x
    y = u
    e = y - y_true
    L = e ** 2
    return dict(u=u, y=y, e=e, L=L)

fwd = forward(x, w, y_true)
print("【2】前向传播结果")
for k, v in fwd.items():
    print(f"    {k} = {v}")
print(f"    → 损失 L = {fwd['L']:.3f}（还没学好，离 4 还远）\n")

# ============================================================
# 【3】反向传播：从损失倒着算每个参数的梯度（链式法则）
# ============================================================
# 目标：求 dL/dw（损失对权重的梯度）。用链式法则一层层拆：
#
#     dL/dw = dL/de · de/dy · dy/du · du/dw
#             ─────   ─────   ─────   ─────
#             2e       1       1       x
#
# 直观理解（3Blue1Brown 的讲法）：
#  - 最后一步 L=e² 对 e 的斜率是 2e（损失怎么随误差变）；
#  - 误差 e=y-y_true 对 y 的斜率是 1；
#  - y 就等于 u，斜率 1；
#  - u=w·x 对 w 的斜率是 x（w 每变一点，u 就变 x 那么多）。
# 把这些斜率『乘』起来，就是损失对 w 的总灵敏度。

def backward(x, w, y_true):
    """反向传播：返回 L 对 w 的梯度（一个数）。"""
    # 先前向，拿到中间量
    u = w * x
    e = u - y_true
    # 反向（从 L 倒推到 w），每一步是一个局部导数：
    dL_de = 2 * e      # L = e²     → dL/de = 2e
    de_dy = 1.0        # e = y - yt → de/dy = 1
    dy_du = 1.0        # y = u      → dy/du = 1
    du_dw = x          # u = w·x    → du/dw = x
    # 链式法则：全部相乘
    dL_dw = dL_de * de_dy * dy_du * du_dw
    return dL_dw

grad_w = backward(x, w, y_true)
print("【3】反向传播（链式法则求 dL/dw）")
print(f"    dL/de = 2e = {2*fwd['e']}")
print(f"    de/dy = 1,  dy/du = 1")
print(f"    du/dw = x = {x}")
print(f"    dL/dw = 2e × 1 × 1 × x = {grad_w}")
print(f"    （负号表示 w 该往大调，才能减小损失）\n")

# ============================================================
# 【4】梯度下降更新 w，重复多轮，看 w 收敛到正确值
# ============================================================
# 真正的解：要 y = w·x = y_true，即 w = y_true/x = 4/2 = 2。
# 看反向传播 + 梯度下降能不能自己「学」到 w=2。
lr = 0.05
epochs = 60
w_train = 0.5   # 重新初始化
history = []    # 记录每轮损失
w_path = []     # 记录 w 的变化

for epoch in range(epochs):
    fwd = forward(x, w_train, y_true)
    g = backward(x, w_train, y_true)
    w_train = w_train - lr * g          # 梯度下降
    history.append(fwd['L'])
    w_path.append(w_train)

print("【4】训练 %d 轮（lr=%.2f）" % (epochs, lr))
print(f"    w 从 0.500 → {w_train:.4f}（理论正确值 w = y_true/x = {y_true/x}）")
print(f"    损失从 {history[0]:.3f} → {history[-1]:.6f}\n")

# ============================================================
# 【5】扩展：反向传播为什么是深度学习的「灵魂」
# ============================================================
print("【5】为什么反向传播这么重要？")
print("  • 真正的网络有几千~几亿个 w，不可能手算每个梯度。")
print("  • 反向传播用链式法则『一次反向遍历』就能算出所有 w 的梯度，")
print("    复杂度只比前向多一点点（约 2~3 倍）。")
print("  • PyTorch / TensorFlow 的 autograd（自动求导）就是自动做这件事。\n")

# ============================================================
# 【6】可视化：loss 下降 + w 收敛
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：损失下降
axes[0].plot(history, 'r.-', lw=2, ms=5)
axes[0].set_title("反向传播训练：损失随轮次下降", fontsize=12)
axes[0].set_xlabel("训练轮次 epoch"); axes[0].set_ylabel("损失 L = (w·x − y_true)²")
axes[0].grid(True, alpha=0.3)
axes[0].annotate('初始损失', xy=(0, history[0]), xytext=(15, history[0]*0.85),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
axes[0].annotate('接近 0，学好了', xy=(epochs-1, history[-1]),
                 xytext=(epochs*0.55, history[-1]+history[0]*0.25),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

# 右：w 收敛到正确值
axes[1].plot(w_path, 'b.-', lw=2, ms=5, label='w 的学习轨迹')
axes[1].axhline(y_true / x, color='green', ls='--', lw=2, label=f'正确 w = {y_true/x}')
axes[1].set_title("权重 w 自动收敛到正确值", fontsize=12)
axes[1].set_xlabel("训练轮次 epoch"); axes[1].set_ylabel("权重 w")
axes[1].grid(True, alpha=0.3); axes[1].legend()

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "12-反向传播.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【6】图已保存：{out}\n")

print("=" * 62)
print("✅ 第12课要点")
print("=" * 62)
print("  • 计算图：把一次预测拆成简单的加/乘步骤（前向）")
print("  • 链式法则：复合函数的导数 = 各步导数相乘")
print("  • 反向传播：从损失倒着用链式法则，一次算出所有参数的梯度")
print("  • 配合梯度下降更新参数 → 模型就『学』会了")
print()
print("🎯 AI 里的应用（深度学习训练的灵魂）：")
print("  • 神经网络的每一层、每个权重都靠反向传播得到梯度")
print("  • loss.backward() 这一行代码 = 触发反向传播（PyTorch）")
print("  • 《动手学深度学习》3.7：反向传播通过链式法则高效算梯度")
print("=" * 62)
