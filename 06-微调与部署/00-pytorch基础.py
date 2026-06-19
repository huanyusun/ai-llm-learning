"""00 - PyTorch 基础：tensor / autograd / nn.Module / 训练循环（从零，CPU 可跑）

全景图块八「开发框架与工具」此前一个 PyTorch 项目都没有——本文件补上。它也是本模块
05-最小SFT训练.py 的【前置基础】：把 PyTorch 四件套用最小例子讲清楚，再用一个能真学会
XOR 的小 MLP 把「训练循环」跑通。

【为什么需要 PyTorch（对比 numpy）】
    06/01-lora原理.py 用 numpy 手写了前向 + 手推了反向（链式法则）。那是"学原理"。
    但真要训练，手写每个算子的梯度又累又错。PyTorch 的核心价值：
      ① tensor  ：类 numpy 的多维数组，但带 GPU 支持和【自动求导】。
      ② autograd：你只写前向，它自动建计算图、自动算梯度（.backward() 一行）。
      ③ nn.Module：把"模型"写成类，参数自动注册。
      ④ optimizer：SGD/Adam 自动用梯度更新参数。

【本文件四步】
    1. tensor 基本操作（和 numpy 对照）
    2. autograd 自动求导（对比手推）
    3. nn.Module 写一个小 MLP
    4. 训练循环：在 XOR 上训练，看 loss 下降、预测从瞎猜变正确
       XOR 是经典：线性模型学不会（非线性），必须靠隐藏层——正好演示 MLP 的价值。

【对应】knowledge/微调与部署 §1（微调要用 PyTorch）；本模块 01/05 都建立在 PyTorch 上。
【依赖】需手动装（不自动安装）：uv pip install torch
   （Intel Mac 上 torch≤2.2.2 与 numpy 2.x 有 ABI 警告；纯 torch 不受影响，已静音。）
【运行】uv run "06-微调与部署/00-pytorch基础.py"
"""

from __future__ import annotations


def _import_torch_quietly():
    """导入 torch，静默 torch2.2/numpy2 的 ABI 噪声（详见 05-最小SFT训练.py 同名函数注释）。"""
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):
        import torch
    return torch


def section_tensors(torch):
    print("\n" + "=" * 60)
    print(" ① tensor：类 numpy 的多维数组")
    print("=" * 60)
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.ones(2, 2)
    print("  a =\n", a)
    print("  a.shape =", tuple(a.shape), "| a @ b =\n", (a @ b).tolist())
    print("  → API 几乎和 numpy 一样（torch.tensor / .shape / @）；多了 GPU + autograd。")


def section_autograd(torch):
    print("\n" + "=" * 60)
    print(" ② autograd：只写前向，自动算梯度")
    print("=" * 60)
    w = torch.tensor(2.0, requires_grad=True)   # 要对 w 求导
    y = w ** 2 + 2 * w                          # y = w² + 2w
    y.backward()                                # ★ 一行：自动反向传播
    print(f"  y = w² + 2w，w=2 → dy/dw = 2w+2 = {2*2+2}")
    print(f"  autograd 算出的 w.grad = {w.grad.item()}  ← 和手推一致")
    print("  → 06/01 里 LoRA 的反向是手推的；这里 .backward() 一行搞定，PyTorch 的灵魂。")


def section_train_xor(torch):
    import torch.nn as nn

    print("\n" + "=" * 60)
    print(" ③④ nn.Module + 训练循环：小 MLP 学 XOR")
    print("=" * 60)

    # XOR 数据：线性不可分，必须靠隐藏层（非线性）
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    Y = torch.tensor([[0.], [1.], [1.], [0.]])

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 8)   # 隐藏层：2 → 8
            self.fc2 = nn.Linear(8, 1)   # 输出层：8 → 1

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))  # 返回 logits

    torch.manual_seed(0)
    model = MLP()
    loss_fn = nn.BCEWithLogitsLoss()        # 二分类（用 logits，数值稳定）
    optim = torch.optim.Adam(model.parameters(), lr=0.1)

    def predict():
        with torch.no_grad():
            probs = torch.sigmoid(model(X))
        return [round(float(p), 2) for p in probs]

    print("  训练前预测（瞎猜 ≈0.5）：", predict())

    # 标准训练循环：forward → loss → backward → step
    for step in range(1, 1501):
        logits = model(X)            # 前向
        loss = loss_fn(logits, Y)    # 算损失
        optim.zero_grad()            # 清旧梯度
        loss.backward()              # 反向（autograd）
        optim.step()                 # 更新参数
        if step % 500 == 0:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")

    print("  训练后预测（≈ 0/1 正确）：", predict())
    print("  → XOR 正确答案 [0,1,1,0]；MLP 学会了（线性模型学不会，靠隐藏层 ReLU）。")


def main():
    try:
        torch = _import_torch_quietly()
    except ImportError:
        import sys
        print("缺少 torch。请先手动安装：uv pip install torch", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" PyTorch 基础 —— tensor / autograd / nn.Module / 训练循环")
    print(f" torch {torch.__version__}")
    print("=" * 60)
    section_tensors(torch)
    section_autograd(torch)
    section_train_xor(torch)
    print("\n" + "=" * 60)
    print(" 跑通 ✓  → 这四件套是后面 01-LoRA / 05-SFT 的地基。")
    print("=" * 60)


if __name__ == "__main__":
    main()
