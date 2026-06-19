"""05 - 最小 SFT 训练（PyTorch，从零，CPU 可跑）—— 把"训练"从原理变成动手

本模块之前的 01~04 讲了 LoRA 原理、量化、部署、推理优化，但【从没真正训过一次】。
本文件补上这个缺口：从零写一个最小的 SFT（监督微调）训练循环，亲眼看到
【训练前输出乱码 → 训练后输出正确】，把"训练三阶段里的 SFT"真正跑懂。

【SFT 的核心知识点（本文件要讲清楚的）】
    1. 数据格式：(instruction, output) 对。模型学的是"给定指令，生成答案"。
    2. ★ loss mask（最关键、最易错）：
       把 prompt+answer 拼成一串喂进去做"预测下一个 token"，但【只在答案那段算 loss】，
       prompt 部分的 target 置成 -100（交叉熵自动忽略）。
       —— 因为我们要教的是"怎么回答"，不是"怎么复述问题"。
    3. 训练循环：前向 → 带掩码的交叉熵 → 反向 → 优化器 step（Adam）。
    4. 自回归生成：贪心解码，逐 token 取 argmax。

【为什么用 PyTorch 而不是接着用 numpy？】
    训练循环的灵魂是【自动求导 + 优化器】。numpy 版（06/01）手写了 LoRA 的前向/反向，
    但要在 numpy 里跑一个完整 transformer 训练循环，等于从零造 autograd——
    反而把"训练循环本身"这件要学的事埋进一堆求导代码里。用 PyTorch 才能让训练循环
    【清晰可见】（forward → loss → backward → step）。这正是项目优先原则：学清楚 SFT，
    就用真正训得起来的工具。

【对应 knowledge/微调与部署】§1.4 SFT / loss mask；LoRA 部分见 01-lora原理.py。
【依赖】需手动装（不自动安装）：uv pip install torch
   （Intel Mac 上 torch 最高只能装 2.2.2，它与 numpy 2.x 有 ABI 警告；本脚本只用纯 torch、
    已在 import 时静音该警告，不影响运行。Apple Silicon 上可装更新的 torch，无此问题。）
【运行】uv run "06-微调与部署/05-最小SFT训练.py"
"""

from __future__ import annotations

import random


# ============================================================================
# 一、SFT 数据集：(instruction, output) 对。用 3 个确定性小任务，微型模型能真学会。
# ============================================================================
WORDS = ["cat", "dog", "ai", "code", "hello", "world", "go", "py", "abc", "xyz",
         "sun", "run", "fun", "bit", "log"]


def make_dataset():
    """生成 (instruction, output) 对。任务：转大写 / 反转 / 求长度。"""
    data = []
    for w in WORDS:
        data.append((f"大写: {w}", w.upper()))   # 大写: cat -> CAT
        data.append((f"反转: {w}", w[::-1]))      # 反转: cat -> tac
        data.append((f"长度: {w}", str(len(w))))  # 长度: cat -> 3
    return data


# held-out 词（训练集里没有），用来测"是学到了规则还是死记"
HOLDOUT = ["loop", "data"]


# ============================================================================
# 二、字符级 tokenizer（vocab 直接从语料字符构造）
# ============================================================================
class CharTok:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ============================================================================
# 三、微型 GPT（2 层、d=64）—— PyTorch，带因果掩码。架构对应 01-Transformer。
#    这里它只是"被训练的对象"，重点不在架构而在【训练循环 + loss mask】。
# ============================================================================
def build_model(vocab_size: int, d: int = 64, n_heads: int = 2, n_layers: int = 2, max_len: int = 64):
    import torch
    import torch.nn as nn

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d)
            self.pos = nn.Embedding(max_len, d)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=n_heads, dim_feedforward=d * 4,
                batch_first=True, norm_first=True, activation="relu",
            )
            self.blocks = nn.TransformerEncoder(layer, n_layers)
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab_size)

        def forward(self, idx):  # idx: [B, T]
            _, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T))
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # 上三角=禁止看未来
            x = self.blocks(x, mask=causal)
            return self.head(self.ln(x))  # [B, T, V]

    return TinyGPT()


# ============================================================================
# 四、★ SFT 的灵魂：带 loss mask 的交叉熵
#    prompt="Q:{q}\nA:"  answer="{a}\n"
#    模型在位置 i 预测 full[i+1]；【只有当 full[i+1] 属于 answer 时才算 loss】。
# ============================================================================
def sft_loss(model, tok, prompt: str, answer: str):
    import torch
    import torch.nn.functional as F

    full = torch.tensor(tok.encode(prompt) + tok.encode(answer))  # [L]
    L = full.shape[0]
    p_len = len(tok.encode(prompt))

    logits = model(full[:-1].unsqueeze(0))  # [1, L-1, V]：位置 i 预测 full[i+1]
    targets = full[1:].clone()              # [L-1]

    # ★ loss mask：把 prompt 段的 target 置 -100（交叉熵忽略），只保留 answer 段
    labels = torch.full_like(targets, -100)
    keep = torch.arange(p_len - 1, L - 1)   # 这些位置的 target 才是 answer token
    labels[keep] = targets[keep]

    loss = F.cross_entropy(logits[0], labels)  # -100 的位置自动不计入
    return loss


# ============================================================================
# 五、贪心生成（自回归，逐 token argmax）。调用方用 torch.no_grad() 包裹即可。
# ============================================================================
def generate(model, tok, prompt: str, max_new: int = 16):
    import torch

    ids = torch.tensor(tok.encode(prompt))
    p_len = ids.shape[0]
    for _ in range(max_new):
        logits = model(ids.unsqueeze(0))[0, -1]  # 只取最后一个位置
        nxt = int(logits.argmax())
        ids = torch.cat([ids, torch.tensor([nxt])])
        if tok.itos[nxt] == "\n":  # 生成到换行就停（答案结束）
            break
    return tok.decode(ids[p_len:].tolist())


# ============================================================================
# 六、main：训练前 → 训练 → 训练后，把"学到了"跑给人看
# ============================================================================
def _import_torch_quietly():
    """导入 torch，静默 torch2.2/numpy2 的 ABI 噪声（详见下方注释）。"""
    import contextlib
    import io
    # Intel Mac 上 torch 最高只能装 2.2.2（新 torch 只发 arm64 wheel），它编译于
    # numpy 1.x，与本仓库 numpy 2.4.6 ABI 不兼容，import 时往 stderr 喷一大段
    # "compiled using NumPy 1.x..."（C 级写入，filterwarnings 拦不住）。
    # 本脚本只用纯 torch（无 torch<->numpy 互转），功能完全正常；这里只静音该噪声。
    # 若装了兼容版 torch（>=2.3 / arm64），此重定向也无害。
    with contextlib.redirect_stderr(io.StringIO()):
        import torch
    return torch


def main():
    import warnings

    warnings.filterwarnings("ignore")  # 静默 enable_nested_tensor 等 Python 级警告
    try:
        torch = _import_torch_quietly()
    except ImportError:
        import sys
        print("缺少 torch。请先手动安装（不自动装）：uv pip install torch", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(42)
    random.seed(42)
    device = "cpu"  # 微型模型 CPU 足够；有 GPU 可改 "cuda"/"mps"

    data = make_dataset()
    # tokenizer 的语料：所有 prompt + answer + holdout 测试用到的字符
    corpus = "".join(q + a for q, a in data) + "Q:\nA: " + " ".join(HOLDOUT) + "".join(w.upper() for w in HOLDOUT)
    tok = CharTok(corpus)
    print(f"[准备] 训练样本 {len(data)} 条 | 词表大小 {tok.vocab_size} | device={device}")

    model = build_model(tok.vocab_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    test_prompts = ["大写: cat", "反转: abc", f"大写: {HOLDOUT[0]}", f"长度: {HOLDOUT[1]}"]

    def show(tag: str):
        model.eval()
        print(f"\n--- {tag} ---")
        for q in test_prompts:
            with torch.no_grad():
                out = generate(model, tok, f"Q:{q}\nA:")
            print(f"  Q:{q}\n   A:{out!r}")

    # ① 训练前：未训练的模型输出是乱码
    show("① 训练前（随机权重）")

    # ② 训练循环
    print("\n[训练] 开始 SFT ……")
    model.train()
    steps = 600
    for step in range(1, steps + 1):
        q, a = random.choice(data)
        prompt, answer = f"Q:{q}\nA:", f"{a}\n"
        optim.zero_grad()
        loss = sft_loss(model, tok, prompt, answer)
        loss.backward()
        optim.step()
        if step % 100 == 0:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")

    # ③ 训练后：模型应能正确回答（训练词必对；holdout 词看是否学到规则）
    show("③ 训练后（SFT 之后）")

    print("\n[结论]")
    print("  - 训练词（cat/abc）：SFT 后正确 → 模型学会了'照指令回答'。")
    print("  - holdout 词（loop/data）：若也对，说明学到了【规则】而非死记；")
    print("    若不对，是数据/模型太小→偏记忆，这也是真实 SFT 的常态（需更多数据/更大模型）。")
    print("  - ★ 核心：本文件讲清楚了 SFT 的 loss mask——只对 answer 段算 loss。")
    print("\n[LoRA 扩展] 把 01-lora原理.py 里的 LoRALinear 挂到本模型每个 nn.Linear 上，")
    print("  冻结基座只训 LoRA，就是【LoRA-SFT】（PEFT 的最小形态）。本文件先保证把训练循环跑通。）")


if __name__ == "__main__":
    main()
