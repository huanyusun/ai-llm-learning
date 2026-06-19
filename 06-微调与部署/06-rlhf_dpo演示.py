"""
06 · RLHF / DPO 流程演示
==============================================
面试考点：RLHF 三步流程、DPO 为什么更简单、KL 约束的作用
运行：uv run python "06-微调与部署/06-rlhf_dpo演示.py"
"""

import numpy as np

print("=" * 60)
print("RLHF / DPO 流程演示")
print("=" * 60)

# ── 1. RLHF 三步流程 ──────────────────────────────────────
print("""
┌─────────────────────────────────────────────────────────────┐
│                    RLHF 三步流程                              │
│                                                             │
│  Step 1: SFT（监督微调）                                      │
│    预训练模型 + 指令数据 → SFT 模型（π_SFT）                    │
│                                                             │
│  Step 2: 训练奖励模型（RM）                                    │
│    同一 prompt → 模型生成多个回答 → 人类排序 → 训练 RM          │
│    RM 输入: (prompt, response) → 输出: scalar reward           │
│                                                             │
│  Step 3: PPO 强化学习优化                                      │
│    用 RM 的分数作为奖励，PPO 优化策略                            │
│    目标: max E[R(x,y)] - β·KL(π || π_SFT)                    │
│    KL 约束防止模型偏离 SFT 太远（reward hacking）               │
└─────────────────────────────────────────────────────────────┘
""")

# ── 2. 模拟奖励模型训练 ──────────────────────────────────────
print("=" * 60)
print("Step 2 模拟：奖励模型训练（Bradley-Terry 模型）")
print("=" * 60)

# 模拟偏好数据：人类标注 (chosen, rejected) 对
preference_data = [
    {"prompt": "什么是机器学习？",
     "chosen": "机器学习是人工智能的一个分支，通过数据和算法让计算机自动学习和改进。",
     "rejected": "机器学习就是让机器学习。"},
    {"prompt": "Python 有什么优点？",
     "chosen": "Python 语法简洁、生态丰富、适合数据科学和 AI 开发，有大量第三方库支持。",
     "rejected": "Python 是一种编程语言，可以写代码。"},
    {"prompt": "如何学习深度学习？",
     "chosen": "建议从线性代数和微积分基础开始，然后学习 PyTorch，通过实战项目巩固理论。",
     "rejected": "去网上搜一下就行了。"},
]

# 模拟 RM 的分数（真实场景中由模型计算）
np.random.seed(42)

print(f"\n偏好数据 ({len(preference_data)} 对):")
for i, pair in enumerate(preference_data):
    chosen_score = np.random.uniform(0.6, 0.9)
    rejected_score = np.random.uniform(0.1, 0.4)
    print(f"\n  Prompt: \"{pair['prompt']}\"")
    print(f"  ✓ Chosen  (score={chosen_score:.2f}): \"{pair['chosen'][:40]}...\"")
    print(f"  ✗ Rejected(score={rejected_score:.2f}): \"{pair['rejected'][:40]}...\"")

# Bradley-Terry loss
print(f"\nBradley-Terry Loss:")
print(f"  L = -log(σ(r_chosen - r_rejected))")
print(f"  目标：让 chosen 的分数高于 rejected")

# ── 3. DPO：直接偏好优化 ──────────────────────────────────
print("\n" + "=" * 60)
print("DPO（Direct Preference Optimization）")
print("=" * 60)
print("""
DPO 的核心洞察（Rafailov et al., 2023）：

  RLHF 需要三步：SFT → RM → PPO
  DPO 只需两步：SFT → DPO

  数学推导：在 KL 约束下，最优策略 π* 和奖励 r* 有闭式关系：
    r*(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

  → 把这个关系代入 Bradley-Terry loss，消掉显式 RM：
    L_DPO = -log σ(β · [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])

  其中 y_w = chosen, y_l = rejected, π_ref = SFT 模型
""")

# 模拟 DPO loss 计算
print("--- 模拟 DPO Loss 计算 ---\n")

beta = 0.1  # KL 约束强度

for i, pair in enumerate(preference_data):
    # 模拟 log 概率（真实场景中由模型前向传播计算）
    log_pi_chosen = np.random.uniform(-2, -0.5)
    log_pi_rejected = np.random.uniform(-3, -1)
    log_ref_chosen = np.random.uniform(-2.5, -1)
    log_ref_rejected = np.random.uniform(-2.5, -1)

    # DPO loss
    reward_diff = beta * ((log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected))
    loss = -np.log(1 / (1 + np.exp(-reward_diff)))

    print(f"  样本 {i+1}: loss = {loss:.4f}  (reward_diff = {reward_diff:.4f})")

# ── 4. RLHF vs DPO 对比 ──────────────────────────────────
print("\n" + "=" * 60)
print("RLHF vs DPO 对比")
print("=" * 60)
print("""
| 维度         | RLHF (PPO)                  | DPO                         |
|-------------|-----------------------------|-----------------------------|
| 步骤         | SFT → RM → PPO（三步）       | SFT → DPO（两步）            |
| 是否需要 RM   | 需要单独训练奖励模型           | 不需要（隐式奖励）             |
| 在线采样      | 需要（PPO 要在线生成样本）      | 不需要（离线偏好数据即可）      |
| 训练稳定性    | 不稳定（RL 固有问题）          | 稳定（等价于分类 loss）        |
| 计算成本      | 高（4 个模型：策略/参考/RM/critic）| 低（2 个模型：策略/参考）     |
| 效果         | 理论上限更高                   | 实践中效果相当                 |
| 代表应用      | InstructGPT, ChatGPT         | Llama 2, Zephyr              |

面试关键点：
1. DPO 的核心公式：策略即奖励（"Your LM is Secretly a Reward Model"）
2. β 控制 KL 约束强度：β 大 → 更保守（接近 SFT），β 小 → 更激进
3. DPO 不是"没有奖励"，而是用策略参数化隐式表达了奖励
4. RLHF 的 reward hacking 问题：模型学会"骗"RM 拿高分但回答质量差
""")
