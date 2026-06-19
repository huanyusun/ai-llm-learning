"""
07 · QLoRA 与 Flash Attention 原理说明
==============================================
面试考点：QLoRA 和 LoRA 的区别、Flash Attention 为什么快
运行：uv run python "06-微调与部署/07-qlora与flash_attention.py"
"""

import numpy as np

print("=" * 60)
print("QLoRA 与 Flash Attention 原理说明")
print("=" * 60)

# ── 1. QLoRA vs LoRA ──────────────────────────────────────
print("""
╔═══════════════════════════════════════════════════════════╗
║                    QLoRA vs LoRA                          ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  LoRA:                                                    ║
║    原始权重 W (FP16) + 低秩矩阵 A·B (FP16)                 ║
║    冻结 W，只训 A 和 B                                      ║
║    显存：W 占大头（FP16 = 2 bytes/param）                    ║
║                                                           ║
║  QLoRA:                                                   ║
║    原始权重 W (4-bit 量化) + 低秩矩阵 A·B (FP16)            ║
║    冻结量化后的 W，只训 A 和 B                               ║
║    显存：W 大幅缩小（4-bit = 0.5 bytes/param）              ║
║                                                           ║
║  核心区别：QLoRA 把基座模型量化到 4-bit 再做 LoRA            ║
╚═══════════════════════════════════════════════════════════╝
""")

# 显存对比计算
print("--- 显存对比（以 LLaMA-7B 为例）---\n")

params_b = 7  # 70亿参数
lora_rank = 16
lora_params_ratio = 0.01  # LoRA 参数约占 1%

full_fp16 = params_b * 2  # FP16: 2 bytes/param
lora_fp16 = params_b * 2 + params_b * lora_params_ratio * 2  # 基座FP16 + LoRA FP16
qlora_4bit = params_b * 0.5 + params_b * lora_params_ratio * 2  # 基座4bit + LoRA FP16

print(f"  全参数微调 (FP16):  {full_fp16:.1f} GB（仅权重，不含优化器/梯度/激活）")
print(f"  LoRA (FP16 基座):   {lora_fp16:.1f} GB")
print(f"  QLoRA (4-bit 基座): {qlora_4bit:.1f} GB")
print(f"\n  → QLoRA 显存约为 LoRA 的 {qlora_4bit/lora_fp16:.0%}")
print(f"  → 7B 模型 QLoRA 可在单张 24GB 消费级 GPU 上微调！")

# QLoRA 三大技术
print(f"\n--- QLoRA 三大关键技术 ---")
print("""
1. NF4 量化（4-bit NormalFloat）
   - 专为正态分布权重设计的量化格式
   - 比普通 INT4 更精确（信息论最优）

2. 双重量化（Double Quantization）
   - 对量化常数本身再做一次量化
   - 进一步节省 ~0.37 bit/param

3. 分页优化器（Paged Optimizers）
   - 用 CPU 内存作为 GPU 显存的"交换区"
   - 处理显存峰值（长序列时的梯度累积）
""")

# ── 2. Flash Attention ──────────────────────────────────────
print("=" * 60)
print("Flash Attention 原理")
print("=" * 60)

print("""
问题：标准 Self-Attention 的瓶颈

  标准实现：
    1. 计算 S = Q @ K^T          → O(n²d) 计算，O(n²) 显存
    2. 计算 P = softmax(S)       → O(n²) 显存
    3. 计算 O = P @ V            → O(n²d) 计算

  瓶颈不是计算，而是【显存带宽】！
  - GPU 计算很快（FLOPS 高），但 HBM ↔ SRAM 的数据搬运很慢
  - n² 的注意力矩阵要在 HBM 中读写多次 → IO 瓶颈

Flash Attention 的解决方案：【分块计算 + 在线 softmax】

  核心思想：
    1. 把 Q, K, V 分成小块（tile），每块能放进 SRAM
    2. 在 SRAM 中完成注意力计算（不写回 HBM）
    3. 用在线 softmax 算法，边算边更新，无需存储完整 n² 矩阵

  效果：
    - 显存从 O(n²) 降到 O(n)
    - 速度提升 2-4x（减少 HBM 读写）
    - 数学上完全等价（不是近似！）
""")

# 模拟显存对比
print("--- 显存对比 ---\n")
seq_lengths = [512, 2048, 8192, 32768]
d_model = 128  # head dim

print(f"{'序列长度':>8} {'标准Attention':>15} {'Flash Attention':>16} {'节省':>8}")
print("-" * 52)
for n in seq_lengths:
    standard_mem = n * n * 4 / (1024**2)  # FP32, MB
    flash_mem = n * d_model * 4 / (1024**2)  # O(n*d), MB
    saving = 1 - flash_mem / standard_mem
    print(f"{n:>8} {standard_mem:>12.1f} MB {flash_mem:>13.3f} MB {saving:>7.1%}")

print("""
Flash Attention 版本演进：
  - Flash Attention 1 (2022): 分块 + 在线 softmax
  - Flash Attention 2 (2023): 更好的并行化，速度再提升 2x
  - Flash Attention 3 (2024): 利用 Hopper GPU 特性（FP8、异步）

面试关键点：
1. Flash Attention 不是近似算法，结果和标准 Attention 完全一致
2. 核心优化是减少 HBM 读写（IO-aware），不是减少计算量
3. 显存从 O(n²) 降到 O(n)，使长序列训练成为可能
4. PyTorch 2.0+ 已内置 `torch.nn.functional.scaled_dot_product_attention`
5. 几乎所有现代 LLM 训练都使用 Flash Attention
""")

# ── 3. 综合对比表 ──────────────────────────────────────────
print("=" * 60)
print("微调与推理优化技术总览")
print("=" * 60)
print("""
| 技术             | 阶段   | 核心思想                    | 效果                |
|-----------------|--------|---------------------------|---------------------|
| LoRA            | 微调   | 低秩分解，只训小矩阵          | 参数量降 99%+        |
| QLoRA           | 微调   | 4-bit 量化基座 + LoRA       | 显存再降 ~75%        |
| Flash Attention | 训练/推理 | 分块计算，减少显存读写       | 速度 2-4x，显存 O(n) |
| KV Cache        | 推理   | 缓存历史 K/V 避免重算        | 推理延迟大幅降低      |
| 量化 (GPTQ/AWQ) | 推理   | 权重量化到 4/8-bit          | 显存降 2-4x          |
| vLLM            | 推理   | PagedAttention + 连续批处理  | 吞吐量提升 2-24x     |
""")
