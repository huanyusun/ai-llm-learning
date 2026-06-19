# 项目#3 设计：真实最小 SFT（PyTorch 从零）

> 日期：2026-06-19 ｜ 状态：已批准 ｜ 对应：`06-微调与部署/` + `knowledge/微调与部署/系统知识.md §1.4`
> 原则：项目优先、在项目中学清楚；spec 从简。

## 1. 目标
补模块最大缺口：**真正训一次**。把"训练三阶段的 SFT"从原理（01-04 只有 LoRA 原理、量化、部署、推理优化）变成动手——亲眼看到训练前乱码 → 训练后正确。

## 2. 交付物
- `06-微调与部署/05-最小SFT训练.py`：PyTorch 从零微型字符级 GPT（2 层、d=64）+ SFT 训练循环。
- 核心知识点：①`(instruction, output)` 数据格式 ②**loss mask 只对 answer 段算 loss**（prompt target=-100）③训练循环 forward→masked-CE→backward→Adam ④自回归贪心生成。
- 数据：大写/反转/求长度 三任务，45 条；holdout 词测泛化。
- LoRA 作为文档化扩展（挂 nn.Linear 即 LoRA-SFT，原理见 01）。
- 依赖：手动 `uv pip install torch`；延迟 import + 缺依赖友好提示；不自动安装。

## 3. 验收（已端到端实测）
- [x] `uv run "06-微调与部署/05-最小SFT训练.py"` 跑通（CPU ~1 分钟），输出干净（已静音 torch2.2/numpy2 ABI 警告）。
- [x] loss 从 ~3.0 降到 <0.5；训练前输出乱码，训练后 `大写: cat→CAT`、`反转: abc→cba` 正确。
- [x] 泛化：`长度: data→4` 正确（规则迁移）；`大写: loop→CODE` 错（偏记忆）——诚实展示小模型 SFT 局限。
- [x] `06-微调与部署/笔记.md` 补 §1.6 实战小节。

## 4. 环境约束（已解决）
- Intel Mac：torch 最高 2.2.2（新 torch 只发 arm64 wheel），与 numpy 2.4.6 ABI 不兼容 → import 喷 C 级警告。
- 解法：`import torch` 包在 `redirect_stderr` 里静音（filterwarnings 拦不住 C 级输出）；脚本只用纯 torch，功能不受影响。

## 5. 范围外（YAGNI）
- 不做真正 RLHF/DPO 训练（只做 SFT；对齐见 §1.5）。
- 不接 HF Trainer/PEFT（从零手写循环更"学清楚"）；LoRA 只作文档化扩展。
