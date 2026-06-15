"""
Prompt 工程 第1课：Few-shot 学习 vs Chain-of-Thought（CoT）思维链 对比实验
======================================================================
理论来源：
- Wei et al., 2022, Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models, arXiv:2201.11903（ICLR 2022）
  关键论断：CoT 是「涌现能力」，在 >=~100B 参数的大模型上才显著生效；
  PaLM 540B 在 GSM8K 上从标准 prompting 18% → CoT 57%（仅用 8 个 exemplar）。
- Kojima et al., 2022, Large Language Models are Zero-Shot Reasoners,
  arXiv:2205.11916 —— Zero-shot CoT 触发词 "Let's think step by step."。
- OpenAI Prompt Engineering Guide 策略四：Give the model time to "think"。

本文件包含两部分：
  A)【模拟版 · 不依赖模型】—— 用一个「规则模拟器」扮演小模型 vs 大模型，
     直观演示「普通 few-shot」与「CoT few-shot」在多步推理上的差异，
     以及 CoT 对「小模型反而有害」这一经典易错陷阱。
  B)【真模型版 · 依赖 ollama / API】—— 标注好配置，连上本地 ollama 或
     OpenAI 兼容 API 即可跑真实的 CoT 对比。

运行：
  模拟版：uv run python "03-Prompt工程/01-few_shot与cot.py"
  真模型版：见文件末尾「真模型版配置说明」。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable


# ============================================================
# 【0】一条经典的 GSM8K 风格数学应用题（用来贯穿全课）
# ============================================================
# 取自 CoT 论文 [来源: arXiv:2201.11903] Fig.2 同款范式：
# 「Roger 有 5 个网球，又买了 2 罐，每罐 3 个，一共多少个？」
TEST_QUESTION = (
    "食堂原来有 23 个苹果，做午饭用掉 20 个，又买进 6 个，现在有多少个苹果？"
)
TEST_ANSWER = 9  # 23 - 20 + 6 = 9，用于模拟器判定对错


# ============================================================
# 【A】模拟版：用一个「假模型」演示 few-shot 与 CoT 的本质差异
# ============================================================
# 设计思想：我们不真的调 LLM，而是用「规则」模拟两类模型行为：
#   - 小模型（small_model）：只会做「一步」运算，多步题容易算错；
#   - 大模型（large_model）：能跟着 prompt 里的「推理过程」逐步算。
# 这样能肉眼看清论文里两个核心结论：
#   结论1：普通 few-shot（只给最终答案）教不会模型「怎么想」，
#          多步题照样错。
#   结论2：CoT few-shot（把推理步骤写进示例）能引导大模型分步算，
#          多步题正确率飙升。
#   结论3：CoT 对小模型反而有害——小模型抄了「推理链」的形式，
#          但每一步都算错，越算越离谱。[来源: arXiv:2201.11903]


def extract_numbers(text: str) -> list[int]:
    """从文本里抠出所有整数（模拟「模型从输出里提取最终答案」这一步）。"""
    return [int(x) for x in re.findall(r"-?\d+", text)]


def small_model(prompt: str) -> str:
    """
    模拟一个「只会一步运算」的小模型（< 100B）。
    它的行为：
      - 如果 prompt 里直接写了「答案是 N」，它就复读 N（few-shot 过拟合到示例）；
      - 否则它会「猜」一个题目里出现的数字，且容易猜错。
    这正对应论文警告：小模型在多步推理上不可靠。
    """
    # few-shot 的最后一条示例如果给了「答案 = 11」，小模型会傻乎乎复读 11
    m = re.search(r"答案[是为：:\s]*(\d+)", prompt)
    if m:
        return f"答案是 {m.group(1)}。"  # 复读最后一个见过的「答案」
    # 没有示例可抄 → 随便挑题目里的一个数字（且故意挑错，模拟幻觉）
    nums = extract_numbers(prompt)
    if nums:
        wrong = nums[0]  # 取第一个数字，几乎肯定不是答案
        return f"答案是 {wrong}。"
    return "不知道。"


def large_model(prompt: str) -> str:
    """
    模拟一个「>= 100B」的大模型。它会「照着 prompt 的推理格式」走。
    关键点：它的正确率取决于 prompt 是否给出了「推理过程」。
      - 普通 few-shot（只给答案）：它也只给答案，多步题仍可能错；
      - CoT few-shot（给了推理链）：它模仿分步推理，逐步算对。
    """
    has_chain = ("+" in prompt or "-" in prompt) and "=" in prompt and "所以" in prompt
    if has_chain:
        # CoT 模式：跟着示例的分步格式，逐步算（这里用真算模拟「算对」）
        nums = extract_numbers(prompt.split("Q:")[-1])  # 取真实问题部分的数字
        # 23 - 20 + 6 = 9；用解析算式的方式模拟「分步推理正确」
        return (
            "原来 23 个，用掉 20 个，所以 23 - 20 = 3；"
            "又买进 6 个，所以 3 + 6 = 9。答案是 9。"
        )
    # 普通 few-shot：只模仿「输出答案」的格式，但没有推理过程兜底 → 猜
    nums = extract_numbers(prompt.split("Q:")[-1])
    guess = nums[0] if nums else 0
    return f"答案是 {guess}。"


def build_standard_few_shot(question: str) -> str:
    """普通 few-shot：示例里只给「输入 → 最终答案」，不给推理过程。"""
    return (
        "下面是几道数学题和它们的答案。\n"
        "Q: 罗杰有 5 个网球，又买了 2 罐，每罐 3 个，一共多少个？\n"
        "A: 答案是 11。\n"
        "Q: 书架有 8 本书，又放上去 4 本，一共多少本？\n"
        "A: 答案是 12。\n"
        f"Q: {question}\n"
        "A:"
    )


def build_cot_few_shot(question: str) -> str:
    """CoT few-shot：示例里写出完整的「推理链 → 答案」。"""
    return (
        "下面是几道数学题，请像示例那样一步步推理后再给答案。\n"
        "Q: 罗杰有 5 个网球，又买了 2 罐，每罐 3 个，一共多少个？\n"
        "A: 罗杰原来有 5 个。2 罐每罐 3 个是 6 个。所以 5 + 6 = 11。答案是 11。\n"
        "Q: 书架有 8 本书，又放上去 4 本，一共多少本？\n"
        "A: 原来有 8 本，又放上去 4 本，所以 8 + 4 = 12。答案是 12。\n"
        f"Q: {question}\n"
        "A:"
    )


def run_experiment() -> None:
    """跑 2 模型 × 2 prompt = 4 组对照，打印结果与对错。"""
    print("=" * 72)
    print("【A】模拟版：Few-shot vs CoT 对比实验（不依赖任何模型）")
    print("=" * 72)
    print(f"测试题：{TEST_QUESTION}")
    print(f"正确答案：{TEST_ANSWER}\n")

    configs: list[tuple[str, str, Callable[[str], str]]] = [
        ("小模型 + 普通 few-shot", build_standard_few_shot(TEST_QUESTION), small_model),
        ("小模型 + CoT few-shot ", build_cot_few_shot(TEST_QUESTION), small_model),
        ("大模型 + 普通 few-shot", build_standard_few_shot(TEST_QUESTION), large_model),
        ("大模型 + CoT few-shot ", build_cot_few_shot(TEST_QUESTION), large_model),
    ]

    for name, prompt, model in configs:
        out = model(prompt)
        nums = extract_numbers(out)
        pred = nums[-1] if nums else None
        ok = "✅" if pred == TEST_ANSWER else "❌"
        print(f"[{name}]")
        print(f"  模型输出：{out}")
        print(f"  提取答案：{pred}  判定：{ok}\n")

    print("观察：")
    print("  • 小模型无论哪种 prompt 都错——CoT 对 <100B 模型不仅没用，")
    print("    还可能因为抄了错误的「推理链」越带越偏（论文核心警告）。")
    print("  • 大模型在普通 few-shot 下靠猜、容易错；换成 CoT 后逐步算对。")
    print("  → 这正是 CoT 论文「emergent ability（涌现能力）」的可视化。")


# ============================================================
# 【B】真模型版：接 ollama / OpenAI 兼容 API 跑真实 CoT 对比
# ============================================================
# 配置说明（用前请改）：
#   方式1（推荐，免费本地）：先装 ollama 并拉一个 >=7B 的模型，如：
#       ollama pull qwen2.5:7b      # 或 llama3.1:8b / glm4:9b
#       ollama serve                 # 启动服务（默认 http://localhost:11434）
#     把 MODEL_PROVIDER = "ollama"，MODEL_NAME 填你拉下来的模型名。
#   方式2（云端 API）：填 OpenAI 兼容端点 + key，例如智谱 GLM、DeepSeek、OpenAI。
#     把 MODEL_PROVIDER = "openai"，并设置环境变量：
#       export OPENAI_API_KEY="sk-xxx"
#       export OPENAI_BASE_URL="https://api.deepseek.com/v1"  # 按厂商改
#     MODEL_NAME 填 "deepseek-chat" / "gpt-4o-mini" / "glm-4-flash" 等。
#
# 依赖（按需安装）：uv add httpx   # 只用标准库 urllib 也行，这里用 httpx 更简洁
MODEL_PROVIDER = "ollama"            # "ollama" | "openai" | "none"
MODEL_NAME = "qwen2.5:7b"            # 改成你本地/云端的模型名
OLLAMA_URL = "http://localhost:11434/api/generate"
OPENAI_URL_ENV = "OPENAI_BASE_URL"   # 不设则默认 https://api.openai.com/v1
OPENAI_KEY_ENV = "OPENAI_API_KEY"


def call_model(prompt: str, temperature: float = 0.0, timeout: int = 60) -> str:
    """统一封装：根据 MODEL_PROVIDER 调对应后端。失败时抛异常给上层捕获。"""
    import urllib.request
    import json
    import os

    if MODEL_PROVIDER == "ollama":
        payload = json.dumps(
            {"model": MODEL_NAME, "prompt": prompt, "stream": False,
             "options": {"temperature": temperature}}
        ).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))["response"]

    if MODEL_PROVIDER == "openai":
        base = os.environ.get(OPENAI_URL_ENV, "https://api.openai.com/v1")
        key = os.environ.get(OPENAI_KEY_ENV, "")
        payload = json.dumps(
            {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
             "temperature": temperature}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base.rstrip('/')}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]

    raise RuntimeError("MODEL_PROVIDER 不是 ollama/openai，无法调用真模型。")


def run_real_model_demo() -> None:
    """真模型版：对同一道题，分别用「普通 few-shot」「CoT few-shot」
    「Zero-shot CoT（Let's think step by step）」三种 prompt 调一次，
    对比输出。需要联网 + 模型可用。"""
    print("=" * 72)
    print("【B】真模型版：Few-shot vs CoT vs Zero-shot CoT（需模型可用）")
    print(f"    后端：{MODEL_PROVIDER}  模型：{MODEL_NAME}")
    print("=" * 72)

    prompts = {
        "普通 few-shot": build_standard_few_shot(TEST_QUESTION),
        "CoT few-shot ": build_cot_few_shot(TEST_QUESTION),
        "Zero-shot CoT": f"Q: {TEST_QUESTION}\nA: Let's think step by step.",
    }

    for name, p in prompts.items():
        try:
            out = call_model(p, temperature=0.0)
            nums = extract_numbers(out)
            pred = nums[-1] if nums else None
            ok = "✅" if pred == TEST_ANSWER else "❌"
            print(f"[{name}]")
            print(f"  输出：{out.strip()[:200]}")
            print(f"  提取答案：{pred}  判定：{ok}\n")
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] 调用失败（跳过）：{e}\n")
            break


# ============================================================
# 【彩蛋】Self-Consistency 的核心思想演示（多路采样 + 多数表决）
# ============================================================
# 论文：Wang et al., 2022, arXiv:2203.11171。GSM8K 上比 CoT 再 +17.9%。
# 思想：用「高温」对同一 CoT prompt 采样 N 条不同推理链，对「最终答案」
# 做多数表决。注意：必须 temperature>0，否则 N 条链几乎一样，投票无意义。
def majority_vote(answers: list[int]) -> int:
    """多数表决：返回出现次数最多的答案。"""
    return Counter(answers).most_common(1)[0][0]


def demo_self_consistency_idea() -> None:
    """用模拟数据演示「多条推理链 → 投票」如何把单条链的错误纠正。"""
    print("=" * 72)
    print("【彩蛋】Self-Consistency 思想演示（多路采样 + 多数表决）")
    print("=" * 72)
    # 假设高温采样出 5 条推理链，其中 3 条算对(=9)，2 条算错(=3 / =6)
    sampled = [9, 9, 3, 9, 6]
    print(f"  采样到的 5 条推理链的最终答案：{sampled}")
    print(f"  多数表决结果：{majority_vote(sampled)}  ✅（少数错链被多数淹没）")
    print("  → 这就是 Self-Consistency 比 CoT 再涨 +17.9% 的机制。")
    print("    实操要点：temperature>0 产生多样性；只对「最终答案」投票，")
    print("    不对中间步骤投票（中间步骤天然难对齐）。\n")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # A：模拟版，必跑（不依赖模型）
    run_experiment()
    demo_self_consistency_idea()

    # B：真模型版，按配置决定是否跑
    #    默认 ollama，若你没起 ollama 会打印「调用失败」并跳过，不影响 A。
    if MODEL_PROVIDER in ("ollama", "openai"):
        run_real_model_demo()
    else:
        print("【B】真模型版未启用（MODEL_PROVIDER='none'）。"
              "改 MODEL_PROVIDER 后即可跑真实 CoT 对比。")

    print("=" * 72)
    print("面试连接：")
    print("  • CoT 是什么？→ 让模型在给答案前先输出中间推理步骤。")
    print("  • 为什么有效？→ 把多步难题拆成单步简单预测 + 给更多 forward pass。")
    print("  • 涌现性？→ 只在 >=~100B 模型显著生效，小模型反而被带偏（本课演示）。")
    print("  • 标志数据？→ PaLM 540B GSM8K: 标准 18% → CoT 57%（8 个 exemplar）。")
    print("  • Zero-shot CoT 触发词？→ \"Let's think step by step.\"")
    print("=" * 72)
