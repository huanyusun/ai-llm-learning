"""02 - 手写 ReAct Agent 完整循环（Thought → Action → Observation → ... → Final Answer）

参考 [S1] ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
论文核心：把"思考"也纳入动作空间 A ∪ L（语言空间），让推理与行动交替产生协同。
论文模板（附录 C，原文抓取）：

    Question: <问题>
    Thought 1: <分解问题 / 决定搜什么>
    Action 1: Search[<实体>]
    Observation 1: <结果>
    Thought 2: <从观察提取信息 / 推理>
    Action 2: Lookup[<字符串>] 或 Search[<新实体>]
    Observation 2: ...
    Thought N: <综合信息得出结论>
    Action N: Finish[<最终答案>]

ReAct 的动作空间设计得很朴素（论文特意为之）：
    - Search[entity]   搜索实体，返回前几句
    - Lookup[string]   在当前页面里向后查找含字符串的句子（模拟 Ctrl+F）
    - Finish[answer]   结束任务并给出最终答案

为可离线直接跑通，本文件：
    1. 内置一个"模拟知识库"（替代真实 Wikipedia API）；
    2. 用一个"脚本化 LLM"——按预定的 Thought/Action 序列产出，让你能直观看到
       ReAct 的完整推理轨迹（真实环境只需把 mock_llm() 换成真实模型调用即可）。
    3. 完整打印每一步的 Thought/Action/Observation，最后给出 Final Answer。

运行： uv run 05-Agent/02-react_agent.py
"""

from __future__ import annotations

from typing import Callable


# ============================================================================
# 一、模拟外部环境：一个微型"知识库 + 搜索引擎"
# ============================================================================

# 用一个小型 dict 模拟 Wikipedia 页面（key=实体，value=页面正文，按句号分句）
KNOWLEDGE_BASE: dict[str, str] = {
    "ReAct": (
        "ReAct 是 2022 年由 Princeton 和 Google 的研究者提出的范式。"
        "它让大语言模型交替进行推理（Reasoning）与行动（Acting）。"
        "ReAct 论文发表在 arXiv 上，编号 2210.03629。"
        "该论文的第一作者是 Shunyu Yao。"
    ),
    "Shunyu Yao": (
        "Shunyu Yao 是 Princeton 大学 NLP 组的博士生。"
        "他因提出 ReAct 范式而知名。"
        "他还参与撰写了 Tree of Thoughts 论文。"
    ),
    "Python": (
        "Python 是一种解释型、高级编程语言。"
        "Python 由 Guido van Rossum 于 1991 年发布。"
    ),
}


def search(entity: str) -> str:
    """ReAct 的 Search[entity]：返回该实体页面的前几句。论文里是前 5 句，这里给前 3 句。"""
    if entity in KNOWLEDGE_BASE:
        sentences = [s.strip() for s in KNOWLEDGE_BASE[entity].split("。") if s.strip()]
        return "。".join(sentences[:3]) + "。"
    # 不存在则返回最相似的几个实体（论文行为）
    similar = [k for k in KNOWLEDGE_BASE if entity.lower() in k.lower() or k.lower() in entity.lower()]
    return f"未找到 {entity}。相似实体：{similar or '无'}"


def lookup(string: str) -> str:
    """ReAct 的 Lookup[string]：在"所有页面"里找含 string 的下一句（模拟 Ctrl+F）。"""
    for page in KNOWLEDGE_BASE.values():
        for sent in page.split("。"):
            if string in sent:
                return sent.strip() + "。"
    return f"未找到包含「{string}」的句子。"


# ReAct 动作空间注册表：动作名 -> (执行函数, 是否为结束动作)
ACTIONS: dict[str, tuple[Callable[[str], str], bool]] = {
    "Search": (search, False),
    "Lookup": (lookup, False),
    "Finish": (lambda ans: ans, True),  # Finish[answer] 直接返回答案，结束循环
}


# ============================================================================
# 二、ReAct 提示模板（论文附录 C 的骨架）
# ============================================================================

REACT_PROMPT_TEMPLATE = """\
你是一个使用 ReAct（Reasoning + Acting）范式回答问题的 Agent。
对每一步，你必须严格按照如下格式输出：

Thought N: <你的推理：分解问题 / 决定下一步动作 / 综合观察>
Action N: <Search[实体] | Lookup[字符串] | Finish[最终答案]>

收到 Observation 后继续下一轮 Thought/Action，直到用 Finish[...] 给出最终答案。

Question: {question}
"""


# ============================================================================
# 三、脚本化 LLM：模拟"冻结大模型 + few-shot 提示"产出的轨迹
#     （真实环境把这里换成 client.chat.completions.create(...) 即可）
# ============================================================================

def mock_llm_step(question: str, history: list[tuple[str, str, str]], step: int) -> tuple[str, str]:
    """返回 (thought, action_text)。

    这里按问题类型预编排了 ReAct 轨迹，让推理链路清晰可见。
    生产中：把 REACT_PROMPT_TEMPLATE + 历史 Observation 喂给真实模型，
    让模型自己输出 "Thought N: ... Action N: ..."，再用正则解析。
    """
    # 一个需要"搜索 + 再搜索 + 综合"两跳推理的问题
    if question.startswith("ReAct 论文的第一作者是谁") and step == 1:
        return (
            "我需要先找到 ReAct 论文的相关信息，确定它的第一作者。",
            "Search[ReAct]",
        )
    if step == 2:
        # 上一步 Observation 提到了 "Shunyu Yao"
        return (
            "Observation 提到 ReAct 论文的第一作者是 Shunyu Yao，"
            "我需要再搜一下确认这个人的身份。",
            "Search[Shunyu Yao]",
        )
    if step == 3:
        return (
            "现在确认了 Shunyu Yao 是 Princeton 博士生，且因提出 ReAct 而知名。"
            "信息足够，给出最终答案。",
            "Finish[ReAct 论文的第一作者是 Shunyu Yao，Princeton 大学 NLP 组博士生。]",
        )
    # 兜底：直接结束
    return ("信息不足，结束。", "Finish[无法确定答案。]")


# ============================================================================
# 四、ReAct 主循环：编排 Thought → Action → Observation，直到 Finish
# ============================================================================

def parse_action(action_text: str) -> tuple[str, str]:
    """把 'Search[ReAct]' 解析成 ('Search', 'ReAct')。"""
    head, _, body = action_text.partition("[")
    body = body.rstrip("]")
    return head.strip(), body.strip()


def react_agent(question: str, max_steps: int = 7) -> str:
    """ReAct 主循环。

    论文里 HotpotQA 步数上限设 7、Fever 设 5（更多步不提升性能 [S1]）。
    这里默认 7。
    """
    print("=" * 72)
    print("ReAct Agent 启动")
    print(REACT_PROMPT_TEMPLATE.format(question=question))
    print("=" * 72)

    history: list[tuple[str, str, str]] = []  # [(thought, action, observation), ...]

    for step in range(1, max_steps + 1):
        # 1) LLM 产出 Thought + Action
        thought, action_text = mock_llm_step(question, history, step)
        print(f"Thought {step}: {thought}")
        print(f"Action {step}:  {action_text}")

        action_name, arg = parse_action(action_text)
        if action_name not in ACTIONS:
            print(f"[错误] 未知动作：{action_name}")
            break

        func, is_finish = ACTIONS[action_name]

        # 2) Finish 则结束
        if is_finish:
            print()
            print(">>> Final Answer:", arg)
            return arg

        # 3) 否则执行动作得到 Observation
        observation = func(arg)
        print(f"Observation {step}: {observation}")
        print("-" * 72)
        history.append((thought, action_text, observation))

    print("[警告] 达到最大步数仍未 Finish（论文里这种情况下回退到 CoT-SC [S1]）")
    return ""


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    # 一个需要 2 跳推理的问题：ReAct → 第一作者 → 作者身份
    react_agent("ReAct 论文的第一作者是谁？他是哪个学校的？")
