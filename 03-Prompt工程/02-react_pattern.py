"""
Prompt 工程 第2课：手写 ReAct 循环（Thought-Action-Observation）
==================================================================
理论来源：
- Yao et al., 2022, ReAct: Synergistic Reasoning and Acting in Language
  Models, arXiv:2210.03629（ICLR 2023）。
  核心论断：让模型「交错输出」推理痕迹(Thought)与任务行动(Action)，
  行动拿到观察(Observation)后再回到推理——形成
      Thought → Action → Observation
  循环。它克服了纯 CoT 的两大顽疾：幻觉 与 错误传播。
  标志数据：HotpotQA/FEVER 上比 CoT 更可信；ALFWorld 比模仿/强化学习
  高 +34% 绝对成功率，WebShop +10%，且只用 1~2 个 in-context 示例。

本文件【完全用纯 Python】实现一个不依赖任何 LLM 框架（LangChain 等）
的 ReAct 循环。为了「能直接跑」，我们用「脚本化的模拟 Agent」扮演
模型：它按 ReAct 范式产出 Thought/Action，调用一组「模拟工具」拿到
Observation，直到给出最终答案。

之所以用模拟 Agent：ReAct 的精髓是「循环结构 + 工具接口 + 终止条件」，
这些与具体模型无关。把循环写清楚，换成真模型只需要把
`ScriptedAgent.decide()` 换成一次 LLM 调用（见文件末尾「接真模型」说明）。

运行：uv run python "03-Prompt工程/02-react_pattern.py"
"""
from __future__ import annotations

import re
from typing import Callable


# ============================================================
# 【1】定义「工具」：ReAct 里的 Action 受限动作空间
# ============================================================
# 论文 [arXiv:2210.03629] 原始动作空间是 Search/Lookup/Finish。
# 这里我们用一组「模拟知识库」工具，演示「外部信息如何回填 Observation」。
# 在真实系统里，这些函数会换成：搜索引擎 API、数据库查询、计算器、代码执行等。

# 一个迷你的「事实知识库」，模拟 Wikipedia API（论文里就是用 Wikipedia）。
_FAKE_KB: dict[str, str] = {
    "北京": "北京是中华人民共和国的首都，常住人口约 2189 万（2020）。"
            "它位于华北平原北部，是中国的政治、文化中心。",
    "上海": "上海是中国直辖市，常住人口约 2487 万（2020），是中国最大的经济中心。",
    "东京": "东京是日本的首都，东京都人口约 1396 万（2021）。",
    "长城": "长城是中国古代军事防御工程，总长超过 2.1 万千米，"
            "1987 年被列入世界文化遗产。",
    "故宫": "故宫位于北京中轴线中心，是明清两代皇宫，建于 1420 年。",
    "人口": "中华人民共和国 2020 年第七次人口普查总人口约 14.1 亿。",
}


def tool_search(query: str) -> str:
    """Search[q]：在模拟知识库里查 q，返回相关条目（找不到给提示）。"""
    q = query.strip()
    if q in _FAKE_KB:
        return _FAKE_KB[q]
    # 容错：包含关系
    for key, val in _FAKE_KB.items():
        if q in key or key in q:
            return val
    return f"（知识库中没有关于「{q}」的条目，请换个关键词或拆分子问题。）"


def tool_calculator(expr: str) -> str:
    """Calculate[expr]：安全地算一个纯算式（只允许数字和 +-*/() ）。"""
    expr = expr.strip()
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return "（拒绝执行：算式含非法字符，仅支持 + - * / 和括号。）"
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 已用正则白名单限制
        return f"{expr} = {result}"
    except ZeroDivisionError:
        return "（除零错误。）"
    except Exception as e:  # noqa: BLE001
        return f"（计算失败：{e}）"


def tool_finish(answer: str) -> str:
    """Finish[answer]：终止循环并给出最终答案。返回一个哨兵字符串。"""
    return f"__FINISH__{answer}"


# 工具注册表：动作名 → 函数。这是 ReAct Agent 的「能力边界」。
TOOLS: dict[str, Callable[[str], str]] = {
    "Search": tool_search,
    "Calculate": tool_calculator,
    "Finish": tool_finish,
}


# ============================================================
# 【2】ReAct 循环本体（与具体模型无关的核心逻辑）
# ============================================================
class ReActLoop:
    """
    一个不依赖框架的 ReAct 循环。

    参数：
        agent：任何带 .decide(history) -> (thought, action_name, action_arg) 的对象。
               模拟版用 ScriptedAgent；真模型版把 LLM 包一层即可。
        tools：动作名 → 可调用函数。
        max_steps：防止死循环（论文也强调长轨迹会「卡住」，需设上限）。

    循环格式（论文范式）：
        Thought i: <推理>
        Action i: <ToolName>[<arg>]
        Observation i: <工具返回>
        ... 直到 Action = Finish[...]
    """

    def __init__(self, agent, tools: dict[str, Callable[[str], str]],
                 max_steps: int = 8):
        self.agent = agent
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> tuple[str, list[str]]:
        """
        跑一次完整的 Thought-Action-Observation 循环。
        返回 (最终答案, 轨迹日志)。
        """
        log: list[str] = [f"Question: {question}", ""]
        history = ""  # 累积上下文，喂给 agent 做下一步决策

        for step in range(1, self.max_steps + 1):
            # 1) Agent（模型）产出 Thought + Action
            thought, action_name, action_arg = self.agent.decide(question, history)
            log.append(f"Thought {step}: {thought}")
            log.append(f"Action {step}: {action_name}[{action_arg}]")

            # 2) 校验动作是否在工具表里（模型可能输出非法动作）
            if action_name not in self.tools:
                obs = f"（未知动作「{action_name}」，可用：{list(self.tools)}）"
                log.append(f"Observation {step}: {obs}")
                history += f"\nThought {step}: {thought}\nAction {step}: {action_name}[{action_arg}]\nObservation {step}: {obs}"
                continue

            # 3) 执行工具，拿到 Observation
            result = self.tools[action_name](action_arg)
            if action_name == "Finish":
                answer = result.replace("__FINISH__", "", 1)
                log.append(f"Observation {step}: 任务完成，最终答案 = {answer}")
                return answer, log

            log.append(f"Observation {step}: {result}")
            history += (
                f"\nThought {step}: {thought}\n"
                f"Action {step}: {action_name}[{action_arg}]\n"
                f"Observation {step}: {result}"
            )

        # 超过最大步数仍没 Finish —— 兜底（论文强调长轨迹要设上限）
        log.append(f"（达到最大步数 {self.max_steps}，强制停止，未给出答案。）")
        return "(未能在限定步数内完成)", log


# ============================================================
# 【3】模拟 Agent：脚本化地「扮演」一个会推理的模型
# ============================================================
# 这里用一个状态机模拟模型决策，目的是让 ReAct 循环能「直接跑通」、
# 让你肉眼看到 Thought-Action-Observation 的真实交互节奏。
# 把它换成真模型只需：decide() 内部拼好 ReAct prompt，调一次 LLM，
# 再用正则解析出 thought / action_name / action_arg 即可（见文件末尾）。
class ScriptedAgent:
    """按预设剧本推进的模拟 Agent。演示两道题的不同轨迹。"""

    def __init__(self, script: list[tuple[str, str, str, str]]):
        # 每条：(触发条件关键词, thought, action_name, action_arg)
        # 触发条件用「当前 history 是否包含某关键词」来推进状态机。
        self.script = script
        self._idx = 0

    def decide(self, question: str, history: str) -> tuple[str, str, str]:
        """根据 history 推进到剧本下一步。"""
        # 第一条总是触发（history 为空）
        while self._idx < len(self.script):
            trigger, thought, action_name, action_arg = self.script[self._idx]
            self._idx += 1
            if trigger == "" or trigger in history or trigger in question:
                return thought, action_name, action_arg
        # 剧本用尽 → 兜底 Finish
        return ("剧本用尽，直接收尾。", "Finish", "(未知)")


# ============================================================
# 【4】两个演示场景：单跳问答 + 多跳问答
# ============================================================
def case_single_hop() -> None:
    """场景1：单跳问答（只需一次 Search 就能答）。"""
    print("=" * 72)
    print("场景1：单跳问答 —— 「长城是什么？」")
    print("=" * 72)
    agent = ScriptedAgent([
        ("", "用户问长城是什么，我先用知识库查「长城」。",
         "Search", "长城"),
        ("世界文化遗产", "知识库已给出长城的定义和价值，信息充分，收尾。",
         "Finish", "长城是中国古代军事防御工程，总长超 2.1 万千米，"
                   "1987 年被列入世界文化遗产。"),
    ])
    loop = ReActLoop(agent, TOOLS, max_steps=6)
    answer, log = loop.run("长城是什么？")
    print("\n".join(log))
    print("-" * 72)
    print(f"最终答案：{answer}\n")


def case_multi_hop() -> None:
    """场景2：多跳问答（需要先查 A，再基于 A 查 B，再计算）。
    这正是 ReAct 论文 HotpotQA 场景的缩影：纯 CoT 会幻觉/错传，
    ReAct 用工具逐步把真实信息拉回来。"""
    print("=" * 72)
    print("场景2：多跳问答 —— 「北京和上海哪个城市人口多？多多少？」")
    print("=" * 72)
    agent = ScriptedAgent([
        ("", "这是一个比较题，需要分别拿到北京和上海的人口，再相减。"
         "先查北京人口。", "Search", "北京"),
        ("北京", "拿到北京人口约 2189 万。再查上海人口。", "Search", "上海"),
        ("上海", "拿到上海约 2487 万。现在用计算器算差值 2487-2189。",
         "Calculate", "2487-2189"),
        ("298", "算出差值 298 万。信息齐全，收尾。",
         "Finish", "上海人口更多。上海约 2487 万，北京约 2189 万，"
                   "上海比北京多约 298 万人。"),
    ])
    loop = ReActLoop(agent, TOOLS, max_steps=8)
    answer, log = loop.run("北京和上海哪个城市人口多？多多少？")
    print("\n".join(log))
    print("-" * 72)
    print(f"最终答案：{answer}\n")


def case_error_recovery() -> None:
    """场景3：错误恢复 —— Agent 用了非法动作名，循环不该崩溃，
    而是把「未知动作」作为 Observation 回灌，让 Agent 自我纠偏。
    这演示了 ReAct 论文里「handle exceptions」的能力。"""
    print("=" * 72)
    print("场景3：错误恢复 —— Agent 第一步误用动作「Wiki」，循环回灌纠偏")
    print("=" * 72)
    agent = ScriptedAgent([
        ("", "我想查北京，先试着调 Wiki 工具。", "Wiki", "北京"),
        ("未知动作", "哦，没有 Wiki 这个动作，可用的有 Search。改用 Search。",
         "Search", "北京"),
        ("首都", "拿到了北京的信息，收尾。", "Finish", "北京是中国的首都。"),
    ])
    loop = ReActLoop(agent, TOOLS, max_steps=8)
    answer, log = loop.run("介绍一下北京。")
    print("\n".join(log))
    print("-" * 72)
    print(f"最终答案：{answer}\n")


# ============================================================
# 【5】ReAct Prompt 模板（接真模型时直接用）
# ============================================================
REACT_SYSTEM_PROMPT = """你是一个 ReAct Agent。请严格按下面格式交替输出 \
Thought（推理）与 Action（行动），直到你能给出最终答案。

可用动作（只能用这几种）：
- Search[查询词]：在知识库中检索。
- Calculate[算式]：计算一个仅含 + - * / 和括号的算式。
- Finish[最终答案]：给出最终答案并结束。

格式（每一步三行，缺一不可）：
Thought <step>: <你的推理，分析当前观察、决定下一步>
Action <step>: <ActionName>[<参数>]

之后系统会把 Observation（工具返回）追加进来，你再输出下一轮 \
Thought/Action。当你认为信息足够，用 Finish[...] 结束。
不要输出 Observation，它由系统填回。
"""


def print_react_template() -> None:
    print("=" * 72)
    print("【附】ReAct System Prompt 模板（接真模型时直接拷贝）")
    print("=" * 72)
    print(REACT_SYSTEM_PROMPT)


# ============================================================
# 【接真模型】说明（不跑，仅文档）
# ============================================================
# 把 ScriptedAgent.decide() 换成真模型调用，循环结构原封不动：
#
#   import json, urllib.request
#   def llm(prompt: str) -> str:  # ollama 或 OpenAI 兼容
#       payload = json.dumps({"model": "qwen2.5:7b", "prompt": prompt,
#                              "stream": False}).encode()
#       req = urllib.request.Request("http://localhost:11434/api/generate",
#                                    data=payload,
#                                    headers={"Content-Type": "application/json"})
#       with urllib.request.urlopen(req) as r:
#           return json.loads(r.read())["response"]
#
#   class LlmAgent:
#       def decide(self, question, history):
#           prompt = REACT_SYSTEM_PROMPT + f"\nQuestion: {question}\n" + history
#           out = llm(prompt)
#           # 用正则解析出模型输出的 Thought 和 Action[...]
#           m = re.search(r"Action[^:]*:\s*(\w+)\s*\[([^\]]*)\]", out)
#           thought = re.search(r"Thought[^:]*:\s*(.*)", out).group(1)
#           name, arg = m.group(1), m.group(2)
#           return thought, name, arg
#
#   loop = ReActLoop(LlmAgent(), TOOLS, max_steps=8)
#   print(loop.run("你的问题"))
#
# 注意真模型要做的两件事：①解析 Action 名和参数（正则/JSON）；
# ②给 max_steps 兜底（长轨迹会卡住/重复 Action，论文明确警告）。


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    case_single_hop()
    case_multi_hop()
    case_error_recovery()
    print_react_template()
    print("=" * 72)
    print("面试连接：")
    print("  • ReAct 解决 CoT 的什么问题？→ 幻觉 + 错误传播，用工具拿真实信息。")
    print("  • 循环结构？→ Thought(推理) → Action(调工具) → Observation(回填) ↺")
    print("  • 终止条件？→ Action = Finish[answer]（或达 max_steps 兜底）。")
    print("  • 标志数据？→ ALFWorld +34%、WebShop +10% 绝对成功率，仅用 1-2 例。")
    print("  • 与纯 Act 区别？→ 有 Thought 才能规划、纠错（场景3 演示纠偏）。")
    print("=" * 72)
