"""03 - 多工具 Agent：搜索 + 计算器 + 数据库查询，演示"工具选择"

本文件把 01（Function Calling 决策）与 02（ReAct 循环）结合，做一个能从
"多个工具"里挑选用哪个、并支持"多轮调用 + 组合"的 Agent。

核心看点（面试重点）：
    1. 工具选择（tool selection）：用户一句话可能涉及多个工具，Agent 要先想清楚
       "这一步该用哪个"，而不是把所有工具一股脑全调。
    2. 多轮组合：复杂问题需要"先搜索拿数据 → 再用计算器加工 → 再回答"。
    3. 单一动作 vs 并行调用：本演示每轮只调一个工具（ReAct 风格），
       并在文末标注真实环境如何用 parallel_tool_calls 一次调多个（[S8]）。

工具集：
    - search(query)         模拟知识搜索
    - calculator(expr)      四则运算
    - query_db(table, key)  模拟查数据库（用户订单/库存等业务表）

为可离线跑通，模型决策用"规则路由 + 上下文状态机"模拟；完整推理轨迹会逐行打印。
真实环境把 step_llm() 换成真实 LLM 的 function-calling 接口即可（文末有标注）。

运行： uv run 05-Agent/03-多工具agent.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================================
# 一、三个工具的定义与实现
# ============================================================================

# 工具 1：搜索
def search(query: str) -> str:
    """模拟搜索引擎/知识库检索。"""
    db = {
        "iPhone": "iPhone 最新款起售价 799 美元，存储 128GB 起。",
        "人民币汇率": "当前 1 美元 ≈ 7.2 人民币。",
    }
    for k, v in db.items():
        if k in query:
            return v
    return f"搜索 {query}：未找到相关信息。"


# 工具 2：计算器（复用 01 的安全求值）
def calculator(expression: str) -> str:
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return f"非法表达式：{expression}"
    try:
        return f"{expression} = {eval(expression, {'__builtins__': {}}, {})}"
    except ZeroDivisionError:
        return "错误：除以零"
    except Exception as e:  # noqa: BLE001
        return f"计算失败：{e}"


# 工具 3：数据库查询（模拟业务表）
_DB: dict[str, dict[str, Any]] = {
    "orders": {  # 订单表
        "A001": {"product": "iPhone", "qty": 2, "price_usd": 799},
        "A002": {"product": "AirPods", "qty": 3, "price_usd": 199},
    },
    "inventory": {  # 库存表
        "iPhone": {"stock": 15},
        "AirPods": {"stock": 40},
    },
}


def query_db(table: str, key: str) -> str:
    """模拟查数据库：table 是表名，key 是主键。"""
    if table not in _DB:
        return f"表 {table} 不存在。"
    row = _DB[table].get(key)
    if row is None:
        return f"在 {table} 中未找到 {key}。"
    return json.dumps(row, ensure_ascii=False)


# 工具注册表：name -> (callable, schema)
# schema 是给"模型"看的；callable 是"宿主"真正执行的。
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "search": {
        "callable": search,
        "schema": {
            "name": "search",
            "description": "搜索商品信息、汇率、常识等公开知识",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "calculator": {
        "callable": calculator,
        "schema": {
            "name": "calculator",
            "description": "计算四则运算表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "query_db": {
        "callable": query_db,
        "schema": {
            "name": "query_db",
            "description": "查询业务数据库（orders 订单表 / inventory 库存表）",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "enum": ["orders", "inventory"]},
                    "key": {"type": "string"},
                },
                "required": ["table", "key"],
            },
        },
    },
}


# ============================================================================
# 二、Agent 运行时：维护对话/工具历史，作为"工作记忆"
# ============================================================================

@dataclass
class AgentState:
    """Agent 的短期记忆：累积的 Thought/Action/Observation。"""
    question: str
    history: list[dict[str, str]] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)  # 中间变量缓存


# ============================================================================
# 三、模拟"模型决策"：工具选择路由
#     （真实环境：把工具 schema 全部塞进 tools 字段，让 LLM 自己选）
# ============================================================================

def step_llm(state: AgentState, step: int) -> tuple[str, str, dict[str, Any]]:
    """返回 (thought, action_text, args)。

    下面的逻辑是"脚本化"的多工具协作轨迹，用来演示一个真实场景：
        用户问："订单 A001 的总价折合人民币多少？库存还够吗？"

    正确的 Agent 思路应该是：
        step1: 查订单 A001   -> 拿到 qty=2, price_usd=799
        step2: 算总价 2*799
        step3: 查汇率        -> 7.2
        step4: 算人民币总价 1598*7.2
        step5: 查库存 iPhone -> stock=15
        step6: 综合给最终答案

    这正是"工具选择 + 多轮组合"的最佳演示：每一步挑一个工具，串成一条链。
    """
    q = state.question

    if "A001" in q and step == 1:
        return ("先查订单 A001 的明细，拿到数量和单价。", "query_db", {"table": "orders", "key": "A001"})

    if step == 2:
        return ("订单里有 2 台 iPhone，单价 799 美元，先算美元总价。", "calculator", {"expression": "2 * 799"})

    if step == 3:
        return ("算出美元总价 1598，需要汇率才能折算人民币，搜一下当前汇率。", "search", {"query": "人民币汇率"})

    if step == 4:
        return ("汇率是 7.2，算人民币总价：1598 * 7.2。", "calculator", {"expression": "1598 * 7.2"})

    if step == 5:
        return ("还需回答库存是否够：订单是 2 台 iPhone，查一下 iPhone 库存。", "query_db", {"table": "inventory", "key": "iPhone"})

    # step 6: Finish
    return (
        "信息齐全：订单 A001 是 2 台 iPhone，美元总价 1598，折合人民币 11505.6，"
        "iPhone 当前库存 15 台，足够发货。",
        "Finish",
        {"answer": "订单 A001（2 台 iPhone）总价 1598 美元，按汇率 7.2 折合 11505.6 元人民币；iPhone 库存 15 台，足够发货。"},
    )


# ============================================================================
# 四、Agent 主循环：工具选择 + 多轮组合 + 终止
# ============================================================================

def run_agent(question: str, max_steps: int = 8) -> str:
    state = AgentState(question=question)
    print("=" * 72)
    print("多工具 Agent 启动")
    print("可用工具:", ", ".join(TOOL_REGISTRY))
    print("Question:", question)
    print("=" * 72)

    for step in range(1, max_steps + 1):
        # 1) 模型决策：选哪个工具 / 给什么参数 / 是否结束
        thought, action_name, args = step_llm(state, step)
        print(f"[step {step}]")
        print(f"  Thought : {thought}")
        print(f"  Action  : {action_name}({args})")

        # 2) Finish：直接给最终答案
        if action_name == "Finish":
            answer = args.get("answer", "")
            state.history.append({"thought": thought, "action": action_name, "result": answer})
            print(f"  >>> Final Answer: {answer}")
            print()
            return answer

        # 3) 否则执行工具
        if action_name not in TOOL_REGISTRY:
            print(f"  [错误] 未知工具 {action_name}")
            break
        result = TOOL_REGISTRY[action_name]["callable"](**args)
        print(f"  Result  : {result}")
        print("-" * 72)
        state.history.append({"thought": thought, "action": action_name, "args": args, "result": result})

    print("[警告] 达到最大步数仍未给出 Final Answer。")
    return ""


# ============================================================================
# 五、真实版标注：用 OpenAI 兼容接口（ollama）做多工具并行调用
# ============================================================================
#
# import openai
#
# def run_agent_real(question: str) -> str:
#     client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
#     tools = [{"type": "function", "function": t["schema"]} for t in TOOL_REGISTRY.values()]
#     messages = [
#         {"role": "system", "content": "你是一个能调用多个工具的业务助手。需要时逐步调用工具。"},
#         {"role": "user", "content": question},
#     ]
#     while True:
#         resp = client.chat.completions.create(model="qwen2.5:7b-instruct", messages=messages, tools=tools)
#         msg = resp.choices[0].message
#         if not msg.tool_calls:        # 模型不再调工具 -> 最终答案
#             return msg.content
#         messages.append(msg)
#         for tc in msg.tool_calls:     # 可能并行多个 tool_call（parallel tool calls [S8]）
#             name, args = tc.function.name, json.loads(tc.function.arguments)
#             result = TOOL_REGISTRY[name]["callable"](**args)
#             messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    run_agent("订单 A001 的总价折合人民币多少？库存还够吗？")
