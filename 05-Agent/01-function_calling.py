"""01 - 函数调用 / 工具调用（Function Calling）原理演示

本文件演示厂商原生 Function Calling 的核心机制（参考 [S8] OpenAI Function Calling 指南）：

    1. 用 JSON Schema 向模型"声明"可用工具（get_weather / calculator）；
    2. 模型根据用户意图"决定"调用哪个工具、生成结构化参数；
    3. 宿主程序（本文件）真正"执行"工具；
    4. 把结果回传给模型，模型再组织成自然语言答案。

关键点（面试常考）：
    - Function Calling 是"厂商在训练时内置"的能力；模型只负责产出结构化的 tool_call
      （含函数名 + 参数），真正的函数执行永远发生在宿主侧（沙箱外）。
    - 与 ReAct（prompt 模板）/ Toolformer（自监督微调）并称工具使用的三条路线。
    - 生产里用 parallel_tool_calls + strict:true 保证参数严格匹配 JSON Schema。

为让它"零依赖能直接跑通"，下面用纯 Python 写一个"模拟 LLM"——它通过简单的
关键词规则模拟"模型决定调哪个工具"。文件末尾还给出了接 ollama 真实模型的标注代码
（OpenAI 兼容接口），无需安装额外依赖即可阅读理解，运行需要 ollama 服务。

运行： uv run 05-Agent/01-function_calling.py
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


# ============================================================================
# 一、定义工具（每个工具 = 一个可执行 Python 函数 + 一份 JSON Schema 声明）
# ============================================================================

def get_weather(city: str) -> str:
    """获取指定城市的天气（这里是模拟数据）。"""
    # 真实环境里这里会调用天气 API；为可离线运行，用伪数据。
    fake = {
        "北京": "晴，气温 28°C，湿度 40%",
        "上海": "多云，气温 26°C，湿度 65%",
        "深圳": "雷阵雨，气温 30°C，湿度 80%",
    }
    return fake.get(city, f"{city}：暂无天气数据（模拟）")


def calculator(expression: str) -> str:
    """计算四则运算表达式（仅支持数字与 + - * / ( )，安全求值）。"""
    # 安全起见：只允许数字与运算符，拒绝任意名字/函数调用（防注入）。
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return f"非法表达式：{expression}"
    try:
        # 注意：生产环境请用 ast.literal_eval 或 sympy，这里 demo 用 eval 仅因已白名单。
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "错误：除以零"
    except Exception as e:  # noqa: BLE001
        return f"计算失败：{e}"


# 工具注册表：名字 -> (可执行函数, JSON Schema 描述)
# 这里的 schema 就是厂商 Function Calling 要求你提供的 "tools" 字段。
TOOLS: dict[str, dict[str, Any]] = {
    "get_weather": {
        "callable": get_weather,
        "schema": {
            "name": "get_weather",
            "description": "查询某个城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京、上海"},
                },
                "required": ["city"],
            },
        },
    },
    "calculator": {
        "callable": calculator,
        "schema": {
            "name": "calculator",
            "description": "计算四则运算数学表达式，如 12 * (3 + 4)",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
}


def tool_schemas_for_model() -> list[dict[str, Any]]:
    """返回给模型看的工具声明列表（去掉可执行函数，只留 schema）。"""
    return [t["schema"] for t in TOOLS.values()]


# ============================================================================
# 二、模拟 LLM：根据用户输入"决定"调用哪个工具（生产中这一步由真实模型完成）
# ============================================================================

def mock_llm_decide_tool(user_input: str) -> dict[str, Any] | None:
    """模拟模型输出一个 tool_call。

    真实模型会阅读 tools 列表的 description，并输出类似下面的结构化结果：
        {
          "name": "get_weather",
          "arguments": {"city": "北京"}
        }
    这里用关键词规则"伪装"这个决策过程，便于离线演示完整链路。
    """
    text = user_input

    # 天气意图：句子里出现"天气/气温/下雨/温度 + 城市"
    m = re.search(r"(北京|上海|深圳)", text)
    if m and any(k in text for k in ("天气", "气温", "温度", "下雨", "热", "冷")):
        return {"name": "get_weather", "arguments": {"city": m.group(1)}}

    # 计算意图：句子里出现数学表达式
    expr = re.search(r"([\d\s+\-*/().]+)", text)
    if expr and any(op in text for op in "+-*/"):
        # 去掉首尾空白
        e = expr.group(1).strip()
        if e:
            return {"name": "calculator", "arguments": {"expression": e}}

    # 不需要工具：模型直接回答
    return None


def mock_llm_final_answer(user_input: str, tool_result: str | None) -> str:
    """模拟模型把工具结果组织成自然语言最终答案。"""
    if tool_result is None:
        return f"（模型直接回答）你好，我无法从「{user_input}」中识别出可调用的工具。"
    return f"（模型综合工具结果回答）根据查询：{tool_result}"


# ============================================================================
# 三、宿主程序：编排 "模型决策 -> 执行工具 -> 回传结果"
# ============================================================================

def run(user_input: str) -> None:
    print("=" * 72)
    print(f"用户输入: {user_input}")
    print("-" * 72)
    print(f"[1] 把工具声明传给模型: {json.dumps(tool_schemas_for_model(), ensure_ascii=False)}")

    tool_call = mock_llm_decide_tool(user_input)
    if tool_call is None:
        print("[2] 模型决策: 不需要工具，直接回答")
        print("[3] 最终答案:", mock_llm_final_answer(user_input, None))
        return

    name = tool_call["name"]
    args = tool_call["arguments"]
    print(f"[2] 模型决策: 调用工具 {name}({args})")

    # 宿主真正执行工具（注意：模型从不直接执行，永远是宿主在沙箱外执行）
    func: Callable = TOOLS[name]["callable"]
    observation = func(**args)
    print(f"[3] 宿主执行工具，得到 Observation: {observation}")

    answer = mock_llm_final_answer(user_input, observation)
    print(f"[4] 模型把结果组织成最终答案: {answer}")


# ============================================================================
# 四、真实版（接 ollama，OpenAI 兼容接口）—— 标注，需 ollama 服务才能跑
# ============================================================================
#
# import openai  # uv pip install openai
#
# def real_llm_function_calling(user_input: str) -> None:
#     client = openai.OpenAI(
#         base_url="http://localhost:11434/v1",   # ollama 的 OpenAI 兼容端点
#         api_key="ollama",                        # ollama 不校验 key，随便填
#     )
#     # tools 字段就是上面的 tool_schemas_for_model()，但要套成 OpenAI 格式：
#     tools_payload = [
#         {"type": "function", "function": t} for t in tool_schemas_for_model()
#     ]
#     messages = [{"role": "user", "content": user_input}]
#
#     # 第 1 次调用：模型决定是否要调用工具
#     resp = client.chat.completions.create(
#         model="qwen2.5:7b-instruct",   # 任意支持 function calling 的模型
#         messages=messages,
#         tools=tools_payload,
#     )
#     msg = resp.choices[0].message
#
#     if msg.tool_calls:                       # 模型要求调用工具
#         for tc in msg.tool_calls:
#             name = tc.function.name
#             args = json.loads(tc.function.arguments)
#             result = TOOLS[name]["callable"](**args)   # 宿主执行
#             # 把工具结果以 role=tool 回传给模型
#             messages.append(msg)
#             messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
#         # 第 2 次调用：模型基于工具结果生成最终答案
#         resp2 = client.chat.completions.create(model="qwen2.5:7b-instruct", messages=messages)
#         print(resp2.choices[0].message.content)
#     else:
#         print(msg.content)                  # 模型直接回答


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    # 三个典型用户输入，覆盖"调天气""调计算器""不调工具"三种决策路径
    run("北京今天天气怎么样？")
    run("帮我算一下 12 * (3 + 4) 等于多少")
    run("你好，给我讲个笑话")  # 模型应判定不需要工具
