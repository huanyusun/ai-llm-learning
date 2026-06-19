"""04 - 多 Agent 协作：Supervisor + 专家 Workers（任务分解 → 分工 → 聚合）

与 03（单 Agent 调多个工具）的本质区别（面试高频对比）：
    03 ：一个 Agent 自己【中心化决策】"这一步调哪个工具"。
    本文件：一个 Supervisor 把复杂任务【拆解】成子任务，委派给多个【专家 Agent】
           （各自有角色 / 能力边界），各自处理后把结果【回传】到共享上下文，
           Supervisor 再让 Writer 综合出最终答案。
    这就是 "多 Agent" 相对 "多工具" 的本质：把【决策与能力】分布到多个自治 Agent，
    而不是一个 Agent 拿着全部工具包打天下。

模式：Supervisor（编排） + Workers（Researcher / Calculator / Writer）
      = 经典的 "Orchestrator-Worker" / "Plan-and-Execute" 多 Agent 范式。
对应 knowledge/Agent：Multi-Agent、Plan-and-Execute。

为可离线跑通，各 Agent 的"思考/决策"用规则路由模拟（像 01-03）。
真实环境：每个 Worker 换成一个带【各自 system prompt】的 LLM 调用——
          不同角色不同 prompt 正是「专家化」的落点（见文末真实版标注）。

运行： uv run 05-Agent/04-多Agent协作.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ============================================================================
# 一、各 Worker 共享的「能力底座」（mock 知识库 + 计算器）
#    真实环境：Researcher 接搜索/RAG，Calculator 接计算工具，Writer 就是 LLM 本身。
# ============================================================================
_KB: dict[str, str] = {
    "iPhone": "iPhone 单价 799 美元",
    "汇率": "当前 1 美元 ≈ 7.2 人民币",
    "A001": "订单 A001：共 2 台 iPhone、单价 799 美元",
}


def kb_lookup(query: str) -> str:
    """mock 知识检索：命中关键词就返回对应知识（真实环境接 RAG/搜索）。"""
    hits = [v for k, v in _KB.items() if k in query]
    return "；".join(hits) if hits else f"知识库未命中：{query}"


def safe_calc(expr: str) -> str:
    """安全求值四则运算（仅允许数字与运算符，防注入）。"""
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return f"非法表达式：{expr}"
    try:
        return f"{expr} = {eval(expr, {'__builtins__': {}}, {})}"
    except ZeroDivisionError:
        return "错误：除以零"
    except Exception as e:  # noqa: BLE001
        return f"计算失败：{e}"


# ============================================================================
# 二、Worker 基类：一个专家 Agent = 角色(system prompt) + 能力 + handle()
# ============================================================================
@dataclass
class Worker:
    """专家 Agent。role 是它的 system prompt（能力边界）；handle 执行一个子任务。"""
    name: str
    role: str  # 这个 Agent 的「人设 / system prompt」——专家化的落点

    def handle(self, task: dict, context: dict) -> str:
        raise NotImplementedError


class Researcher(Worker):
    """研究员：负责"查事实"类子任务（接知识库/搜索）。"""
    def handle(self, task: dict, context: dict) -> str:
        return kb_lookup(task["goal"])


class Calculator(Worker):
    """计算员：负责"算数"类子任务。"""
    def handle(self, task: dict, context: dict) -> str:
        return safe_calc(task["expr"])


class Writer(Worker):
    """撰稿人：负责"综合成文"类子任务——把前面搜集到的事实揉成一段话。

    真实环境：把共享上下文里的事实喂给 LLM（带 writer 的 system prompt）让它总结。
    """
    def handle(self, task: dict, context: dict) -> str:
        facts = context.get("facts", [])
        return "【总结】" + "；".join(facts)


# ============================================================================
# 三、Supervisor：编排者 —— 拆解 / 委派 / 聚合
#    它本身【不直接干活】（不查库、不算数），只决定"交给谁、先做哪个、怎么合"。
# ============================================================================
class Supervisor:
    """编排者：plan() 拆解任务，delegate() 路由给专家 Worker，run() 串起全流程。"""

    def __init__(self, workers: dict[str, Worker]):
        self.workers = workers  # 子任务 type -> Worker

    def plan(self, question: str) -> list[dict]:
        """把复杂问题拆成有序子任务（mock 剧本；真实环境用 LLM 做规划）。

        本演示问题需要三类专家协作：查事实 → 算数 → 总结。
        """
        return [
            {"type": "research", "to": "researcher", "goal": "iPhone 单价、订单 A001 数量、当前汇率"},
            {"type": "calc", "to": "calculator", "expr": "2 * 799 * 7.2"},
            {"type": "write", "to": "writer"},
        ]

    def delegate(self, subtask: dict, context: dict) -> str:
        worker = self.workers[subtask["to"]]
        print(f"    ├─ 委派 → {worker.name}（{worker.role}）")
        result = worker.handle(subtask, context)
        print(f"    │     ↳ {result}")
        return result

    def run(self, question: str) -> str:
        print("=" * 72)
        print("多 Agent 协作启动（Supervisor + 专家 Workers）")
        print("Workers :", ", ".join(w.name for w in self.workers.values()))
        print("Question:", question)
        print("=" * 72)

        plan = self.plan(question)
        print(f"[Supervisor] 拆解出 {len(plan)} 个子任务：")
        for i, st in enumerate(plan, 1):
            print(f"    {i}. {st['type']}  →  @{st['to']}")

        # 共享上下文（blackboard 模式）：各 Worker 把中间事实写进来，供后续（Writer）用
        context: dict[str, Any] = {"facts": []}
        final = ""
        for i, st in enumerate(plan, 1):
            print(f"\n[Supervisor] 执行子任务 {i}/{len(plan)}")
            result = self.delegate(st, context)
            if st["type"] == "write":
                final = result            # Writer 的产出 = 最终答案
            else:
                context["facts"].append(result)  # 其余作为事实累积

        print("\n" + "=" * 72)
        print("[Supervisor] 最终答案：")
        print("  " + final)
        print("=" * 72)
        return final


# ============================================================================
# 四、真实版标注：每个 Worker 换成带各自 system prompt 的 LLM 调用
# ============================================================================
#
# import openai
# _client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
#
# def llm_call(system_prompt: str, user_msg: str) -> str:
#     r = _client.chat.completions.create(
#         model="qwen2.5:7b-instruct",
#         messages=[{"role": "system", "content": system_prompt},
#                   {"role": "user", "content": user_msg}],
#     )
#     return r.choices[0].message.content
#
# # 关键：每个专家用【不同的 system prompt】——这正是"专家化"的落点。
# class Researcher(Worker):
#     role = "你是研究员，只负责查事实。用工具/知识库检索，不要编造。"
#     def handle(self, task, ctx):
#         return llm_call(self.role, f"请查：{task['goal']}\n已有上下文：{ctx}")
# class Calculator(Worker):
#     role = "你是计算员，只做精确四则运算，输出算式与结果。"
#     ...
# class Writer(Worker):
#     role = "你是撰稿人，把给定事实综合成一段简洁中文总结，不要加新信息。"
#     ...


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    team = Supervisor({
        "researcher": Researcher("研究员", "查事实（接知识库/搜索）"),
        "calculator": Calculator("计算员", "做精确四则运算"),
        "writer": Writer("撰稿人", "把事实综合成一段话"),
    })
    team.run("订单 A001 的 2 台 iPhone 总价折合人民币多少？请用一句话总结。")
