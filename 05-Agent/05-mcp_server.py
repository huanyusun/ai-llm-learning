"""05 - 最小 MCP Server（从零写 JSON-RPC over stdio，零依赖）

把 knowledge/Agent §MCP 讲的协议「跑出来给人看」：不依赖官方 mcp SDK，从零用标准库
实现一个【真能通信】的最小 MCP server，再用一个内置 mock client 把完整握手 + 工具调用
的 JSON-RPC 往返打印出来。

【MCP 协议要点（连回 knowledge/Agent §3）】
    - 传输：JSON-RPC 2.0 over stdio。消息【按行分隔】（每行一个 JSON 对象），
      MUST NOT 含内嵌换行；【不用】 Content-Length 分帧（那是 LSP 的做法，别混）。
    - 角色：Host(LLM 应用) ↔ Client(连接器) ↔ Server(本文件，提供能力)。本文件是 Server。
    - 生命周期：client 发 initialize（带 protocolVersion/capabilities 协商）→
                server 回 result → client 发 notifications/initialized（通知，无 id 无回）→
                之后可 tools/list 列工具、tools/call 调工具。
    - 三大原语：Resources(被动数据)/Prompts(模板)/Tools(可执行)。本演示只实现【Tools】。
    - 工具结果：{"content":[{"type":"text","text":...}], "isError":bool}。

【两种运行模式】
    uv run 05-Agent/05-mcp_server.py            # 默认：启动内置 mock client，自验并打印完整往返
    uv run 05-Agent/05-mcp_server.py --serve    # 当真正的 MCP server 跑（读 stdin / 写 stdout）

【参考 / 核对来源】
    - 协议规范（transports / lifecycle / tools）：https://modelcontextprotocol.io/specification
    - 原始 stdio 通信讲解：https://foojay.io/today/understanding-mcp-through-raw-stdio-communication/
    protocolVersion 取 2025-06-18（另有 2024-11-05 / 2025-03-26 等）。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "mini-mcp"
SERVER_VERSION = "0.1.0"


# ============================================================================
# 一、Server 暴露的两个工具（Tools 原语）—— 每个工具 = 可执行函数 + inputSchema
# ============================================================================
def calculator(expression: str) -> str:
    """计算四则运算表达式（仅允许数字与运算符，防注入）。"""
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return f"非法表达式：{expression}"
    try:
        return f"{expression} = {eval(expression, {'__builtins__': {}}, {})}"
    except ZeroDivisionError:
        return "错误：除以零"
    except Exception as e:  # noqa: BLE001
        return f"计算失败：{e}"


_ORDERS = {"A001": "订单 A001：2 台 iPhone，单价 799 美元",
           "A002": "订单 A002：3 台 AirPods，单价 199 美元"}


def get_order(order_id: str) -> str:
    """按订单号查业务订单（mock）。"""
    return _ORDERS.get(order_id, f"未找到订单：{order_id}")


# 工具注册表。注意 MCP 里工具 schema 字段叫【inputSchema】（不是 OpenAI 的 parameters）。
TOOLS: dict[str, dict[str, Any]] = {
    "calculator": {
        "callable": calculator,
        "schema": {
            "name": "calculator",
            "description": "计算四则运算表达式，如 '2 * 799'",
            "inputSchema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "get_order": {
        "callable": get_order,
        "schema": {
            "name": "get_order",
            "description": "按订单号查询订单信息",
            "inputSchema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
}


# ============================================================================
# 二、JSON-RPC 分发：把一条请求消息变成一条响应（通知则返回 None）
# ============================================================================
def dispatch(msg: dict) -> dict | None:
    """处理一条 JSON-RPC 消息，返回响应 dict；若是通知（无 id）返回 None。"""
    method = msg.get("method")
    params = msg.get("params") or {}
    req_id = msg.get("id")  # 通知没有 id

    def ok(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    # --- 生命周期：initialize ---
    if method == "initialize":
        # 真实 server 会就 protocolVersion 做协商；这里回 server 支持的版本。
        return ok({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": True}},  # 本 server 只提供 Tools 原语
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    # --- 生命周期：initialized 通知（无 id，无需回复）---
    if method == "notifications/initialized":
        return None

    # --- Tools 原语：列出工具 ---
    if method == "tools/list":
        return ok({"tools": [t["schema"] for t in TOOLS.values()]})

    # --- Tools 原语：调用工具 ---
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name not in TOOLS:
            return err(-32602, f"Unknown tool: {name}")  # -32602 = Invalid params
        try:
            out = TOOLS[name]["callable"](**args)
            return ok({"content": [{"type": "text", "text": str(out)}], "isError": False})
        except Exception as e:  # noqa: BLE001
            # 工具执行异常：按 MCP 约定返回 isError=True 的结果（而非 JSON-RPC error）
            return ok({"content": [{"type": "text", "text": f"工具执行出错: {e}"}], "isError": True})

    # 未知方法
    return err(-32601, f"Method not found: {method}")  # -32601 = Method not found


# ============================================================================
# 三、Server 模式：循环读 stdin 一行 = 一条 JSON-RPC，处理后写回 stdout 一行
#    约束：stdout 只能出现 JSON-RPC（否则 client 解析乱套）；日志一律走 stderr。
# ============================================================================
def serve() -> None:
    def log(*a):  # 日志走 stderr，绝不污染 stdout
        print(*a, file=sys.stderr, flush=True)

    log(f"[{SERVER_NAME}] server 启动，逐行读取 JSON-RPC（stdio，按行分帧）...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"[{SERVER_NAME}] 非法 JSON，跳过：{e}")
            continue
        log(f"[{SERVER_NAME}] ← method={msg.get('method')} id={msg.get('id')}")
        resp = dispatch(msg)
        if resp is not None:  # 请求回响应；通知不回
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            log(f"[{SERVER_NAME}] → 已回复 id={resp.get('id')}")
    log(f"[{SERVER_NAME}] stdin 结束，server 退出。")


# ============================================================================
# 四、Mock Client 模式：把 server 当子进程拉起，按真实握手顺序驱动并打印往返
# ============================================================================
def _send(proc: subprocess.Popen, msg: dict) -> None:
    proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    return json.loads(line)


def run_as_client() -> None:
    print("=" * 72)
    print(" 最小 MCP Server —— mock client 自验（打印完整 JSON-RPC 往返）")
    print(f" 协议版本: {PROTOCOL_VERSION}   server: {SERVER_NAME}/{SERVER_VERSION}")
    print("=" * 72)

    # 以 --serve 把本文件当 server 子进程拉起；stderr 收集 server 日志
    proc = subprocess.Popen(
        [sys.executable, __file__, "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    def rpc(label: str, msg: dict, expect_reply: bool = True) -> None:
        print(f"\n【{label}】  CLIENT → SERVER:")
        print("   " + json.dumps(msg, ensure_ascii=False))
        _send(proc, msg)
        if expect_reply:
            resp = _recv(proc)
            print(f"  SERVER → CLIENT:")
            print("   " + json.dumps(resp, ensure_ascii=False))

    # ① initialize（请求 → 响应）：能力协商
    rpc("① initialize 握手", {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                   "clientInfo": {"name": "mock-client", "version": "1.0.0"}},
    })

    # ② initialized（通知，无 id → 无响应）
    rpc("② initialized 通知（无 id，不回）", {
        "jsonrpc": "2.0", "method": "notifications/initialized",
    }, expect_reply=False)

    # ③ tools/list（请求 → 响应）：列出 server 的工具
    rpc("③ tools/list 列工具", {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    # ④ tools/call（请求 → 响应）：调 calculator
    rpc("④ tools/call 调 calculator", {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "calculator", "arguments": {"expression": "2 * 799 * 7.2"}},
    })

    # ⑤ tools/call：调 get_order
    rpc("⑤ tools/call 调 get_order", {
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "get_order", "arguments": {"order_id": "A001"}},
    })

    # ⑥ tools/call：调一个不存在的工具 → 看错误响应（-32602）
    rpc("⑥ tools/call 调不存在的工具（→ JSON-RPC error）", {
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "no_such_tool", "arguments": {}},
    })

    # 收尾：关闭 stdin 让 server 退出，打印 server 的 stderr 日志
    proc.stdin.close()
    proc.wait()
    server_log = proc.stderr.read().strip()
    print("\n" + "=" * 72)
    print(" 完整往返跑通 ✓  （这就是 Claude/ChatGPT 作为 Host 连 MCP server 时，")
    print(" 底层走的 JSON-RPC；想直连它们，把本文件 --serve 注册成它们的 MCP server 即可）")
    print("=" * 72)
    print("\n[server 端 stderr 日志]")
    print(server_log)


# ============================================================================
# main：默认跑 mock client 自验；--serve 当真正的 server
# ============================================================================
if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        run_as_client()
