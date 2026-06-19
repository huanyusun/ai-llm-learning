# 项目#2 设计：Agent 进阶（Multi-Agent + 最小 MCP server）

> 日期：2026-06-19 ｜ 状态：已批准 ｜ 对应：`05-Agent/` + `knowledge/Agent/系统知识.md`
> 原则：项目优先、在项目中学清楚；spec 从简，不另起重型多步 plan。

## 1. 目标
- **多 Agent 协作**：把"单 Agent 调多工具"（03）升级为"Supervisor 拆解 → 专家 Worker 分工 → 聚合"。
- **MCP server**：从零用标准库写一个真能通信的最小 MCP server（JSON-RPC over stdio），把协议跑出来。

## 2. 交付物（两个文件，零依赖）
- `04-多Agent协作.py`：Supervisor + Researcher/Calculator/Writer，规则路由 mock；演示"查事实→算→总结"全链路；文末附真实版（每专家一个 system prompt 的 LLM 调用）。
- `05-mcp_server.py`：从零 JSON-RPC MCP server（`initialize`/`notifications/initialized`/`tools/list`/`tools/call`，工具用 `inputSchema`，结果 `content`+`isError`）；内置 mock client 子进程自验，打印完整往返。`--serve` 当真 server。

## 3. 关键决策（已拍板）
- 多 Agent：零依赖 mock（与 01-03 一致）。
- MCP：**从零 JSON-RPC**（不用 SDK），贴合"零依赖、看清原理、不黑盒"哲学；真实可连版作为附录。

## 4. MCP 协议要点（核对自规范）
- 分帧：按行分隔 JSON-RPC 2.0，不用 Content-Length。
- 生命周期：initialize（协商 protocolVersion/capabilities）→ notifications/initialized（无回）→ tools/list、tools/call。
- 工具 schema 字段 `inputSchema`；结果 `{"content":[...],"isError":bool}`；未知工具 error -32602。
- stdout 只放 JSON-RPC，日志走 stderr。

## 5. 验收（均实测通过）
- [x] `uv run 05-Agent/04-多Agent协作.py` 跑通，打印拆解→委派→聚合。
- [x] `uv run 05-Agent/05-mcp_server.py` 跑通，打印 6 步 JSON-RPC 往返（含 error）。
- [x] `05-Agent/笔记.md` 补 §8 实战小节。

## 6. 范围外（YAGNI）
- 不做 Resources / Prompts / Sampling 原语（只实现 Tools）。
- 不接真实 LLM / 不接 Claude Desktop（作为附录标注）。
