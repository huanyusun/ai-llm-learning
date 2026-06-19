# 项目#1 设计：RAG 进阶（Rerank + HyDE）

> 日期：2026-06-19 ｜ 状态：已批准 ｜ 对应：`04-RAG/mini_rag/` + `knowledge/RAG/高级篇/第8章`
> 原则：项目优先，在项目中把知识点学清楚。spec 从简（按用户要求），不另起重型多步 plan。

## 1. 目标与「学清楚」口径
做完能用白话讲清两点：
- **Rerank**：向量检索（bi-encoder 双塔，各自编码再点积）**快但粗** → 需 cross-encoder（query+doc 联合编码重打分）**精排**。典型 pipeline = 向量多召回(top20) → rerank → top3-5。
- **HyDE**：用户 query 与文档有**语义鸿沟** → 嵌入「假设性答案」比嵌入原始 query 召回更准（answer↔answer 在向量空间更近）。

## 2. 文件布局（隔离新概念，不污染基线）
- `main.py`：**不动**，保留干净六步基线作对照。
- **新增** `main_advanced.py`（numpy 原理版）：`from main import *` 复用基线，加 `rerank()` / `hyde_expand()` / `retrieve_advanced()`，在 `sample.txt` 上演示 before/after。
- **扩展** `main_real.py`（真实版）：真实 rerank（`bge-reranker` cross-encoder）+ 真实 HyDE（ollama 生成假设文档再嵌入检索）。
- 更新 `README.md`（文件清单）+ `笔记.md`（动手清单打勾、补机制讲解）。

理由：笔记原写「在 main.py 上加」，但那会污染干净基线、违反「每文件一核心概念」。隔离成 `main_advanced.py` 利于单步调试与面试讲解。

## 3. numpy 版机制演示
- **Rerank**：`rerank(query, candidates, top_n)` 用 toy cross-encoder = query/doc **token 命中 + 覆盖率**联合打分（必须同时看 query 和 doc，纯双塔做不到）。构造双塔易排错的 query，打印 rerank 前/后 top3 → 坐实「双塔粗、精排纠」。
- **HyDE**：`hyde_expand(query)` 无 LLM，规则把 query 关键词拼成「答案样式假设文档」→ 用它 embed 去检索，对比「直接 embed query」召回。构造 query 与文档措辞不同（有鸿沟）的场景，看 HyDE 是否更准。
- 关键正确性：HyDE 只改「检索用的嵌入文本」，**rerank 仍用原始 query** 打分（rerank 看的是 query↔doc 真实相关性）。

## 4. 真实版（main_real.py 扩展）
- rerank：`BAAI/bge-reranker-base`（`sentence-transformers.CrossEncoder`）对候选重打分。
- HyDE：`ollama.generate` 生成假设答案 → 用假设答案 `ollama.embeddings` 检索（**不替换原 query，仅用于嵌入检索**——呼应笔记陷阱）。
- 手动装依赖、手动跑，沿用「不自动安装」规则。

## 5. 验收标准
- [ ] `uv run 04-RAG/mini_rag/main_advanced.py` 零依赖跑通，打印：① rerank 前后 top3 顺序变化；② HyDE vs 原始 query 召回差异。
- [ ] `main_real.py` 补真实 rerank+HyDE 代码 + README 手动运行步骤（代码正确、依赖说明清楚即可，不强求本地跑通）。
- [ ] `笔记.md` 动手清单三项打勾 + before/after 机制讲解。

## 6. 范围外（YAGNI）
- 不做混合检索、不做查询多步分解（SubQuestion/MultiStep）、不做 RAGAS 评估——留给后续项目。
