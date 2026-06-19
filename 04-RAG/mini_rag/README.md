# Mini RAG 实战项目

> `04-RAG/mini_rag/` —— 把 `knowledge/RAG/` 里的理论变成一条可运行、可单步调试的 RAG 流水线。
> 两个版本：**纯 NumPy 自实现版**（零外部依赖，直接跑通，理解原理）+ **真实版**（chromadb + ollama）。

---

## 文件清单

| 文件 | 作用 | 是否能直接跑 |
|---|---|---|
| `main.py` | **纯 NumPy 自实现版**：加载→分块→哈希词袋嵌入→numpy 余弦检索→拼 prompt→toy 生成 | ✅ `uv run` 直接跑通 |
| `main_advanced.py` | **进阶版（Rerank+HyDE，纯 NumPy）**：在基线六步上加 toy cross-encoder 重排 + 规则版 HyDE，演示 before/after | ✅ `uv run` 直接跑通 |
| `main_real.py` | **真实版**：同样的六步，换 chromadb + sentence-transformers/ollama 嵌入 + ollama 生成；并含真实 rerank(bge-reranker) + HyDE(ollama 生成假设答案) | 需手动装依赖 + 起 ollama |
| `sample.txt` | 示例知识文档（内容是关于 RAG 本身的常识，方便自验召回） | — |

---

## 两个版本的区别（一一对应）

| 步骤 | main.py（toy） | main_real.py（真实） |
|---|---|---|
| 加载文档 | `load_text`（标准库） | 同（不变） |
| 分块 | `split_by_sentences`（正则按句切 + overlap） | 同（也可换 LlamaIndex `SentenceSplitter`） |
| 嵌入 | `embed`：字符/二字 bigram + 哈希词袋 + L2 归一 | `ollama.embeddings(model=bge-m3)` |
| 向量存储 | `NumpyVectorStore`：numpy 矩阵 + 暴力余弦 | `chromadb` PersistentClient（`hnsw:space=cosine`） |
| 检索 top_k | `store.query`（归一化向量点积 = 余弦） | `collection.query(n_results=k)` |
| 拼 Prompt | `build_prompt`（含防幻觉约束模板） | 同（不变） |
| 生成 | `generate_dummy`：抽取 + 相似度阈值 | `ollama.generate(stream=True)` 真实大模型 |

> 把"原理骨架"与"具体组件"解耦，是这两个文件并排存在的意义：换组件不动骨架。

---

## 怎么运行

### 版本一：纯 NumPy（推荐先跑这个）

```bash
cd /Users/sunhuanyu/ai-llm-learning
uv run 04-RAG/mini_rag/main.py
```

依赖只有 `numpy`（已在仓库根 `pyproject.toml` 的 `dependencies` 里），无需任何外部服务。
运行后会打印：分块结果 → 嵌入入库 → 4 个示范问题的检索命中 + Prompt + 回答。
其中 "今天晚饭吃什么？" 触发防幻觉回复（知识库里没有）。

### 版本一·进阶：Rerank + HyDE（纯 NumPy，原理版）

```bash
cd /Users/sunhuanyu/ai-llm-learning
uv run 04-RAG/mini_rag/main_advanced.py
```

复用 `main.py` 的基线六步，只加两招进阶检索（对应 `knowledge/RAG` 第8章）：
- **Rerank**：`rerank()` 用 toy cross-encoder（query/doc 词覆盖联合打分）对双塔召回的 top8 二次精排取 top3 —— 演示「双塔快但粗 → cross-encoder 联合打分更准」的机制。
- **HyDE**：`hyde_expand()` 把提问句去提问腔、改写成「答案腔」假设文档再去嵌入检索 —— 演示「answer↔answer 比 query↔answer 更近」（实测与目标块相似度 0.364→0.549）。

三个演示分别打印：① Rerank 前/后 top3 变化；② HyDE vs 原始 query 召回差异 + 量化相似度；③ HyDE+Rerank 完整流水线接生成。
> ⚠ toy 的双塔与 cross-encoder 都基于词重叠，重排幅度有限；**真实语义级** rerank/hyde 见 `main_real.py`。

### 版本二：真实版（chromadb + ollama）

```bash
# 1) 装依赖（不自动安装）
uv pip install chromadb sentence-transformers ollama

# 2) 起 ollama 并拉模型
ollama pull bge-m3          # 嵌入模型
ollama pull qwen2.5:7b      # 生成模型（任意 instruct 模型均可）
ollama serve                # 默认 http://localhost:11434

# 3) 跑
uv run 04-RAG/mini_rag/main_real.py
```

模型名在 `main_real.py` 顶部 `EMBED_MODEL` / `LLM_MODEL` 改。

> 想用 sentence-transformers 本地跑嵌入（不依赖 ollama 嵌入）：把 `embed()` 换成
> `SentenceTransformer("BAAI/bge-small-zh-v1.5").encode(text, normalize_embeddings=True)`，
> 其余六步完全不变。

---

## 依赖说明

- **纯 NumPy 版**：`numpy`（已在仓库 deps）。无网络、无模型、无服务。
- **真实版**（手动装）：
  - `chromadb` — 向量库（持久化、HNSW 近似检索）
  - `ollama` — 调本地 ollama 的 embeddings/generate 接口
  - `sentence-transformers` — 可选，替换 ollama 嵌入为本地权重嵌入

---

## 对应 knowledge/RAG 哪几章

| 项目代码 | 对应章节 | 对应知识点 |
|---|---|---|
| `build_index` 整体六步 | [第3章 3.1.1](../../knowledge/RAG/基础篇/第3章-初识RAG应用开发.md) | 原生代码 RAG 六步流程 |
| `load_text` | [第3章 3.1.1(1)](../../knowledge/RAG/基础篇/第3章-初识RAG应用开发.md) / [第5章](../../knowledge/RAG/基础篇/第5章-数据加载与分割.md) | 数据加载、Document |
| `split_by_sentences` + overlap | [第5章 5.3](../../knowledge/RAG/基础篇/第5章-数据加载与分割.md) | SentenceSplitter、chunk_size/overlap 是上限、中文用正则 |
| `embed`（哈希词袋 + L2） | [第6章 6.1](../../knowledge/RAG/基础篇/第6章-数据嵌入与索引.md) | 向量、嵌入、L2 归一化 |
| `NumpyVectorStore` / chroma collection | [第6章 6.2](../../knowledge/RAG/基础篇/第6章-数据嵌入与索引.md) | SimpleVectorStore、Chroma、`hnsw:space=cosine` |
| `retrieve` | [第7章 7.1](../../knowledge/RAG/基础篇/第7章-检索响应生成与RAG引擎.md) | 检索器、top_k |
| `build_prompt` + `generate` | [第7章 7.2](../../knowledge/RAG/基础篇/第7章-检索响应生成与RAG引擎.md) | 响应生成器、防幻觉 prompt |
| Rerank / HyDE（✅ 已实现） | [第8章 8.1/8.2](../../knowledge/RAG/高级篇/第8章-RAG引擎高级开发.md) | cross-encoder 重排、假设文档嵌入；见 `main_advanced.py`（原理）/ `main_real.py`（真实） |

全流程总结、优化点、面试要点见上一级目录 [`../笔记.md`](../笔记.md)。
