# 智能知识助手 Intelligent Knowledge Assistant

> 综合实战项目：把 **RAG 检索 + ReAct Agent + Prompt 工程 + 多轮对话记忆** 整合成一个完整应用。
> 纯 Python / numpy 模拟版，**零外部依赖、直接 `uv run` 跑通**，骨架可平移到真实模型。

---

## 一、它是什么

一个能"**多轮对话 + 按需检索 + 调用工具推理**"的知识助手。它不是把前面几个模块的代码简单堆在一起，而是把它们组合成一条**端到端的推理链路**：

```
用户提问
   │
   ▼
[意图识别]  事实类？数学类？  ──────────────┐
   │                                       │
   ▼                                       ▼
[ReAct 循环]                          [ReAct 循环]
 Thought → Action: 检索[...]          Thought → Action: 计算[...]
   │ → Observation (RAG top_k)           │ → Observation (42.0)
   ▼                                       ▼
[综合 + 防幻觉 + 溯源标注]              [回传结果]
   │                                       │
   └───────────────┬───────────────────────┘
                   ▼
            [写入对话记忆] → 支持下一轮追问（指代消解）
                   ▼
              自然语言回答
```

四个关键能力，对应四个模块：

| 能力 | 来自哪个模块 | 在本项目的体现 |
|------|-------------|----------------|
| 知识检索 | `04-RAG/mini_rag` | `embed` + `NumpyVectorStore` + 余弦相似度 top_k |
| 工具调用推理 | `05-Agent/02-react_agent` | `react_loop`：Thought→Action→Observation→Finish |
| 结构化提示 | `03-Prompt工程` | `SYSTEM_PROMPT`（角色+工具说明+流程约束）、防幻觉、溯源 |
| 多轮记忆 | 整合性新增 | `Memory` 滑动窗口 + 指代消解（"它/这个"→ 上轮主题） |

---

## 二、架构图（文字版）

```
┌─────────────────────────────────────────────────────────────────┐
│                    KnowledgeAssistant（助手门面）                 │
│                                                                 │
│   chat(query) ─► resolve_coreference ─► react_loop ─► Memory.add │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 调用
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │  RAG 模块    │   │ Agent 模块   │   │  记忆模块    │
    │              │   │              │   │              │
    │ KNOWLEDGE_   │   │ react_loop() │   │  Memory      │
    │  DOCS        │   │  ↕ tools 注册│   │  - turns[]   │
    │ split        │   │    检索/计算/ │   │  - resolve_  │
    │ embed(哈希)   │   │     完成     │   │    coref     │
    │ NumpyVector  │   │  ↕ mock_llm  │   │  - render    │
    │  Store(余弦) │   │    _step     │   │              │
    └──────┬───────┘   └──────┬───────┘   └──────────────┘
           │                  │
           ▼                  ▼
   ┌──────────────────────────────────┐
   │     Prompt 模块（贯穿全局）       │
   │  SYSTEM_PROMPT / 防幻觉 / 溯源标注 │
   └──────────────────────────────────┘
           │
           ▼
   ┌──────────────────────────────────┐
   │  三个可替换点（接真实模型时换这里）│
   │   • mock_llm_step  → OpenAI/Ollama│
   │   • embed          → bge-small-zh │
   │   • NumpyVectorStore → chromadb   │
   └──────────────────────────────────┘
```

**关键设计**：RAG 不再是一个独立流水线，而是**被降级成 Agent 的一个工具**（`检索[...]`）。Agent 决定"要不要查、查什么、查完够不够"。这是从"RAG 应用"升级到"Agent 应用"的核心转变。

---

## 三、数据流（一次对话的完整路径）

以 `RAG 是什么？` 为例：

1. **入口** `KnowledgeAssistant.chat("RAG 是什么？")`
2. **指代消解** `Memory.resolve_coreference`：首轮无历史，原样返回。
3. **意图识别** `_classify`：无数字/运算符 → `intent = "retrieve"`。
4. **ReAct 第 1 步**
   - `mock_llm_step` 产出：`Thought 1: ...需要检索` / `Action 1: 检索[RAG 是什么？]`
   - 执行 `检索` 工具 → `store.query(embed("RAG 是什么？"), top_k=3)`
   - `Observation 1`: 3 个 chunk + 相似度分数（最高 0.319，> 阈值 0.20，标记 relevant）
5. **ReAct 第 2 步**
   - `Thought 2: 已检索到候选内容...` / `Action 2: 完成[...]`
   - 触发 `compose_final_answer`：从轨迹里挑 `is_relevant=True` 的检索结果，拼接 + 标注来源。
6. **写记忆** `Memory.add("RAG 是什么？", 答案)`。
7. **返回** 自然语言答案给用户。

追问 `它和这个 ReAct 有什么区别？` 时：
- 第 2 步 `resolve_coreference` 把"它/这个"替换成上轮主题 `RAG` → 检索 query 变成含 RAG 与 ReAct 的复合查询；
- 记忆窗口里带着上一轮 Q&A 一起进 Prompt，模型"看得见"上下文。

超纲问题 `今天晚饭吃什么？` 时：
- 检索最高分 0.175 < 阈值 0.20 → `is_relevant=False`；
- `compose_final_answer` 收集不到相关片段 → 输出"知识库里没有…我暂时无法回答"（**防幻觉**）。

---

## 四、整合了 01-06 哪些部分

| 模块 | 用到的知识点 | 本项目代码位置 |
|------|------------|----------------|
| **00-数学基础** | 向量、点积、余弦相似度、L2 归一化、argpartition top_k | `embed` / `NumpyVectorStore.query` |
| **01-Transformer** | （概念层）"模型"是什么、为什么需要 embedding | `mock_llm_step` 占位 + 文末真实版说明 |
| **02-LLM基础** | 采样/幻觉/温度（防幻觉阈值） | `RELEVANCE_THRESHOLD`、`compose_final_answer` |
| **03-Prompt工程** | 角色设定、结构化输出格式、思维链（CoT 即 ReAct 的 Thought） | `SYSTEM_PROMPT`、`mock_llm_step` 的 Thought/Action 格式 |
| **04-RAG** | 六步流水线（加载/分块/嵌入/存储/检索/生成） | `KNOWLEDGE_DOCS`→`build_knowledge_base`→`检索`工具 |
| **05-Agent** | ReAct 循环、动作空间、工具注册表 | `react_loop`、`make_tools`、`parse_action` |
| **06-微调与部署** | （出口）如何把模拟版接真实模型、部署成服务 | 文末「真实版说明」+ 本 README 第五节 |

> 可以说这是一个"学完 01–06 后能做出的最小完整产品"——它不依赖任何前面模块没有讲过的东西。

---

## 五、怎么跑

```bash
uv run --directory /Users/sunhuanyu/ai-llm-learning python "projects/智能知识助手/main.py"
```

依赖只有 `numpy`（已在仓库 `pyproject.toml` 的 `dependencies` 里），无需联网、无需下载模型权重。

跑完会依次输出 4 轮对话，每轮完整打印：
`用户提问 → 意图识别 → 记忆 → Thought → Action → Observation → 最终回答`。

自定义知识库：把 `KNOWLEDGE_DOCS` 换成你自己的文档列表，或改 `KnowledgeAssistant(docs=...)` 传参即可。

---

## 六、如何接真实模型（三个替换点）

骨架完全不动，只换三个函数/类：

### 1. LLM：`mock_llm_step` → 真实模型调用
```python
from openai import OpenAI
client = OpenAI()

def real_llm_step(query, history_text, step, intent):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + tools_description},
        {"role": "user",   "content": f"{history_text}\n问题: {query}"},
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    # 解析返回里的 "Thought N: ... Action N: ..."，或改用 function calling
    return parse(resp.choices[0].message.content)
```

### 2. Embedding：`embed`（哈希词袋）→ sentence-transformers
```python
from sentence_transformers import SentenceTransformer
_m = SentenceTransformer("BAAI/bge-small-zh")
def embed(text):
    return _m.encode(text, normalize_embeddings=True)  # 已归一化，点积即余弦
```

### 3. 向量库：`NumpyVectorStore` → chromadb / faiss
```python
import chromadb
col = chromadb.PersistentClient(path="./chroma").get_or_create_collection(
    "kb", metadata={"hnsw:space": "cosine"})
col.add(ids=..., embeddings=..., documents=..., metadatas=...)
col.query(query_embeddings=..., n_results=TOP_K)
```

> 替换后，`react_loop` / `Memory` / `SYSTEM_PROMPT` / `compose_final_answer` 一行都不用改——这正是"骨架与实现分离"带来的好处，也是面试时可以重点讲的设计点。

---

## 七、目录文件

```
projects/智能知识助手/
├── main.py        # 模拟版主程序（RAG+Agent+记忆，纯 numpy，可直接跑）
└── README.md      # 本文件

projects/笔记.md   # 综合项目总结 + 面试亮点（在 projects/ 根目录）
```

---

## 八、为什么这个项目"能体现整合性"

- **RAG 被降级为工具**：不是"先 RAG 再答"，而是"Agent 自己决定要不要查"——这是从 RAG 应用走向 Agent 应用的标志。
- **多工具并存**：`检索` 与 `计算` 是两类完全不同的工具（一个查语义、一个算精确），但通过统一的 `ToolResult` 接口被同一个 ReAct 循环调度。
- **记忆闭环**：Agent 的输出写回 Memory，下一轮的指代消解和 Prompt 上下文都依赖它。
- **防幻觉内建**：阈值 + 来源标注 + 不相关时诚实拒答，贯穿检索层和回答生成层。
