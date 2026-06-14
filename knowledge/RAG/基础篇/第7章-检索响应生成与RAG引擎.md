# 第7章 检索、响应生成与 RAG 引擎

> 本章节覆盖 RAG 的"使用阶段"：检索器（Retriever）及其检索模式、响应生成器（Synthesizer）的多种响应生成模式、查询引擎（Query Engine）、对话引擎（Chat Engine）的多种对话模式、结构化输出。完成本章即具备构造端到端 RAG 引擎的能力。所有 API 以 LlamaIndex 框架为主。

---

## 0. RAG 引擎的定义

RAG 的"G"（Generation）才是最终目的。在完成数据准备与索引后，进入**检索 + 响应生成**阶段，构造起核心的 RAG 引擎。

> **RAG 引擎** = 能接受用户输入问题、借助自身检索与响应生成能力给出最终答案的、基于大模型的软件装置。

两种基础类型：
- **查询引擎（Query Engine）**：最直接形式，自然语言输入 → 检索/排序/生成 → 一次性输出答案
- **对话引擎（Chat Engine）**：连续的、有状态的、有上下文的多轮交互，需上下文记忆能力

构造 RAG 引擎前，需先理解检索器与响应生成器两个基础组件。

---

## 7.1 检索器（Retriever）

检索器是大模型响应生成的基础：根据用户输入检索最相关的知识上下文，以 **Node 列表**形式返回。无检索器即无增强生成。检索器通常基于各种索引构造，索引类型不同 → 检索器不同。

### 7.1.1 快速构造检索器

```python
retriever = vector_index.as_retriever(similarity_top_k=1)
nodes = retriever.retrieve('文心一言的应用场景')
pprint.pprint(nodes)
```

- 最直接方法：`index.as_retriever(...)`
- 常见配置：`retriever_mode`（检索模式）、`similarity_top_k`（检索语义最相似的数量）
- 核心方法 `retrieve(query_str)`：输入检索条件，返回携带相关性评分（Score）的 Node 列表
- 返回类型是 `NodeWithScore`（Node 的增强类型，附带 score）

查询引擎构造时会**隐式自动构造检索器**：`query_engine = index.as_query_engine()` 内部先 `as_retriever`。

也可用底层 API 直接构造（与 `as_retriever` 等价）：

```python
retriever = VectorIndexRetriever(index=vector_index, ...)            # 向量索引检索器
retriever = SummaryIndexLLMRetriever(index=summary_index, choice_batch_size=5)  # 摘要索引检索器
```

建议优先用 `as_retriever`，避免记住具体检索器类型。

### 7.1.2 检索模式与检索参数

不同索引类型有不同检索模式和默认模式，在检索流程、模型/工具、效果、性能上各有特点。

#### 1. 如何指定检索模式

```python
# 用 as_retriever
retriever = treeindex.as_retriever(retriever_mode="root")

# 直接构造具体类型
from llama_index.core.indices.tree.all_leaf_retriever import TreeAllLeafRetriever
retriever = TreeRootRetriever(index=treeindex)   # root 模式对应类型
```

#### 2. 不同索引支持的检索模式

**1）向量存储索引（VectorStoreIndex）**

只支持一种模式：**向量语义相似度检索**（`retriever_mode` 参数被忽略），对应 `VectorIndexRetriever`。

| 参数 | 用途 |
|------|------|
| `similarity_top_k` | 检索相关性最高的 Node 数量（默认 2） |
| `filters` | 元数据过滤器，向量检索前先做元数据过滤 |
| `vector_store_query_mode` | 向量存储查询模式（需向量库支持） |

部分特性依赖底层向量库。

**2）文档摘要索引（DocumentSummaryIndex）**

对 Node 在 Document 级别生成摘要 Node 并嵌入，检索时先查摘要 Node 再溯源基础 Node。两种模式：

- **llm**：用大模型判断摘要与问题的相关性，取最相关摘要 Node → 输出对应基础 Node。`DocumentSummaryIndexLLMRetriever`
  - `choice_select_prompt`：大模型判断相关性的 Prompt 模板
  - `choice_top_k`：选择相关摘要 Node 的数量（注意不是返回的 Node 数）
- **embedding**：用嵌入模型+向量相似度判断。`DocumentSummaryIndexEmbeddingRetriever`
  - `similarity_top_k`：选择相关摘要 Node 的数量（注意不是返回的 Node 数）

**3）树索引（TreeIndex）**

把输入 Node 作为叶子，自底向上对叶子生成带摘要的父 Node 形成索引树。4 种检索模式：

- **select_leaf**：从根逐层检索相关叶子 Node，借大模型判断相关性。`TreeSelectLeafRetriever`
  - `query_template`：单选叶子的 Prompt 模板
  - `query_template_multiple`：多选叶子的 Prompt 模板
  - `child_branch_factor`：每层选叶子时单选还是多选
- **select_leaf_embedding**：同上但用向量相似度判断。`TreeSelectLeafEmbeddingRetriever`（子类）
- **all_leaf**：返回所有叶子 Node（不用索引树）。`TreeAllLeafRetriever`
- **root**：返回所有根 Node。`TreeRootRetriever`

**4）关键词表索引（KeywordTableIndex）**

从 Node 解析关键词并建映射，检索时按输入问题关键词查 Node。3 种模式：

- **default**：大模型解析输入问题关键词。`KeywordTableGPTRetriever`
  - `query_keyword_extract_template`：大模型解析关键词的 Prompt
  - `max_keywords_per_query`：单次最大关键词数量
- **simple**：正则解析关键词。`KeywordTableSimpleRetriever`（不适合中文）
- **rake**：RAKE 库解析关键词。`KeywordTableRAKERetriever`

**5）对象索引（ObjectIndex）**

特殊：依赖其他索引（本质是对象序列化后通过其他索引实现），检索模式取决于底层索引（如用 VectorStoreIndex 则为向量检索）。

**6）知识图谱索引（PropertyGraphIndex）**

底层是图结构（Node + Property + Relationship + 辅助信息如嵌入向量）。3 种检索模式：

- **Text-to-Cypher**：大模型把自然语言转图查询语言（如 Neo4j Cypher）后检索
- **Vector Search**：需图库支持向量检索，构造时给 Node/关系生成向量，检索时向量召回
- **Keywords Search**：大模型从输入提取关键词，借图库能力检索相关 Node/关系

通过子检索器组合实现，构造时（`as_retriever` 或直接构造 `PGRetriever`）用 `sub_retrievers` 参数指定。

| 参数 | 用途 |
|------|------|
| `sub_retrievers` | 检索时使用的多个子检索器列表 |

```python
# 两个子检索器
synonym_retriever = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    include_text=False,
    output_parsing_fn=parse_fn,
    max_keywords=10,
    synonym_prompt=prompt,
    path_depth=1,
)

vector_retriever = VectorContextRetriever(
    index.property_graph_store,
    include_text=False,
    similarity_top_k=2,
    path_depth=1,
)

# 组合成知识图谱检索器
retriever = PGRetriever(sub_retrievers=[synonym_retriever, vector_retriever])

# 也可在查询引擎中指定
query_engine = index.as_query_engine(
    include_text=True, similarity_top_k=1,
    sub_retrievers=[synonym_retriever, vector_retriever]
)
```

### 7.1.3 初步认识递归检索

检索器在 retrieve 出多个 Node 后，**不管什么索引都会有一步递归检索操作**：`_handle_recursive_retrieval`。这是对 `IndexNode`（特殊 Node 类型）的递归检索。

由于 IndexNode 保存了指向其他对象（Node/Retriever/QueryEngine）的引用，可进行"钻取式"二次检索：

| IndexNode 指向的对象 | 二次检索操作 |
|---------------------|--------------|
| 其他 Node | 直接返回指向的 Node（从摘要找源内容 Node） |
| 查询引擎 | 调查询引擎得响应，组装成 Node 返回 |
| 检索器 | 调检索器二次检索，返回 Node |

二次"检索"得到的 Node 会作为检索器最终结果返回给响应生成。

---

## 7.2 响应生成器（Synthesizer）

检索器召回 Node 列表后即具备响应生成条件。响应生成器组件 `Synthesizer` 不是简单地把问题+上下文组装后给大模型一次性生成，而是**有多种响应生成模式**——简单组装在很多场景下质量不高。

不同模式下使用上下文的方式、Prompt 模板、迭代流程都不同。Synthesizer 把这些差异抽象，通过统一接口交给上层 RAG 引擎。相关组件：输入问题、检索出的 Node 列表、Prompt 模板、大模型，以及控制参数（流式、输出格式等）。

### 7.2.1 构造响应生成器

**显式构造**（最常用 `get_response_synthesizer`）：

```python
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

# 直接调用测试
response = response_synthesizer.synthesize("你的输入问题", nodes=nodes)

# 在 RAG 引擎中使用
query_engine = vector_index.as_query_engine(response_synthesizer=response_synthesizer)
# 或直接构造查询引擎时指定
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)
```

**隐式构造**（构造 RAG 引擎时自动构造默认检索器+响应生成器）：

```python
query_engine = vector_index.as_query_engine(
    streaming=True,
    verbose=True,
    response_mode=ResponseMode.COMPACT
)
```

### 7.2.2 响应生成模式（response_mode）

通过 `response_mode` 参数切换，代表不同的使用上下文输出答案方式。

#### 1. refine 模式（迭代细化）

流程：
1. 用第一个 Node 的上下文 + 问题生成**初始答案**
2. 把此答案 + 问题 + 第二个 Node 的上下文组装成新 refine prompt，给大模型生成**细化答案**
3. 循环直到所有 Node 都细化过
4. 中途若某 Node 导致上下文窗口溢出，则分割该 Node 内容形成新 Node 加入队列

特点：N 个 Node → **至少 N 次大模型调用**。第 1 次和后续调用 Prompt 模板不同（后续要带前一次答案 `{existing_answer}`）。指令要求：基于新 Node 重写/补充已有答案，或保留不变（若新上下文无用）。

适合：需要非常详细答案的场景。缺点：耗时、token 代价大。

#### 2. compact 模式（默认）

先把多个 Node 的文本块**组合成更大的整合块**（充分利用上下文窗口），然后用 refine 模式基于整合块生成。本质是"先合并 + 再 refine"（compact 响应生成器继承自 refine 响应生成器）。

特点：通常**更少的大模型调用次数**（但不一定只有一次，整合后若溢出仍会用 refine 迭代）。缺点：上下文较长可能导致生成结果不如 refine 完整详细。

#### 3. tree_summarize 模式（树形总结，适合总结性问题）

流程（假设 N 个 Node）：
1. 合并 Node 以适应最大上下文窗口
2. 若合并后只有 1 个 Node → 直接用它调用大模型生成唯一答案，结束
3. 若仍有多个 Node → **并行**调用大模型生成多个答案
4. 把多个答案构造成新 Node，重复合并→查询过程，直到只剩一个答案

特点：不断合并、递归式总结所有 Node，直到唯一答案。**更适合总结性问题**。

```python
# 用长文档模拟检索出的多个 Node（手工构造的少量 Node 会被直接合并，看不出效果）
reader = SimpleDirectoryReader(input_files=["../../data/AI-survey-cn.pdf"])
docs = reader.load_data()
splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n")
nodes = splitter.get_nodes_from_documents(docs)
node_scores = [NodeWithScore(node=node, score=1.0) for node in nodes]

response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
response = response_synthesizer.synthesize(
    "请使用中文，文中介绍了 AI Agent 哪些方面的内容",
    nodes=node_scores
)
# 观察到多次大模型调用，最后一次的 user 消息携带的都是之前多次生成的答案
```

#### 4. 更多模式

**simple_summarize**：合并 Node 适应窗口，**截断多余内容**，一次大模型调用。优点快，缺点丢信息。注意：合并时**不是优先合并排名靠前的 Node 丢弃后面的**，而是计算后**截断每个 Node 后面的溢出内容**。

**accumulate**：对每个 Node 都调大模型生成答案，用分隔符组合后直接输出。适合需对每个 Node 单独生成、无需二次总结的场景。

**compact_accumulate**：合并 + accumulate。多个 Node 合并后调用，减少调用次数。

**no_text**：不产生真实大模型响应，仅用于获取检索出的 Node 列表信息。

**generation**：直接调大模型回答问题，**不携带任何上下文**（纯生成，无 RAG 增强）。

### 7.2.3 响应生成器的参数

| 参数 | 说明 |
|------|------|
| `llm` | 大模型，不设则从 Settings 取默认 |
| `***_template` | Prompt 模板 |
| `streaming` | 是否流式输出 |
| `output_cls` | 结构化输出类型（Pydantic） |
| `structured_answer_filtering` | 是否过滤不相关 Node 答案（refine/compact 模式，主要针对 GPT 系列） |

**4 种 Prompt 模板与适用模式**：

| 模板 | 适用模式 |
|------|---------|
| `text_qa_template` | 基本 QA 模板，用于 refine/compact/simple_summarize/accumulate/compact_accumulate |
| `refine_template` | refine、compact |
| `summary_template` | tree_summarize |
| `simple_template` | generation（不携带检索上下文） |

自定义 Prompt 模板示例（tree_summarize 模式 + 增加参数）：

```python
qa_prompt_tmpl = (
    "根据以下上下文信息：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "使用{language_name}回答以下问题\n "
    "问题: {query_str}\n"
    "答案: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    streaming=True,
    summary_template=qa_prompt
)

# 响应生成时传入 language_name 参数
streaming_response = response_synthesizer.synthesize(
    "介绍一下小麦手机的优点",
    nodes=nodes,
    language_name="法语"
)
```

**streaming**：从大模型输出第一个 token 就开始输出，不等全部完成。多调用模式中仅最后一次调用用流式。处理流：

```python
for text in streaming_response.response_gen:
    ...   # 自行处理
# 或 streaming_response.print_response_stream() 输出到控制台
```

**output_cls（结构化输出）**：

```python
class Phone(BaseModel):
    name: str
    description: str
    features: List[str]

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    summary_template=qa_prompt,
    output_cls=Phone
)
```

结果会借助大模型转换，试图输出符合类型定义的格式（存在失败可能性，取决于模型指令遵从能力）。

### 7.2.4 实现自定义响应生成器

继承 `BaseSynthesizer`，实现必要接口（`_get_prompts`、`_update_prompts`、`get_response`、`aget_response`）：

```python
class FunnySynthesizer(BaseSynthesizer):
    my_prompt_tmpl = (
        "根据以下上下文信息：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "使用中文且幽默风趣的风格回答以下问题\n "
        "问题: {query_str}\n"
        "答案: "
    )

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self._input_prompt = PromptTemplate(FunnySynthesizer.my_prompt_tmpl)

    def _get_prompts(self) -> PromptDictType:
        pass

    def _update_prompts(self, prompts: PromptDictType) -> None:
        pass

    def get_response(self, query_str, text_chunks, **response_kwargs):
        context_str = "\n\n".join(n for n in text_chunks)
        response = self._llm.predict(
            self._input_prompt,
            query_str=query_str,
            context_str=context_str,
            **response_kwargs,
        )
        return response

    async def aget_response(self, query_str, text_chunks, **response_kwargs):
        context_str = "\n\n".join(n for n in text_chunks)
        response = await self._llm.apredict(
            self._input_prompt, query_str=query_str, context_str=context_str,
            **response_kwargs,
        )
        return response

response_synthesizer = FunnySynthesizer(llm=llm)
```

---

## 7.3 RAG 引擎：查询引擎（Query Engine）

查询引擎通过自然语言查询/提问获得一次性响应，广泛应用于搜索、问答、数据查询分析。使用者可是人或其他系统。

### 7.3.1 构造内置查询引擎的两种方法

**1. 高层 API 快速构造**

```python
query_engine = vector_index.as_query_engine()
response = query_engine.query('客户在没有交定金之前要求出具房地产证原件，怎么办？')

# 流式
query_engine = vector_index.as_query_engine(streaming=True)
response = query_engine.query('...')
response.print_response_stream()
```

**2. 底层 API 组合构造（与 as_query_engine 等价）**

```python
# 构造检索器
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)

# 构造响应生成器
response_synthesizer = get_response_synthesizer(streaming=True)

# 组合
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)
```

底层 API 牺牲简洁性换取可配置性（如可替换为自定义响应生成器）。可用 Langfuse 等平台观察内部运行（输入大模型的 Prompt、检索出的 Node）来判断检索精确度与生成质量。

### 7.3.2 查询引擎内部结构与运行原理

`as_query_engine` 源码核心：

```python
def as_query_engine(self, llm=None, **kwargs) -> BaseQueryEngine:
    from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
    retriever = self.as_retriever(**kwargs)
    llm = resolve_llm(llm, ...) if llm else llm_from_settings_or_context(Settings, ...)
    return RetrieverQueryEngine.from_args(retriever, llm=llm, **kwargs)
```

可见查询引擎依赖两个核心组件：
- **retriever**：借助索引召回相关上下文（Node 列表）
- **llm**：用 Prompt 模板组装知识与原始问题，交给大模型生成

`RetrieverQueryEngine`（以向量存储索引为例）相关组件：
- **VectorIndexRetriever**：完成检索，输出多个相关 Node
- **Synthesizer**：借助大模型组装 Prompt，按响应生成模式生成结果
- **NodePostProcessor**：节点后处理器，检索完成后对 Node 列表补充处理（如重排序）

不用 `as_query_engine` 时，可用底层 API 分别构造这些组件组装。

### 7.3.3 自定义查询引擎

内置查询引擎不总满足需求（如结合微调本地模型实现 C-RAG、Self-RAG 等模块化 RAG 范式）。继承 `CustomQueryEngine` 实现 `custom_query` 接口，即可像内置引擎一样使用。

**基于响应生成器的自定义引擎**：

```python
class MyQueryEngine(CustomQueryEngine):
    response_synthesizer: BaseSynthesizer = Field(default=None, description="response_synthesizer")
    retriever: BaseRetriever = Field(default=None, description="retriever")

    def __init__(self, retriever, response_synthesizer):
        super().__init__()
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response = self.response_synthesizer.synthesize(query_str, nodes)
        return response

retriever = vector_index.as_retriever(similarity_top_k=3)
synthesizer = get_response_synthesizer(llm=llm, streaming=True)
my_query_engine = MyQueryEngine(retriever, synthesizer)
response = my_query_engine.query('你的问题')
```

**基于大模型组件的完全自定义引擎**（更灵活控制生成过程）：

```python
qa_prompt = PromptTemplate(
    "根据以下上下文回答输入问题：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "回答以下问题，不要编造\n"
    "我的问题: {query_str}\n"
    "答案: "
)

class MyLLMQueryEngine(CustomQueryEngine):
    llm: Ollama = Field(default=None, description="llm")
    retriever: BaseRetriever = Field(default=None, description="retriever")

    def __init__(self, retriever, llm):
        super().__init__()
        self.retriever = retriever
        self.llm = llm

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )
        return str(response)
```

底层流程：检索 Node → 构造上下文 → 组装 Prompt → 调用大模型。可根据业务设计任意个性化逻辑。

---

## 7.4 RAG 引擎：对话引擎（Chat Engine）

查询引擎：每次独立提问、不考虑历史。对话引擎：多次连续、有上下文的多轮交互，需跟踪过去对话上下文。

大模型本质是无状态服务，多次对话通过携带历史记录完成。RAG 多轮对话面临挑战：
1. 历史对话记录的透明保存/加载/携带
2. 检索时如何实现基于上下文理解的知识召回
3. 召回知识后采用何种响应生成模式

对话引擎本质是**查询引擎的有状态版本**，部分类型基于查询引擎构造。

### 7.4.1 对话引擎的两种构造方法

**1. 用索引快速构造**

```python
chat_engine = vector_index.as_chat_engine(chat_mode="condense_question")
print(chat_engine.chat('文心一言是什么？'))

chat_engine.reset()         # 复位开始新会话
chat_engine.chat_repl()     # 进入连续多轮交互式对话
```

**2. 底层 API 组合构造（condense_question 模式示例）**

```python
custom_prompt = PromptTemplate(
    """\
请根据以下的历史对话记录和新的输入问题，重写一个新的问题，使其能够捕捉对话中的所有相关上下文。
<Chat History>
{chat_history}
<Follow Up Message>
{question}
<Standalone question>
"""
)

custom_chat_history = [
    ChatMessage(role=MessageRole.USER, content="我们来讨论关于文心一言的一些问题吧"),
    ChatMessage(role=MessageRole.ASSISTANT, content="好的"),
]

query_engine = vector_index.as_query_engine()

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,                  # 对话引擎基于查询引擎构造
    condense_question_prompt=custom_prompt,     # 重写问题的 Prompt
    chat_history=custom_chat_history,           # 携带历史对话记录
    verbose=True,
)
chat_engine.chat_repl()
```

condense_question 模式：查看历史对话记录，把最新问题**重写**成语义更完整的独立问题，再输入查询引擎获答案。依赖组件：查询引擎、重写问题 Prompt、历史对话记录。

效果示例：第二个问题"它的主要场景是什么？"被重写成"'文心一言'的主要应用场景有哪些"，语义变完整，检索更准确。

### 7.4.2 对话引擎内部结构与运行原理

`as_chat_engine` 核心逻辑（根据 chat_mode 构造具体类型）：

```python
def as_chat_engine(self, chat_mode=ChatMode.BEST, llm=None, **kwargs) -> BaseChatEngine:
    query_engine = self.as_query_engine(llm=llm, **kwargs)   # 先构造查询引擎

    if chat_mode in [ChatMode.REACT, ChatMode.OPENAI, ChatMode.BEST]:
        query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)
        return AgentRunner.from_llm(tools=[query_engine_tool], llm=llm, **kwargs)

    if chat_mode == ChatMode.CONDENSE_QUESTION:
        return CondenseQuestionChatEngine.from_defaults(query_engine=query_engine, llm=llm, **kwargs)
    elif chat_mode == ChatMode.CONTEXT:
        return ContextChatEngine.from_defaults(retriever=self.as_retriever(**kwargs), llm=llm, **kwargs)
    elif chat_mode == ChatMode.CONDENSE_PLUS_CONTEXT:
        return CondensePlusContextChatEngine.from_defaults(retriever=self.as_retriever(**kwargs), llm=llm, **kwargs)
    elif chat_mode == ChatMode.SIMPLE:
        return SimpleChatEngine.from_defaults(llm=llm, **kwargs)
    else:
        raise ValueError(f"Unknown chat mode: {chat_mode}")
```

关键观察：
- **所有对话引擎都先构造了查询引擎**（验证对话引擎依赖查询引擎）——除 SIMPLE 只传 llm（不依赖检索器/查询引擎）
- REACT/OPENAI/BEST 通过 `AgentRunner` 实现，把查询引擎转为 `QueryEngineTool` 作为工具交给 Agent
- 各模式传入的参数透露依赖组件（CONTEXT/CONDENSE_PLUS_CONTEXT 传 retriever + llm；CONDENSE_QUESTION 传 query_engine + llm）

对话引擎底层依赖的 3 类组件：

1. **LLM**：作用不限于输出答案。Agent 模式中大模型要规划/推理用哪个工具；Condense 模式中要重写问题
2. **Query Engine 或 Retriever**：只用检索器的更简单；依赖查询引擎的支持更丰富响应生成模式
3. **Memory**：对话引擎区别于查询引擎的显著特征。有状态服务需组件记录/维持历史对话

### 7.4.3 理解不同对话模式

#### 对话模式与引擎类型对照表（表 7-7）

| 对话模式 | 引擎类型 | 依赖主要组件 |
|---------|---------|------------|
| `simple` | SimpleChatEngine | LLM |
| `condense_question` | CondenseQuestionChatEngine | QueryEngine, LLM |
| `context` | ContextChatEngine | Retriever, LLM |
| `condense_plus_context` | CondensePlusContextChatEngine | Retriever, LLM |
| `react` | ReActAgent | [Tool], LLM |
| `openai` | OpenAIAgent | [Tool], LLM |
| `best` | ReActAgent 或 OpenAIAgent | [Tool], LLM |

#### 1. simple 模式

直接与大模型对话，**不使用查询引擎/检索器，不检索上下文**。无需构造索引：

```python
chat_engine = SimpleChatEngine.from_defaults()
chat_engine.chat_repl()
```

可与模型连续带上下文对话，但无知识增强。

#### 2. condense_question 模式

在理解历史对话基础上把当前问题**重写成独立的、语义完整的问题**，再通过查询引擎获答案。适合 RAG 连续对话场景。

```python
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=vector_index.as_query_engine(),
    verbose=True,
)
```

内部流程（用 Langfuse 跟踪可见两次大模型调用）：
- 第一次：重写当前问题（让语义独立完整）
- 第二次：基于检索上下文回答重写后的问题

可指定不同响应生成模式：

```python
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=vector_index.as_query_engine(response_mode="refine"),
    verbose=True,
)
```

**优点**：每次检索前都根据历史记忆完善问题语义，大幅提高召回相关性（连续对话中单个问题常缺完整语义）。
**缺点**：增加大模型调用次数（重写问题 + 复杂响应生成模式）。

#### 3. context 模式

借助检索器从知识库检索上下文，**插入 system 提示信息**，用大模型回答。

```python
chat_engine = ContextChatEngine.from_defaults(
    retriever=vector_index.as_retriever(),
    llm=llm
)
# 等价：vector_index.as_chat_engine(chat_mode="context")
```

内部流程：一次知识检索 → 一次响应生成。system 提示中包含检索出的上下文。

**优点**：过程简单、响应快、不经过查询引擎复杂响应生成。
**缺点**：用当前问题直接检索，连续对话中当前问题语义不完整时可能召回无关知识，降低生成质量。

#### 4. condense_plus_context 模式

condense_question + context 的结合：先重写问题（结合历史+当前生成语义更完整问题）→ 再调用检索器召回与新问题相关上下文 → 大模型响应生成。

```python
chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=vector_index.as_retriever(similarity_top_k=1),
    llm=llm
)
```

内部流程（第二个上下文相关问题"有哪些场合可以使用它"）：两次大模型调用 + 一次向量检索。
- 第一次大模型：重写问题（输出新的独立问题）
- 向量检索：用重写后问题检索（不是原问题）
- 第二次大模型：把检索上下文注入 system 提示，响应生成

**特点**：结合两种模式优点——重写提高召回精确性 + 简化响应生成。代价是丧失响应生成模式的灵活性。

#### 5. Agent 对话模式（react / openai / best）

本质都构造 Agent：把查询引擎作为**工具**交给 Agent，由 Agent 参考输入问题与历史对话，规划并使用工具输出答案。

3 种区别：支持的大模型不同——OpenAI 或支持函数调用的大模型 → OpenAIAgent；否则 → ReActAgent。

```python
# 快速构造
chat_engine = vector_index.as_chat_engine(chat_mode="react")
chat_engine.chat_repl()
```

更精细控制（ReActAgent 拥有更多控制权，可指定工具辅助信息帮助大模型推理）：

```python
query_engine = vector_index.as_query_engine()

# 查询引擎"工具化"
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="query_engine",
    description="用于查询文心一言的相关信息"
)

chat_engine = ReActAgent.from_tools(tools=[query_engine_tool])
chat_engine.chat_repl()
```

Agent 推理过程（典型 3 步）：
1. 调大模型推理下一步动作（如"需要使用工具查询问题"，输出 Action + Action Input）
2. 用 Action Input 调用工具（查询引擎）：内部产生 retrieve + 第二次大模型调用（默认 compact 模式），带入检索上下文
3. 大模型根据工具结果再推理，认为可回答则直接输出答案，流程结束

**能力**：自我规划与使用工具。可挂多个工具（不同知识库查询工具），工具可自定义，不仅能查询知识还能执行复杂任务（如发邮件）。
**缺点**：过程复杂、延迟长、强依赖大模型推理能力，存在不确定性。

---

## 7.5 结构化输出

很多阶段需大模型做结构化输出方便下游处理。如 TreeIndex 检索需大模型输出 `ANSWER:(数字)`；响应生成时希望输出 JSON。

两种常见解析方法：
1. **`output_cls` 参数**：用 Pydantic 对象要求大模型结构化输出
2. **输出解析器**：在 llm 模块插入输出解析器解析与结构化

### 7.5.1 使用 output_cls 参数

```python
class Phone(BaseModel):
    """Information & features of a phone."""
    cpu: str
    memory: str
    storage: str
    screen: str

query_engine = index.as_query_engine(
    llm=Ollama(model='llama3:8b'),
    response_mode="tree_summarize",
    output_cls=Phone
)
response = query_engine.query("小麦手机的主要参数是什么？")
```

原理：指定 `output_cls` 时，调用大模型响应生成会**自动在 system 提示中插入结构化输出指令**，要求输出遵循类型格式，输出后转换为 `output_cls` 类型对象返回。

注意：
- 大模型输出无法通过 Pydantic 校验 → 转换失败抛异常
- 强依赖大模型指令遵从能力，失败时可能需换模型或换 Prompt

### 7.5.2 使用输出解析器（与 LangChain 集成）

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.llms.langchain.base import LangchainOutputParser  # 输出解析器适配

response_schemas = [
    ResponseSchema(name="name", description="手机名称"),
    ResponseSchema(name="cpu", description="手机处理器"),
    ResponseSchema(name="memory", description="手机内存"),
    ResponseSchema(name="features", description="手机特性", type="list"),
]

lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = LangchainOutputParser(lc_output_parser)

llm = OpenAI(output_parser=output_parser)            # 在 llm 模块插入解析器

query_engine = index.as_query_engine(llm=llm, verbose=True)
response = query_engine.query("小麦手机的主要参数是什么、其特性如何？")
```

查看解析器如何修改 Prompt：

```python
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
print(output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL))
```

输出解析器会在传入 Prompt 后**增加一段结构化输出指令**，达到要求结构化输出的目的。

---

## 【基础篇小结回顾】

基础篇从初级 RAG 实例 → 3 种开发方式（原生/LangChain/LlamaIndex）→ 经典 RAG 主要阶段开发 → 检索模式/响应生成模式/对话模式深入。完成基础篇即具备开发经典甚至有一定复杂度 RAG 应用的能力。

---

## 【面试考点】

**Q1：检索器的 retrieve 方法返回什么？as_retriever 和直接构造检索器有何区别？**
A：返回 `NodeWithScore` 列表（Node 增强类型，附带相关性 score）。`as_retriever` 是高层封装，通过 `retriever_mode` 参数选择模式，不需要记住具体检索器类名；直接构造需知道具体类型（如 VectorIndexRetriever）并传 index。建议优先用 `as_retriever`。

**Q2：refine、compact、tree_summarize 三种响应生成模式的区别？如何选型？**
A：
- **refine**：迭代细化，N 个 Node 至少 N 次大模型调用，第 1 次和后续 Prompt 不同，适合需要非常详细答案、不在乎成本的场景
- **compact**（默认）：先合并 Node 充分利用上下文窗口，再用 refine，调用次数更少，平衡成本与质量
- **tree_summarize**：树形递归总结，并行调用大模型生成多答案再合并，适合**总结性问题**

**Q3：查询引擎和对话引擎的本质区别？对话引擎底层依赖哪些组件？**
A：查询引擎无状态、每次独立提问；对话引擎有状态、需跟踪历史上下文。对话引擎是查询引擎的有状态版本，部分基于查询引擎构造。底层 3 类组件：LLM（重写问题/规划/生成）、Query Engine 或 Retriever（检索+生成）、Memory（记录历史对话，区别于查询引擎的关键）。

**Q4：condense_question、context、condense_plus_context 三种 RAG 对话模式的区别？**
A：
- **condense_question**：重写当前问题 → 查询引擎（检索+复杂响应生成）。优点：召回准确；缺点：调用多
- **context**：直接用当前问题检索 → 注入 system 提示 → 大模型回答。优点：简单快；缺点：当前问题语义不完整时召回差
- **condense_plus_context**：重写问题 → 检索 → 注入 system → 生成。结合两者优点，但丧失响应生成模式灵活性

**Q5：Agent 对话模式（react/openai/best）的工作原理？**
A：把查询引擎转为 QueryEngineTool 作为工具交给 Agent（OpenAI 函数调用模型 → OpenAIAgent，否则 → ReActAgent）。Agent 三步：① 大模型推理下一步动作 ② 用 Action Input 调工具（查询引擎内部检索+生成）③ 大模型根据结果再推理输出答案。能力是自规划用工具，可挂多工具；缺点是复杂、延迟长、依赖推理能力、有不确定性。

**Q6：如何实现结构化输出？两种方法的原理？**
A：① `output_cls` 参数：传 Pydantic 类，框架自动在 system 提示插入结构化输出指令，输出后转 Pydantic 对象返回，失败抛异常；② 输出解析器（如 LangchainOutputParser）：在 llm 模块插入解析器，解析器在 Prompt 后追加结构化指令。前者简单，后者可复用 LangChain 生态。

**Q7：递归检索（_handle_recursive_retrieval）发生在什么时候？IndexNode 起什么作用？**
A：发生在检索器 retrieve 出 Node 之后（不管什么索引类型都有这一步）。针对 IndexNode（特殊 Node，保存指向其他对象的引用）。若检索出的 Node 是 IndexNode，根据其指向的对象类型做二次检索：指向 Node→直接返回；指向查询引擎→调引擎返回结果组装成 Node；指向检索器→二次检索返回 Node。用于"从摘要找源内容""一级 Node 找二级检索器"。

**Q8：知识图谱索引支持哪些检索模式？如何组合？**
A：Text-to-Cypher（自然语言转图查询语言）、Vector Search（向量相似度）、Keywords Search（大模型提关键词+图库检索）。通过 `sub_retrievers` 参数组合多个子检索器（如 LLMSynonymRetriever + VectorContextRetriever）构造 PGRetriever。

**Q9：simple_summarize 模式截断 Node 的方式是什么？**
A：**不是**优先合并排名靠前 Node 丢弃后面，而是计算后发现需要截断时，**截断每个 Node 后面的溢出内容**。优点是快、简单，缺点是会丢失相关信息。

**Q10：自定义查询引擎如何实现？**
A：继承 `CustomQueryEngine` 实现 `custom_query` 接口，即可像内置引擎一样用 `.query()`。可用响应生成器（response_synthesizer + retriever 组合），也可直接用大模型组件（llm + retriever，自行构造上下文、组装 Prompt、调用 complete）实现完全自定义的生成逻辑，用于实现 C-RAG/Self-RAG 等模块化范式。

---

## 【易错 / 陷阱】

1. **refine 模式至少 N 次大模型调用**：N 个检索 Node → 至少 N 次。不要在性能敏感场景误用。第 1 次和后续 Prompt 模板不同（后续要带 `{existing_answer}`）。

2. **compact 不一定只调一次大模型**：合并后若上下文窗口溢出，仍会降级到 refine 迭代。说"compact 只调一次"是误解。

3. **tree_summarize 测试时少量 Node 看不出效果**：手工构造 2-3 个 Node 会被直接合并成 1 个。要观察树形总结必须用足够长的文档模拟多个 Node。

4. **simple_summarize 截断不是"丢弃靠后 Node"**：是截断**每个 Node 后面的溢出内容**，会均匀损失信息。

5. **context 模式用当前问题直接检索**：连续对话中当前问题常缺上下文（如"它的主要场景是什么"），直接检索会召回无关知识。RAG 连续对话优先 condense_question 或 condense_plus_context。

6. **condense_question 增加大模型调用**：每次都要先重写问题再生成，成本和延迟翻倍，不是免费的。

7. **output_cls 结构化输出依赖模型能力**：模型指令遵从差会校验失败抛异常。换模型或调 Prompt 才能解决，不要假设一定能成功。

8. **不同索引的 retriever_mode 含义不同**：VectorStoreIndex 忽略 retriever_mode（只有向量检索）；TreeIndex/KeywordTableIndex/DocumentSummaryIndex 才有多模式。choice_top_k / similarity_top_k 在 DocumentSummaryIndex 中是"摘要 Node 数量"不是"返回 Node 数量"。

9. **SIMPLE 对话模式不需要索引**：直接与大模型对话，不检索知识。不要给它传索引或期望 RAG 增强。

10. **Agent 对话模式有不确定性**：强依赖大模型推理能力，可能规划错误、调用错工具或循环。生产需设最大迭代次数、超时、回退。

11. **as_query_engine 会隐式构造默认检索器**：默认 similarity_top_k=2，不显式设置可能与预期不符。生产需显式 `as_retriever(similarity_top_k=N)` 或在 as_query_engine 中指定。

12. **streaming 在多调用模式中只有最后一次流式**：refine/compact 多次调用大模型时，只有最后一次用流式输出，前面几次不可见。

13. **NodePostProcessor 在检索后生效**：重排序等后处理在 retrieve 之后、synthesizer 之前，容易忽略这层导致检索结果与生成输入不一致。

14. **对话引擎的 Memory 是有状态关键**：查询引擎无 Memory，复用查询引擎做"对话"不会自动带历史，必须用对话引擎或在查询引擎外自行维护历史。
