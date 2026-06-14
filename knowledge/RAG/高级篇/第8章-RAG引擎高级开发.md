# 第 8 章 RAG 引擎高级开发

> 来源：严灿平《基于大模型的 RAG 应用开发与优化——构建企业级 LLM 应用》【高级篇】
> 定位：模块化 RAG 时代的高级开发阶段与相关技术应用，面向程序员系统学习。

## 本章总览

随着 RAG 应用在生产中不断改进，更多的组件、算法、优化流程/范式不断出现，使 RAG 具备更广泛的适应能力与更精确的生成能力。本章聚焦模块化 RAG 的常见高级开发阶段：

1. **检索前：查询转换**（HyDE / 多步 / 子问题）
2. **检索后：节点后处理器**（过滤、重排序 Rerank）
3. **语义路由**（Router 选择数据源/索引/响应类型）
4. **SQL 查询引擎**（Text-to-SQL 处理结构化数据）
5. **多模态文档处理**（PDF 文本/表格/图片）
6. **查询管道 QueryPipeline**（基于 Graph 编排 RAG 工作流）

---

## 8.1 检索前查询转换

### 8.1.0 概念与动机

**查询转换（Query Transformation）**，又称查询重写 / 查询分析，是一个 **"检索前"** 阶段，用于将输入问题转换成一种或多种其他形式的查询输入。

**为什么需要查询转换？** 用户习惯用单个问题查询，但单个问题可能无法完整或深入地表达真实意图，导致召回知识无法覆盖所需内容。

**4 种常见类型：**

1. 将问题转换为更有利于嵌入（embedding）的问题，提高召回精确性。
2. 对问题进行语义丰富与扩展，生成更全面/准确的答案。
3. 将初始查询分解成多个子问题，分别查询后合成答案。
4. 将初始查询分解成可多步完成的子查询，分步查询得出答案。

**示例：** 输入"GPT-4 模型" → 生成"GPT-4 基准测试性能""GPT-4 使用定价""GPT-4 API 介绍"等相关问题，多视角召回。

**执行模式：**
- **一次完成**：检索前对输入重写。
- **多次迭代**：新型 RAG 范式中，重写 → 检索 → 生成 → 评估质量 → 再次重写，多次迭代。

### 8.1.1 简单查询转换

借助大模型 + Prompt 进行简单重写：

```python
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

prompt_rewrite_temp = """\
您是一个聪明的查询生成器。请生成与以下查询相关的{num_queries}个查询问题 \n
注意每个查询问题都占一行 \n
我的查询：{query}
生成查询列表：
"""
prompt_rewrite = PromptTemplate(prompt_rewrite_temp)
llm = OpenAI(model="gpt-3.5-turbo")

def rewrite_query(query: str, num: int = 3):
    response = llm.predict(prompt_rewrite, num_queries=num, query=query)
    queries = response.split("\n")   # 假设每行一个问题
    return queries

print(rewrite_query("中国目前大模型的发展情况如何？"))
```

**缺点：** 缺少上下文与额外指令时，不确定性大，易产生真实意图偏离。

**适用场景（不在检索前用，建议在数据准备阶段用）：**
1. 用原始（问答类）知识生成相似问题，做语义丰富。
2. 用于元数据抽取，用原始知识生成假设性查询问题用于嵌入。

### 8.1.2 HyDE 查询转换

**HyDE（Hypothetical Document Embeddings，假设性文档嵌入）**：根据输入问题生成一个 **假设性答案**，然后对该假设性答案进行嵌入与检索（可同时携带原问题）。

**原理机制（图 8-1）：**
```
输入问题 → LLM 生成假设性答案(文档) → 对假设性答案做嵌入 → 检索
```
核心思想：**答案与答案在向量空间更接近**（query 与 answer 之间常有语义鸿沟，doc 与 doc 更近）。

**关键 API：** `HyDEQueryTransform`

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

# 中文化 Prompt
hyde_prompt_temp = """\
请生成一段文字来回答输入问题\n
尽可能含有更多的关键细节\n
{context_str}
生成内容：
"""
hyde_prompt = PromptTemplate(hyde_prompt_temp)

llm = OpenAI(model="gpt-3.5-turbo")
hyde = HyDEQueryTransform(llm=llm)
hyde.update_prompts({'hyde_prompt': hyde_prompt})

query_bundle = hyde.run("请介绍小麦手机的主要配置")
print(query_bundle.__dict__)
```

**关键点（重要陷阱）：**
- HyDE 转换结果用于 **嵌入与检索**，并不直接返回新的 `query_str`。
- 转换后的假设性文档被放在 `query_bundle` 的 **`custom_embedding_strs` 字段**，查询时该字段被用于嵌入与检索。

**最佳实践 —— 用 `TransformQueryEngine` 透明包装：**

```python
query_engine = create_city_engine('南京市')
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

response_hyde = hyde_query_engine.query("南京市的人口是多少？经济发展如何？")
```

效果对比：未加 HyDE 时召回相关性一般；加 HyDE 后召回知识相关性提高，答案质量更高。

### 8.1.3 多步查询转换

**思想：** 对复杂问题，单次检索召回不够完整。从初始复杂查询开始，经过多步查询转换与检索生成，直至能完整回答。**每一步查询转换都基于之前的推理过程**，提出下一步问题（图 8-5）。

**关键 API：** `StepDecomposeQueryTransform` + `MultiStepQueryEngine`

```python
from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform
from llama_index.core import PromptTemplate, MultiStepQueryEngine

query_engine = create_city_engine(['北京市','上海市'])

prompt_templ = """
我们有机会从知识源中回答部分或全部问题。知识源的上下文如下，提供了之前的推理步骤。
根据上下文和之前的推理，返回一个可以从上下文中回答的问题：
1. 这个问题可以帮助回答原问题，与原问题密切相关。
2. 可以是原问题的子问题，或者是解答原问题需要的一个步骤中需要的问题。
如果无法从上下文中提取更多信息，则提供"无"作为答案。下面给出了一个示例：

-----
问题：2020 年澳大利亚网球公开赛冠军获得了多少个大满贯冠军？
知识源上下文：提供了 2020 年澳大利亚网球公开赛冠军的名字
之前的推理：无
新问题：谁是 2020 年澳大利亚网球公开赛的冠军？
-----

我的问题：{query_str}
知识源上下文：{context_str}
之前的推理：{prev_reasoning}
新问题：
"""

step_transformer = StepDecomposeQueryTransform(llm=llm_openai, verbose=True)
step_transformer.update_prompts({'step_decompose_query_prompt': PromptTemplate(prompt_templ)})

step_query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_transformer,
    index_summary='这是一个关于城市的知识库，用于回答与城市信息相关的问题'
)

response = step_query_engine.query("中国首都的城市人口有多少？和上海相比呢？")
```

**关键参数：** `index_summary` —— 向大模型描述知识库的整体范围，帮助判断"能否从知识库回答"。

**效果：** 普通查询引擎遇到此类非事实性、需多处召回的问题，在 top_k 较小时容易回答失败或"幻觉"；多步转换分步生成子问题后能正确回答。

### 8.1.4 子问题查询转换

**思想：** 通过生成与原问题相关的多个 **具体的子问题** 来解释/理解原问题（图 8-7）。

**特殊场景：** 借助 Agent 思想，根据 **可用的工具** 将输入问题转换为 **每个工具都可以解答** 的子问题。

**与多步查询的区别：**
| 维度 | 子问题转换 | 多步转换 |
|------|-----------|---------|
| 约束 | 需参考可用工具，更具约束性 | 基于推理过程 |
| 生成时机 | 一次性生成 | 多步迭代生成 |

#### 8.1.4.1 子问题生成器

```python
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core.tools import ToolMetadata

llm = OpenAI()
question_rewriter = OpenAIQuestionGenerator.from_defaults(llm=llm)
question_rewriter.update_prompts({'question_gen_prompt': PromptTemplate(question_gen_prompt_templ)})

# 注意：这里只是提供工具的"元数据"，并未真正提供工具实现
tool_choices = [
    ToolMetadata(name="query_tool_beijing",
                 description="用于查询北京市各个方面的信息，如基本信息、旅游指南、城市历史等"),
    ToolMetadata(name="query_tool_shanghai",
                 description="用于查询上海市各个方面的信息，如基本信息、旅游指南、城市历史等"),
]

query_str = "北京与上海的人口差距是多少？它们的面积相差多少？"
choices = question_rewriter.generate(tool_choices, QueryBundle(query_str=query_str))
```

**关键 Prompt 要点：**
- 子问题尽可能具体、与用户问题相关。
- 子问题应可通过提供的工具回答。
- 可为每个工具生成多个子问题。
- 工具必须用 **名称**（而非描述）指定。
- 不相关时可不使用工具。

**两种生成器选择：**
- `OpenAIQuestionGenerator`：依赖函数调用（function calling），**只能用于支持函数调用的大模型**。
- `LLMQuestionGenerator`：用于不支持函数调用的大模型，用法一致。

**技巧（重要）：** 工具不一定是真实存在的查询引擎，可以把一些 **用于约束/引导子问题生成的描述"假装"成工具**，修改 Prompt 模板后生成子问题，再根据输出的工具名称有针对性地处理 → 提供了更大灵活性。

#### 8.1.4.2 子问题查询引擎（SubQuestionQueryEngine）

**工作原理（图 8-9）：**
1. 查询转换器对输入问题判断与分解。
2. 对分解出的多个子问题调用对应的查询引擎（工具）进行查询。
3. 将查询结果汇总作为上下文交给大模型生成最终答案。

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_nanjing = create_city_engine('南京市')
query_engine_shanghai = create_city_engine('上海市')

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine_nanjing,
        metadata=ToolMetadata(name="query_tool_nanjing",
                              description="用于查询南京市各个方面的信息"),
    ),
    QueryEngineTool(
        query_engine=query_engine_shanghai,
        metadata=ToolMetadata(name="query_tool_shanghai",
                              description="用于查询上海市各个方面的信息"),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,   # 异步并行执行子问题查询
)

response = query_engine.query("北京与上海的人口差距是多少？GDP 大约相差多少？使用中文回答")
```

#### 8.1.4.3 子问题查询引擎 vs Agent（重要对比）

两者都是把工具集交给工作引擎，让其细分任务并选择工具。**区别：**

| 维度 | SubQuestionQueryEngine | Agent（如 ReAct） |
|------|----------------------|------------------|
| 任务规划 | 借助大模型 **一次性** 完成子问题生成 | **动态** 完成（执行中观察结果再推理下一步） |
| 执行特点 | 确定性强 | 更符合人类行为模式，更灵活，但不确定性更大 |

---

## 8.2 检索后处理器（Node Postprocessor）

### 8.2.0 概念

从检索器输出到响应生成器输入之间，往往需要额外处理（关键词筛选、多检索器重排序等）。LlamaIndex 提供 **节点后处理器（Node Postprocessor）** 模块化方案。

**工作位置：** 检索器之后、响应生成器之前（图 8-11）。

**两种使用方式：**
1. 独立使用：构造处理器 → 调用 `postprocess_nodes`。
2. 插入查询引擎：作为 `node_postprocessors` 参数传入。

### 8.2.1 使用节点后处理器

**独立使用：**

```python
processor = SimilarityPostprocessor(similarity_cutoff=0.8)
filtered_nodes = processor.postprocess_nodes(nodes_with_scores)
```

**插入查询引擎：**

```python
vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
)
```

### 8.2.2 实现自定义的节点后处理器

从 `BaseNodePostprocessor` 派生，实现 `_postprocess_nodes` 接口：

```python
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class MyNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        pattern = r"过滤正则表达式"
        filtered_nodes = [node for node in nodes if not re.search(pattern, node.text)]
        return filtered_nodes

query_engine = vector_index.as_query_engine(
    node_postprocessors=[MyNodePostprocessor()]
)
```

### 8.2.3 常见的预定义节点后处理器

#### 1. 相似度过滤处理器
- **类型：** `SimilarityPostprocessor`
- **参数：** `similarity_cutoff`（评分阈值），只保留高于阈值的 Node。

#### 2. 关键词过滤处理器
- **类型：** `KeywordNodePostprocessor`
- **参数：** `required_keywords`（需匹配）、`exclude_keywords`（需排除）。

**注意（中文陷阱）：** 内置关键词处理器使用 **spacy NLP 库**（支持英文词形还原/词干提取），中文需在构造时指定 `lang` 参数：

```python
processor = KeywordNodePostprocessor(
    required_keywords=["小麦手机"],
    exclude_keywords=[],
    lang='zh-Hans'
)
```

#### 3. 元数据替换处理器（MetadataReplacementPostProcessor）

**作用：** 用元数据中某个 key 的内容替换 Node 中的 text 内容。

**典型配合 `SentenceWindowNodeParser`（句子窗口分割器）使用：**

```python
node_parser = SentenceWindowNodeParser.from_defaults(
    sentence_splitter=my_chunking_tokenizer_fn,
    window_size=3,                        # 前后保留 3 句
    window_metadata_key="window",         # 窗口内容存放 key
    original_text_metadata_key="original_text",
)
nodes = node_parser.get_nodes_from_documents(docs)
vector_index = VectorStoreIndex(nodes=nodes)
```

**机制：** Node 的 `text` 只存当前句子（嵌入用小粒度提升精度），但 `metadata["window"]` 保存前后 3 句上下文。检索时用元数据替换处理器把 `text` 替换为完整窗口内容，从而在生成阶段携带更多上下文：

```python
query_engine = vector_index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
```

**核心思想（重要）：** **小粒度嵌入、大粒度生成**（small-to-big retrieval）—— 嵌入精准，生成完整。

#### 4. 固定时间排序处理器（FixedRecencyPostprocessor）

根据元数据中的时间 key 倒排序，返回最近 N 个 Node：

```python
query_engine = vector_index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[FixedRecencyPostprocessor(top_k=1, date_key="create_time")],
)
```

**其他变体：** `EmbeddingRecencyPostprocessor` —— 用嵌入模型判断相似度，删除过于相似的旧 Node 后再时间排序。

### 8.2.4 Rerank 节点后处理器（重点）

**Rerank（重排序）**：对检索出的 Node 列表重新排序，使越准确/相关的 Node 排名越靠前，让大模型优先考虑更相关内容。

**为什么有了向量语义检索还需要重排序？**
1. RAG 有多种索引类型，很多索引并非基于语义/向量（如关键词索引），其检索结果需要独立阶段重排序。
2. 复杂范式常用混合检索（hybrid retrieval），来自不同来源/不同技术的知识更需要统一重排序。
3. 即使纯向量索引，受嵌入模型、相似度算法、语言环境、领域知识特点影响，语义排序也可能偏差，需独立重排序纠正。

**核心：重排序通常需要独立的 Rerank 模型实现。** Rerank 模型与 embedding 模型不同 —— Rerank 是 cross-encoder（query 与 doc 拼接后联合编码），精度更高但成本更高。

#### 8.2.4.1 使用 Cohere Rerank 模型

Cohere Rerank 是商业闭源 Rerank 模型，专门用于重排关键词/向量搜索结果。

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

retriever = vector_index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("百度文心一言的逻辑推理能力怎么样？")

cohere_rerank = CohereRerank(
    model='rerank-multilingual-v3.0',
    api_key='***',
    top_n=2
)
rerank_nodes = cohere_rerank.postprocess_nodes(
    nodes, query_str='百度文心一言的逻辑推理能力怎么样？'
)
```

**典型现象：** Rerank 前后 top-2 排名可能完全相反（重排序后评分与相关性更匹配）。

#### 8.2.4.2 使用 bge-reranker-large 模型（开源）

智源研究院开源，使用 HuggingFace **TEI（Text Embeddings Inference）** 部署：

```bash
model=BAAI/bge-reranker-large
text-embeddings-router --model-id $model --port 8080
```

**LlamaIndex 没有内置 bge-reranker 处理器，需自定义：**

```python
import requests
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class BgeRerank(BaseNodePostprocessor):
    url: str = Field(description="Rerank server url.")
    top_n: int = Field(description="Top N nodes to return.")

    def __init__(self, top_n: int, url: str):
        super().__init__(url=url, top_n=top_n)

    def rerank(self, query, texts):
        url = f"{self.url}/rerank"
        request_body = {"query": query, "texts": texts, "truncate": False}
        response = requests.post(url, json=request_body)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to rerank, detail: {response}")
        return response.json()

    @classmethod
    def class_name(cls) -> str:
        return "BgeRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        texts = [node.text for node in nodes]
        results = self.rerank(query=query_bundle.query_str, texts=texts)

        new_nodes = []
        for result in results[0: self.top_n]:
            new_node_with_score = NodeWithScore(
                node=nodes[int(result["index"])].node,
                score=result["score"],
            )
            new_nodes.append(new_node_with_score)
        return new_nodes
```

**使用：**

```python
customRerank = BgeRerank(url="http://localhost:8080", top_n=2)
query_engine = vector_index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[customRerank],
)
```

**重要模式：** 自定义处理器通过 `BaseNodePostprocessor` + `_postprocess_nodes` 实现，接收 `nodes` 与 `query_bundle`，返回重排序后的 `NodeWithScore` 列表。

---

## 8.3 语义路由（Semantic Router）

### 8.3.1 了解语义路由

**场景：** 根据不同知识库/应用特点构造了不同的查询引擎（不同领域、不同索引如 VectorIndex 与 GraphIndex），需给使用者提供一致体验，无需关心后端真实引擎。

**路由模块（Router）** 在检索之前识别使用者意图，根据意图将输入问题交给不同的检索生成流程（图 8-22）。

**4 种典型场景：**
1. 单纯选择器，在多种选择中决策。
2. 在多种知识数据源中选择查询目标。
3. 在多种索引/响应类型中选择（事实性回答 vs 总结）。
4. 选择多个查询引擎同时响应并合并结果。

**路由模块组成（两部分）：**
1. **Selector（选择器）：** 借助大模型实现。两种类型：
   - **Pydantic 选择器**：依赖 OpenAI 函数调用实现路由。
   - **通用 LLM 选择器**：把可选择信息组装进 Prompt，要求大模型按语义选择。
2. **多个候选项：** 多个查询引擎、检索器，甚至简单字符串选项。

```python
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[...多个 tool...]
)
```

### 8.3.2 带有路由功能的查询引擎（RouterQueryEngine）

**场景 A：不同数据源路由**

```python
docs_xiaomai = SimpleDirectoryReader(input_files=[".../xiaomai.txt"]).load_data()
docs_yiyan = SimpleDirectoryReader(input_files=[".../yiyan.txt"]).load_data()

vectorindex_xiaomai = VectorStoreIndex.from_documents(docs_xiaomai)
query_engine_xiaomai = vectorindex_xiaomai.as_query_engine()
vectorindex_yiyan = VectorStoreIndex.from_documents(docs_yiyan)
query_engine_yiyan = vectorindex_yiyan.as_query_engine()

# 把候选引擎包装成工具
tool_xiaomai = QueryEngineTool.from_defaults(
    query_engine=query_engine_xiaomai,
    description="用于查询小麦手机的信息",
)
tool_yiyan = QueryEngineTool.from_defaults(
    query_engine=query_engine_yiyan,
    description="用于查询文心一言的信息",
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[tool_xiaomai, tool_yiyan]
)
response = query_engine.query("什么是文心一言，用中文回答")
```

**场景 B：同数据源不同响应类型路由**

```python
query_engine_quesiton = vectorindex_xiaomai.as_query_engine(response_mode="compact")
query_engine_summary = vectorindex_xiaomai.as_query_engine(response_mode="simple_summarize")

tool_question = QueryEngineTool.from_defaults(
    query_engine=query_engine_quesiton,
    description="用于回答事实性与细节性的问题",
)
tool_summarize = QueryEngineTool.from_defaults(
    query_engine=query_engine_summary,
    description="用于回答总结性的问题",
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[tool_question, tool_summarize],
    verbose=True
)
```

### 8.3.3 带有路由功能的检索器（RouterRetriever）

把检索器（而非查询引擎）作为候选工具，输出 Node 而非最终答案：

```python
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.tools import RetrieverTool

retriever_xiaomai = vector_index.as_retriever()
retriever_yiyan = vector_index2.as_retriever()

tool_xiaomai = RetrieverTool.from_defaults(retriever=retriever_xiaomai,
                                           description="用于查询小麦手机的信息")
tool_yiyan = RetrieverTool.from_defaults(retriever=retriever_yiyan,
                                         description="用于查询文心一言的信息")

retriever = RouterRetriever(
    selector=LLMSingleSelector.from_defaults(),
    retriever_tools=[tool_xiaomai, tool_yiyan]
)
nodes = retriever.retrieve("什么是文心一言？")
```

**注意：** RouterRetriever 本质只是检索器，无法直接回答问题，只能输出 Node。

### 8.3.4 使用独立的选择器

**用于任意选择决策（不只是引擎）**，选项用 `ToolMetadata` 或字符串定义：

```python
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata

choices = [
    ToolMetadata(description="查询当前实时的信息", name="web_search"),
    ToolMetadata(description="知识查询或内容创作", name="query_engine"),
]

selector = LLMSingleSelector.from_defaults()
selector_result = selector.select(choices, query="写一个悬疑小故事?")
print(selector_result.selections)   # 大模型选择 index=1
```

### 8.3.5 可多选的路由查询引擎（LLMMultiSelector）

将查询 **同时路由到多个引擎**，利用多种索引特点综合生成，最后用大模型汇总。

```python
from llama_index.core.selectors import LLMMultiSelector

summary_index = SummaryIndex.from_documents(docs, chunk_size=100, chunk_overlap=0)
vector_index = VectorStoreIndex.from_documents(docs, chunk_size=100, chunk_overlap=0)
keyword_index = SimpleKeywordTableIndex.from_documents(docs, chunk_size=100, chunk_overlap=0)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
    description="有助于总结与小麦手机相关的问题",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(),
    description="适合检索与小麦手机相关的特定上下文",
)
keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_index.as_query_engine(),
    description="适合使用关键词从文章中检索特定的上下文",
)

query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),   # 关键：多选
    query_engine_tools=[summary_tool, vector_tool, keyword_tool],
    verbose=True
)
response = query_engine.query("小麦手机的屏幕特点和优势是什么")
```

**关键点：** 不同查询语义下，多选路由会选择不同组合的引擎。

---

## 8.4 SQL 查询引擎

### 8.4.0 概念

前面查询引擎主要处理半结构化/非结构化数据。企业中大量 **结构化数据** 存在关系数据库中，最方便的不是向量化，而是让它们留在数据库中用 **SQL 语句** 查询。借助大模型实现 **Text-to-SQL**（自然语言 → SQL），构造 SQL 查询引擎（图 8-26）。

### 8.4.1 使用 NLSQLTableQueryEngine（最简洁）

```python
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI

engine = create_engine("postgresql://postgres:****@localhost:5432/postgres")
sql_database = SQLDatabase(engine, include_tables=["customers","orders"])

llm_openai = OpenAI(model='gpt-3.5-turbo')
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["customers","orders"],
    llm=llm_openai
)

response = query_engine.query("一共有多少个订单")
```

**参数：** `SQLDatabase` 对象 + 查询的表 + 大模型。

**原理（图 8-29）：** 把用户问题 + 表的 Schema（结构与描述）组装成 Prompt → 大模型生成 SQL → 执行 SQL → 对结果总结输出答案。

### 8.4.2 基于实时表检索的查询引擎（解决大库上下文溢出）

**问题：** `NLSQLTableQueryEngine` 把所有表 Schema 组装进 Prompt。表过多会导致：
- 上下文窗口溢出。
- 干扰大模型判断。
- token 成本过高。

**解决思路：** Text-to-SQL 之前，先根据输入问题 **检索需要的数据库表**，再基于相关表 Schema 生成 SQL。

**关键 API：** `SQLTableRetrieverQueryEngine` + `ObjectIndex`（检索 `SQLTableSchema` 对象）

```python
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import (
    SQLTableNodeMapping, SQLTableSchema, ObjectIndex,
)
from llama_index.core import VectorStoreIndex

engine = create_engine("postgresql://postgres:****@localhost:5432/postgres")
sql_database = SQLDatabase(engine, include_tables=["customers","orders"])

# table_node_mapping: SQLTableSchema 对象与向量索引 Node 的映射
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name="customers"),
    SQLTableSchema(table_name="orders"),
    SQLTableSchema(table_name="mystore"),
]

# 对象索引，底层用向量存储索引做语义检索
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=1)   # 传入 retriever 而非 SQLTableSchema 列表
)
response = query_engine.query("所有订单总金额是多少？")
```

**验证检索结果：**

```python
table_retriever = obj_index.as_retriever(similarity_top_k=1)
tables = table_retriever.retrieve("所有订单总金额是多少")
print(tables)   # 应回退到 orders 表
```

**核心模式（重要）：ObjectIndex —— 对任意 Python 对象构建向量索引**，通过 `as_retriever` 提供语义检索能力，是 LlamaIndex 中把"非文本对象"纳入 RAG 流程的通用手段。

### 8.4.3 使用 SQL 检索器（NLSQLRetriever）

如果想用标准 `RetrieverQueryEngine` 构造 SQL 查询引擎，可用 `NLSQLRetriever` 而非 SQL 查询引擎：

```python
from llama_index.core.query_engine import RetrieverQueryEngine

nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["customers","orders"], return_raw=True
)
query_engine = RetrieverQueryEngine.from_args(nl_sql_retriever)
response = query_engine.query("所有订单总金额是多少？")
```

---

## 8.5 多模态文档处理

### 8.5.1 多模态文档处理架构（图 8-32）

**通用架构（3 步）：**

**第 1 步：分类提取。** 借助解析工具从 PDF 中分类提取：
- 文本（Text）→ Markdown 表示
- 表格（Table）→ 本地/网络文档
- 图片（Image）

**第 2 步：对不同形态用不同索引/检索方法。**
- ① **文本**：与普通文本相同，构造向量存储索引。
- ② **表格**：直接嵌入效果欠佳 → 借助大模型 **生成表格摘要** 用于嵌入/检索，检索阶段 **递归检索出原始表格** 用于生成。
- ③ **图片**：借助多模态视觉大模型（Qwen-VL、GPT-4V）+ OCR：
  - a. 纯文字图片 → OCR 识别成文本 → 普通文本处理。
  - b. 其他图片 → 多模态大模型生成摘要用于索引/检索，检索后递归检索原始图片用于生成。

**第 3 步：检索相关知识 → 大模型生成。** 需要原始图片时须用多模态大模型。

**3 种核心技术：**

| 技术 | 说明 | 常见工具 |
|------|------|---------|
| 文档解析与提取 | 半结构化/结构化文档解析 | LlamaParse、Unstructured、Open-Parse |
| 多模态视觉大模型 | 理解图片/OCR | Qwen-VL、GPT-4V、LLaVA；专业 OCR 模型/工具 |
| 递归检索 | 通过摘要 Node 找到原始表/图 | LlamaIndex 的 IndexNode |

**工具详解：**
- **LlamaParse**：LlamaIndex 在线文档解析服务，与框架深度集成（可自动生成表格摘要），必须在线。
- **Unstructured**：非结构化数据处理平台，提供商业 API + 开源 SDK，支持清理/语义分割/实体提取；类似 OmniParse。
- **Open-Parse**：轻量级开源库，支持语义分块 + OCR，可与 LlamaIndex 集成。

### 8.5.2 使用 LlamaParse 解析文档

#### 安装

```bash
pip install llama-parse
```

#### 基本使用

```python
from llama_parse import LlamaParse

documents = LlamaParse(
    result_type="markdown",
    language='ch_sim'
).load_data("./../data/zte-report-simple.pdf")
print(f'{len(documents)} documents loaded.\n')
```

**两个核心参数：**
- `result_type`：解析出的 Document 内容格式（建议 `markdown`）。
- `language`：文档语言（简体中文 `ch_sim`）。

**作为自定义文档阅读器配合 SimpleDirectoryReader：**

```python
parser = LlamaParse(result_type="markdown", language='ch_sim')
documents = SimpleDirectoryReader(
    "./data", file_extractor={".pdf": parser}
).load_data()
```

**典型问题：** LlamaParse 解析出表格为 Markdown 文本，但直接嵌入后语义信息不足，导致事实性数据查询错误（典型召回不精确）。解决方法见 8.5.3。

### 8.5.3 多模态文档中的表格处理（MarkdownElementNodeParser）

**核心组件：** `MarkdownElementNodeParser`，针对复杂 Markdown 格式的 Node，区分不同"元素"（文本 vs 表格），并增强处理表格元素。

**机制：** 对表格内容借助大模型生成 **内容摘要 + 结构描述**，构造成 **IndexNode（索引节点）**。检索时通过 IndexNode 找到表格内容 Node，一起输入大模型生成。

```python
from llama_index.core.node_parser import MarkdownElementNodeParser

DEFAULT_SUMMARY_QUERY_STR = """\
请用中文简要介绍表格内容。\
这个表格是关于什么的？给出一个非常简洁的摘要（想象你正在为这个表格添加一个新的标题和摘要），\
如果提供了上下文，那么请输出真实/现有的表格标题/说明。\
如果提供了上下文，那么请输出真实/现有的表格 ID。\
还要输出表格是否应该保留的信息。\
"""

node_parser = MarkdownElementNodeParser(summary_query_str=DEFAULT_SUMMARY_QUERY_STR)
nodes = node_parser.get_nodes_from_documents(documents)

# 分离普通文本 Node (TextNode) 与索引 Node (IndexNode)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

index = VectorStoreIndex(nodes=base_nodes + objects, storage_context=storage_context)
query_engine = index.as_query_engine(similarity_top_k=10, verbose=True)
```

**内部处理逻辑（5 步，重要）：**
1. 解析 PDF → Document，文本与表格内容构造成 TextNode。
2. 对表格 Node 用大模型 + Prompt 生成摘要/介绍/标题，构造新的 **IndexNode**。
3. `get_nodes_and_objects` 分离表格内容 Node，用 IndexNode 指向它们。
4. 用 base_nodes（普通文本）+ objects（IndexNode，含辅助信息并指向表格 Node）构造向量索引。
5. 检索时若命中 IndexNode → 自动 **递归检索** 其指向的表格内容 Node → 用于生成。

**为什么有效：** 用生成的表格描述/摘要做向量检索（精度高），生成时又能把原始表格交给大模型（信息全）。

### 8.5.4 多模态大模型的基础应用

#### 1. 从图片到自然语言文本（Qwen-VL）

```python
from llama_index.multi_modal_llms.dashscope import (
    DashScopeMultiModal, DashScopeMultiModalModels,
)
from llama_index.multi_modal_llms.dashscope.utils import (
    create_dashscope_multi_modal_chat_message, load_local_images,
)
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
import os

os.environ["DAHSCOPE_API_KEY"] = "sk-***"

image_documents1 = load_image_urls(["https://.../dog_and_girl.jpeg"])
image_documents2 = load_local_images(["file:///Users/.../xiaomi.png"])

dashscope_multi_modal_llm = DashScopeMultiModal(
    model_name=DashScopeMultiModalModels.QWEN_VL_PLUS
)

chat_message = create_dashscope_multi_modal_chat_message(
    "请概括这两张图片中的信息",
    MessageRole.USER,
    image_documents1 + image_documents2
)
chat_response = dashscope_multi_modal_llm.chat([chat_message])
print(chat_response.message.content[0]["text"])
```

**流程：**
1. 申请阿里云 DashScope API Key，设置环境变量。
2. 构造 `DashScopeMultiModal` 对象指定 Qwen-VL-Plus。
3. `chat` 接口输入消息（含图片 + Prompt）。

**能力：** 通用 OCR、视觉推理、中文文本理解，可批量输入图片。

#### 2. 从图片到结构化对象

```python
from pydantic import BaseModel
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser

class Phone(BaseModel):
    """定义对象结构"""
    name: str
    cpu: str
    battery: str
    display: str

prompt_template_str = """\
{query_str}
请把结果作为一个 Pydantic 对象返回，对象格式如下:
"""

mm_program = MultiModalLLMCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Phone),       # 输出解析器指定对象类型
    image_documents=image_documents,                  # 输入图片
    prompt_template_str=prompt_template_str,          # Prompt 模板
    multi_modal_llm=dashscope_multi_modal_llm,        # 多模态大模型
    verbose=True,
)

response = mm_program(query_str="请描述图片中的信息。")
pprint.pprint(response.__dict__)
```

**机制：** Prompt 要求多模态大模型输出 Pydantic 对象格式 → `MultiModalLLMCompletionProgram` 自动封装消息/生成/解析 → 输出 Pydantic 对象。

#### 3. 直接嵌入图片（多模态嵌入）

依赖专门的多模态嵌入模型，利用 Chroma 内置的 OpenCLIPEmbeddingFunction：

```python
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

image_uris = sorted([os.path.join('./jpgs/', n) for n in os.listdir('./jpgs/')])
ids = [str(i) for i in range(len(image_uris))]

chroma_collection.add(ids=ids, uris=image_uris)

# 用文本查询图片（跨模态检索）
retrieved = chroma_collection.query(
    query_texts=["很多辆汽车"],
    include=['data'], n_results=2
)
print(retrieved['uris'])
```

**核心思想：** 文本与图片嵌入到 **同一向量空间**，实现跨模态语义检索。

### 8.5.5 多模态文档中的图片处理（完整案例）

**处理流程（4 步）：**
1. LlamaParse 深度解析 PDF，分离提取图片并保存到本地。
2. Qwen-VL 模型理解图片 → 转换为文本。
3. 嵌入索引上述文本 → 构造查询引擎。

#### 1. 解析 PDF 文档（get_json_result + get_images）

```python
def load_docs():
    parser = LlamaParse(language='ch_sim', verbose=True)
    json_objs = parser.get_json_result("../../data/xiaomi14.pdf")
    json_list = json_objs[0]["pages"]
    image_list = parser.get_images(json_objs, download_path="pdf_images")
    return json_list, image_list
```

**JSON 对象结构：** 包含 `page`（页数）、`text`（文本）、`md`（Markdown）、`images`（图片信息）、`items`（明细项目）。

#### 2. 处理文本（构造 TextNode）

```python
from llama_index.core.schema import TextNode

def get_text_nodes(json_list: List[dict]):
    text_nodes = []
    for idx, page in enumerate(json_list):
        text_node = TextNode(
            text=page["text"],
            metadata={"page": page["page"]}
        )
        text_nodes.append(text_node)
    return text_nodes
```

#### 3. 处理图片（多模态大模型 → TextNode）

```python
def get_text_of_image(image_path):
    mm_llm = DashScopeMultiModal(model_name=DashScopeMultiModalModels.QWEN_VL_PLUS)
    image = load_local_images(["file://./" + image_path])
    chat_message_local = create_dashscope_multi_modal_chat_message(
        "请详细描述图片中的信息，包括图片中的文字和图像。",
        MessageRole.USER,
        image
    )
    chat_response = mm_llm.chat([chat_message_local])
    return chat_response.message.content[0]["text"]

def get_image_text_nodes(image_list: List[dict]):
    img_text_nodes = []
    for idx, image in enumerate(image_list):
        response = get_text_of_image(image["path"])
        text_node = TextNode(
            text=str(response),
            metadata={"path": image["path"]}
        )
        img_text_nodes.append(text_node)
    return img_text_nodes
```

#### 4. 构造主查询引擎

```python
def create_engine():
    (json_list, image_list) = load_docs()
    text_nodes = get_text_nodes(json_list)
    img_text_nodes = get_image_text_nodes(image_list)

    collection = chroma.get_or_create_collection(name="llamaparse_mm")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 文本 Node + 图片 Node 一起构造索引
    index = VectorStoreIndex(
        nodes=text_nodes + img_text_nodes,
        storage_context=storage_context
    )
    query_engine = index.as_query_engine(similarity_top_k=5, verbose=True)
    return query_engine
```

**核心模式：** 图片本质被转换为 TextNode（多模态大模型生成的描述文本），与文本 Node 混合索引，统一检索。

---

## 8.6 查询管道：编排基于 Graph 的 RAG 工作流

> 注意：在后续 LlamaIndex 版本中，查询管道将逐渐被 **Workflows** 替代。

### 8.6.1 理解查询管道

**查询管道（Query Pipeline）** 是 LlamaIndex 提供的 **声明式 API**，允许用更简洁方式将不同模块连接起来，编排从简单到复杂的 RAG 工作流。类似 LangChain 的 **LangGraph**。

**核心抽象：`QueryPipeline`。** 以它为核心加载组件（LLM、Prompt、检索器、合成器），以"图"的方式连接成简单顺序链（Chain）或有向无环图（DAG）。

**图（Graph）基本概念：**
- 图表示多个元素及其关系，任何两个元素都可联系，适合表达复杂关系。
- 图 = N 个 Node（顶点）+ 这些元素之间的关系（边）的集合。
- **有向无环图（DAG）：** "有向"指边有方向，"无环"指无法从某 Node 经若干边返回该 Node。

**优势：**
1. 用更少代码声明工作流，简洁灵活。
2. 更高的代码可读性。
3. 可与上层低代码/无代码方案集成。

**演进背景：** 模块化 RAG 演进带来更多工作模块（查询转换、路由、重排序、不同响应生成），需要灵活组装编排工作流的方案。

### 8.6.2 查询管道支持的两种使用方式

#### 1. 简单顺序链（Chain）

前一个模块输出作为下一个模块输入：

```python
from llama_index.core.query_pipeline import QueryPipeline, PromptTemplate

prompt_str = "请为产品设计一句简单的宣传语，我的产品是{product_name}"
prompt_tmpl = PromptTemplate(prompt_str)
llm = OpenAI()
p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)
```

#### 2. DAG

```python
from llama_index.core.query_pipeline import QueryPipeline, InputComponent

docs = SimpleDirectoryReader(input_files=["../../data/xiaomai.txt"]).load_data()
index = VectorStoreIndex.from_documents(docs)

# 准备组件
input = InputComponent()
llm = Ollama(model='qwen:14b')
prompt_tmpl = PromptTemplate("对问题进行完善，输出新的问题：{query_str}")
retriever = index.as_retriever(similarity_top_k=3)
summarizer = get_response_synthesizer(response_mode="tree_summarize")

# 构造查询管道
p = QueryPipeline(verbose=True)
p.add_modules({
    "input": input,
    "prompt": prompt_tmpl,
    "llm": llm,
    "retriever": retriever,
    "summarizer": summarizer,
})

# 连接模块（构成 DAG）
p.add_link("input", "prompt")
p.add_link('prompt', "llm")
p.add_link('llm', 'retriever')
p.add_link("retriever", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")

output = p.run(input='小麦手机的优势是什么')
```

**关键 API：**
- `add_modules({...})`：添加模块，key 是模块名（用于后续连接引用）。
- `add_link(src, dst, dest_key=...)`：连接模块，`dest_key` 指定目标模块接收输入的参数名。
- `run(input=...)`：从入口运行。

### 8.6.3 深入理解查询管道的内部原理

**核心信息（图 8-55）：**

1. **QueryComponent 统一类型：** 插入查询管道的组件被统一转换为 `QueryComponent` 类型，每个都实现统一的 `run_component` 接口。
2. **as_query_component 方法：** 每个可插入的功能组件（LLM、PromptTemplate、Retriever、NodePostprocessor、Synthesizer）都需实现 `as_query_component`，把自己转换为可组装进管道的 QueryComponent。
3. **内部 DAG：** 编排时保存一个 DAG 内部对象（由 QueryComponent Node + 连接关系组成），查询管道据此运行。**底层用 NetworkX**（Python 图结构库）实现。
4. **管道嵌套：** 查询管道本身也是 QueryComponent，可作为组件插入其他查询管道，实现嵌套调用。

**运行原理：**
- **编排时：** 添加功能组件 → 转换为 QueryComponent → 信息与连接关系存入内部 DAG。
- **运行时：** 根据 DAG 确定入口组件 → 调用 Node 的 `run_component` → 按 DAG 自动运行 → 直至末 Node（不再连接下游）或异常。

**可插入组件（表 8-1，重要）：**

| 基础类型 | 说明 | 输入信息 | 输出类型 |
|---------|------|---------|---------|
| LLM | LLM | prompt | CompletionResponse |
| PromptTemplate | 提示词模板 | 模板变量 | String |
| BaseQueryTransform | 转换器 | query_str | query_str |
| BaseRetriever | 检索器 | input | List[BaseNode] |
| BaseNodePostprocessor | 后处理器 | nodes / query_str | List[BaseNode] |
| BaseSynthesizer | 生成器 | nodes / query_str | Response |
| BaseOutputParser | 输出解析器 | input | 由输出解析器定义 |
| BaseQueryEngine | 查询引擎 | input | Response |
| QueryComponent | 其他查询管道 | input | 由查询管道定义 |

**注意：** 没有单独的路由组件，路由功能已在检索器/查询引擎中实现。

### 8.6.4 实现并插入自定义的查询组件

两种自定义方法。

#### 1. 从 CustomQueryComponent 派生（需实现 3 个接口）

```python
from llama_index.core.query_pipeline import CustomQueryComponent
from pydantic import Field, BaseModel
from llama_index.core.program import LLMTextCompletionProgram
from typing import List, Optional, Dict, Any

class MyOutputParser(CustomQueryComponent):
    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """校验组件输入参数（可不实现）"""
        return input

    @property
    def _input_keys(self) -> set:
        """定义输入 keys"""
        return {"response"}

    @property
    def _output_keys(self) -> set:
        """定义输出 keys"""
        return {"output"}

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """定义运行逻辑"""
        class Phone(BaseModel):
            name: str
            cpu: str
            memory: str
            storage: str
            screen: str
            battery: str
            features: List[str]

        prompt_template_str = """根据以下内容提取结构化信息{input}"""
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=Phone,
            prompt_template_str=prompt_template_str,
            verbose=True,
        )
        output = program(input=kwargs['response'])
        return {"output": output}   # 必须包含 _output_keys 中定义的 key
```

**3 个必须实现的接口：**
- `_input_keys`：输入参数名集合，运行时输入，通过 `add_link` 的 `dest_key` 指定。
- `_output_keys`：输出参数名集合，运行完成后输出。
- `_run_component(**kwargs)`：运行逻辑，接收 `_input_keys` 参数，输出含 `_output_keys` 的 dict。

**插入管道：**

```python
p = QueryPipeline(verbose=True)
p.add_modules({
    "input": input,
    "prompt": prompt_tmpl,
    "llm": llm,
    "retriever": retriever,
    "summarizer": summarizer,
    'output_parser': MyOutputParser()   # 自定义模块
})

p.add_link("input", "prompt")
p.add_link('prompt', "llm")
p.add_link('llm', 'retriever')
p.add_link("retriever", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")
p.add_link("summarizer", "output_parser", dest_key="response")   # 别忘了连接

output = p.run(input='小麦手机的优势是什么')
```

#### 2. 使用 FnComponent（轻量级）

用函数快速构造组件，函数主体是运行逻辑，函数参数即组件输入：

```python
from llama_index.core.query_pipeline import FnComponent

def addExtaInfo(phone: Phone) -> str:
    phone_info = f"Name: {phone.name}\nCPU: {phone.cpu}\n..."
    extra_info = "Extra information: This phone has a great camera."
    return f"{phone_info}\n{extra_info}"

# 包装成组件
'post_processor': FnComponent(fn=addExtaInfo, output_key="output")

# 连接
p.add_link("output_parser", "post_processor", dest_key="phone")
```

**适用：** 逻辑简单的轻量级组件，比派生类更简洁。

---

## 面试考点

### Q1：什么是 HyDE？为什么有效？实现要点是什么？
**要点：**
- HyDE = Hypothetical Document Embeddings，先让 LLM 根据问题生成"假设性答案文档"，再用该假设文档做嵌入/检索。
- **有效性原理**：query 与 answer 之间存在语义鸿沟，而 doc 与 doc 在向量空间更接近（检索空间一致性）。
- **实现要点**：`HyDEQueryTransform` 转换结果放在 `query_bundle.custom_embedding_strs`（**不是 query_str**）；建议用 `TransformQueryEngine` 透明包装。

### Q2：多步查询 vs 子问题查询的区别？
**要点：**
- 多步查询：基于前一步推理结果生成下一步问题，**迭代式**；用 `MultiStepQueryEngine` + `StepDecomposeQueryTransform`，需提供 `index_summary`。
- 子问题查询：参考 **可用工具** 一次性生成多个子问题，更具约束性；用 `SubQuestionQueryEngine` + `OpenAIQuestionGenerator`（或 `LLMQuestionGenerator`）。
- 与 Agent 区别：子问题引擎 **一次性** 规划，Agent **动态** 规划（ReAct）。

### Q3：为什么需要 Rerank？Rerank 模型与 embedding 模型有什么区别？
**要点：**
- 需要 Rerank 的原因：① 非向量索引需独立重排；② 混合检索需统一重排；③ 向量检索受嵌入模型/算法/领域影响排序有偏差。
- **区别**：embedding 模型是 bi-encoder（query 和 doc 分别编码，速度快但精度有限）；Rerank 模型是 **cross-encoder**（query 与 doc 拼接联合编码，精度高但成本高，仅用于小规模候选集精排）。
- 典型：Cohere Rerank（闭源）、bge-reranker-large（开源，TEI 部署）。

### Q4：small-to-big 检索是什么？如何实现？
**要点：**
- 思想：**小粒度嵌入（精准）、大粒度生成（完整上下文）**。
- 实现：`SentenceWindowNodeParser`（小句子作为 Node text 嵌入，窗口存元数据）+ `MetadataReplacementPostProcessor`（检索后用窗口内容替换 text）。

### Q5：语义路由的两种 Selector 区别？单选 vs 多选？
**要点：**
- Pydantic Selector：依赖 OpenAI 函数调用。
- 通用 LLM Selector：组装信息进 Prompt 让大模型选择。
- `LLMSingleSelector`（单选）vs `LLMMultiSelector`（多选，可同时路由到多个引擎并合并）。

### Q6：Text-to-SQL 中如何处理大型数据库？
**要点：**
- 问题：`NLSQLTableQueryEngine` 把所有表 Schema 塞进 Prompt，表过多导致上下文溢出/干扰/成本高。
- 方案：Text-to-SQL **之前先检索相关表**，用 `SQLTableRetrieverQueryEngine` + `ObjectIndex`（对 `SQLTableSchema` 对象建向量索引）+ `as_retriever(similarity_top_k=N)`。
- **ObjectIndex 是通用模式**：对任意 Python 对象构建向量索引，纳入 RAG 流程。

### Q7：多模态 PDF 中表格如何处理才能精确召回？
**要点：**
- 直接嵌入 Markdown 表格语义不足，召回不精确。
- 用 `MarkdownElementNodeParser`：对表格用 LLM 生成摘要/标题 → 构造 **IndexNode** → 用 IndexNode 检索（精度高）→ 递归检索原始表格 Node 用于生成（信息全）。
- `get_nodes_and_objects` 分离 base_nodes 与 objects。

### Q8：QueryPipeline 是什么？内部原理？
**要点：**
- 声明式 API，以 Graph（Chain / DAG）方式编排 RAG 工作流，类似 LangGraph。
- 内部：所有组件统一转换为 `QueryComponent`（实现 `run_component` 接口），用 NetworkX 维护 DAG。
- 组件需实现 `as_query_component`；管道本身也是 QueryComponent，可嵌套。
- 两种自定义：`CustomQueryComponent`（实现 `_input_keys`/`_output_keys`/`_run_component`）或 `FnComponent`（包装函数）。

---

## 易错 / 陷阱

1. **HyDE 输出位置陷阱**：转换结果在 `query_bundle.custom_embedding_strs`，**不是** `query_str`。直接打印 `query_str` 看不到变化，会误以为没生效。

2. **简单查询转换不建议在检索前用**：缺上下文时不确定性大，易偏离意图。建议用于数据准备阶段（生成相似问题、生成假设性查询用于嵌入）。

3. **KeywordNodePostprocessor 中文陷阱**：内置用 spacy（默认英文，支持词形还原），中文必须指定 `lang='zh-Hans'`，否则关键词匹配失效。

4. **子问题生成器的模型依赖**：`OpenAIQuestionGenerator` 依赖 **函数调用功能**，非函数调用模型要用 `LLMQuestionGenerator`。

5. **Rerank 前必须先召回候选**：Rerank 是 **精排**，需先用 retriever 召回 top_k（如 5~10），再 rerank 到 top_n（如 2~3）。Rerank 模型成本高，不宜在大候选集上跑。

6. **bge-reranker 需自定义处理器**：LlamaIndex 没有内置 bge-rerank 处理器，需继承 `BaseNodePostprocessor` 自己实现 `_postprocess_nodes`，调用 TEI 的 `/rerank` 接口。

7. **SQL 查询引擎的上下文溢出**：`NLSQLTableQueryEngine` 把所有表 Schema 组装进 Prompt，大库会溢出 → 必须用 `SQLTableRetrieverQueryEngine` + ObjectIndex 检索表。

8. **多模态表格直接嵌入效果差**：LlamaParse 解析出 Markdown 表格直接嵌入，语义不足，事实性数据查询易错 → 必须用 `MarkdownElementNodeParser` 增强。

9. **QueryPipeline add_link 别忘了 dest_key**：当目标组件有多个输入参数（如 summarizer 需 `nodes` 和 `query_str`），必须用 `dest_key` 指定每个连接对应哪个参数。

10. **路由组件不在 QueryPipeline 中**：QueryPipeline 没有单独路由组件，路由功能已在检索器/查询引擎中实现。

11. **自定义 CustomQueryComponent 必须返回 _output_keys 中定义的 key**：`_run_component` 返回的 dict 必须包含所有 `_output_keys` 声明的输出 key。

12. **DashScope API Key 变量名**：书中代码用的是 `DAHSCOPE_API_KEY`（注意是 DAH 不是 DA），实际应核对阿里云文档使用正确的环境变量名。
