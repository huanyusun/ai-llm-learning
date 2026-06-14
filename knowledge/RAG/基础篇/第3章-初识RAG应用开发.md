# 第3章 初识 RAG 应用开发

> 来源：严灿平《基于大模型的 RAG 应用开发与优化——构建企业级 LLM 应用》【基础篇】第3章
> 定位：把第1～2章的架构与工作流思想转换为"可运行、可生产"的软件。从开发、跟踪与调试 RAG 应用入手，在技术层建立对 RAG 应用开发的初步认识。
> 适用读者：具备 Python 基础、希望系统掌握企业级 RAG 工程化开发的程序员。

---

## 0. 本章导览与核心脉络

- **目标**：从"符合 RAG 经典范式的原型应用"走向"满足企业级业务需求与工程化要求的生产就绪应用"。
- **生产就绪的两个硬要求**：
  1. 满足复杂多变的功能需求；
  2. 投产前完成完整的测试与评估，并借助模块/算法持续优化。
- **本章三条主线**：
  1. **3.1** 用三种方式（原生代码 / LlamaIndex / LangChain）开发同一个最简 RAG 应用，对比理解原理与框架价值；
  2. **3.2** RAG 应用的跟踪与调试（LlamaDebugHandler + 第三方平台 Langfuse）；
  3. **3.3** LlamaIndex 核心组件学习路线图（为后续章节铺路）。

---

## 3.1 开发一个最简单的 RAG 应用

### 3.1.0 统一技术场景设计

为了让原理对照公平，本章用 **3 种不同方法实现同一个 RAG 应用**：

- （1）使用原生代码开发；
- （2）使用 LlamaIndex 框架开发；
- （3）使用 LangChain 框架开发。

**场景定义**：从本地目录加载若干 TXT 文档（模拟私有知识）→ 分割 → 嵌入 → 存入向量库 Chroma → 借助大模型做增强生成。

**目录结构（图 3-1 对应）**：

| 目录/文件 | 作用 |
|---|---|
| `chroma_db/` | 向量库 Chroma 的持久化存储目录 |
| `data/` | 存放代表本地知识的 TXT 文档（如文心一言简介） |
| `src/` | RAG 应用的 Python 代码目录 |

---

### 3.1.1 使用原生代码开发

不借助任何第三方 RAG 框架，**自行实现 RAG 必备模块**：加载读取文档、分割文档、嵌入、向量存储与索引、检索与生成，并模拟最终用户的查询/对话。

#### (1) 加载与读取文档

定义 TXT 加载器：路径检查 → 类型判断 → 读取内容输出。

```python
# 加载与读取文档
import mimetypes
import os, configparser

def loadtext(path):
    path = path.rstrip()
    path = path.replace(' \n', '')

    # 转换绝对路径
    filename = os.path.abspath(path)

    # 判断文档存在，并获得文档类型
    filetype = ''
    if os.path.isfile(filename):
        filetype = mimetypes.guess_type(filename)[0]
    else:
        print(f"File {filename} not found")
        return None

    # 读取文档内容
    text = ""
    if filetype != 'text/plain':
        return None
    else:
        with open(filename, 'rb') as f:
            text = f.read().decode('utf-8')

    return text

# 配置器：读取模型名称等配置
def getconfig():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return dict(config.items("main"))
```

**要点**：
- `mimetypes.guess_type` 用于按文件后缀推断 MIME 类型，这里只接受 `text/plain`；
- `getconfig()` 把模型名等配置从 `config.ini` 解耦，便于切换模型。

#### (2) 分割文档（Chunking）

**为什么不能整篇嵌入**：基于向量的语义检索要求把文档切成大小合适的"知识块（Chunk）"，整篇嵌入会稀释语义、超出模型上下文窗口。**不能简单用固定字符长度硬切**，那样会切断句子、破坏语义完整性。

本例策略：在"换行/句末"处切分句子，每个 Chunk 放若干句。

```python
# 把文档分割成知识块
import jieba, re
from typing import List

def split_text_by_sentences(source_text: str,
                            sentences_per_chunk: int,
                            overlap: int) -> List[str]:
    """
    简单地把文档分割为多个知识块，每个知识块都包含指定数量的句子
    """
    if sentences_per_chunk < 2:
        raise ValueError("一个句子至少要有 2 个 chunk！")
    if overlap < 0 or overlap >= sentences_per_chunk - 1:
        raise ValueError("overlap 参数必须大于等于 0，且小于 sentences_per_chunk")

    # 用正则表达式按中文句末标点分割句子
    sentences = re.split('(?<=[。！？])\s+', source_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']

    if not sentences:
        print("Nothing to chunk")
        return []

    # 处理 overlap 参数（相邻 Chunk 间重叠若干句）
    chunks = []
    i = 0
    while i < len(sentences):
        end = min(i + sentences_per_chunk, len(sentences))
        chunk = ' '.join(sentences[i:end])

        if overlap > 0 and i > 1:
            overlap_start = max(0, i - overlap)
            overlap_end = i
            overlap_chunk = '\n'.join(sentences[overlap_start:overlap_end])
            chunk = overlap_chunk + ' ' + chunk

        chunks.append(chunk.strip())
        i += sentences_per_chunk

    return chunks
```

**`overlap` 参数机制（关键知识点）**：
- 是什么：允许相邻知识块之间有重叠部分（这里以"重叠句子数量"实现）。
- 为什么：避免在 Chunk 边界处丢失上下文，提升语义覆盖与检索召回。
- 约束：`0 <= overlap < sentences_per_chunk - 1`（overlap 等于块大小减一会退化为完全重复）。

> **结论**：分割是 RAG 中影响检索/生成质量的关键阶段，块大小、切分方式都会显著影响最终效果（后续章节会深入）。

#### (3) 嵌入、向量存储与索引

索引准备的最后阶段：用嵌入模型把 Chunk 转成向量 → 存入 Chroma → 建立索引。

```python
import ollama, chromadb

# 引入自定义模块
from load import loadtext, getconfig
from splitter import split_text_by_sentences

# 向量模型（从 config.ini 读取）
embedmodel = getconfig()["embedmodel"]

# 向量库（HTTP 模式的 Chroma 服务）
chroma = chromadb.HttpClient(host="localhost", port=8000)
chroma.delete_collection(name="ragdb")
collection = chroma.get_or_create_collection(name="ragdb")

# 读取文档列表，依次处理
with open('docs.txt') as f:
    lines = f.readlines()
    for filename in lines:
        # 加载文档内容
        text = loadtext(filename)

        # 把文档分割成知识块
        chunks = split_text_by_sentences(source_text=text,
                                         sentences_per_chunk=8,
                                         overlap=0)

        # 对知识块依次处理
        for index, chunk in enumerate(chunks):
            # 借助 Ollama 部署的本地嵌入模型生成向量
            embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']

            # 存储到向量库 Chroma，注意这里的参数
            collection.add(
                [filename + str(index)],
                [embed],
                documents=[chunk],
                metadatas={"source": filename}
            )
```

**设计要点**：
- 文档清单放在独立 `docs.txt` 里依次处理，而非直接扫 `data/` 目录——目的是**方便扩展多源数据**（如配置一个 URL 从网络加载 HTML）。
- `collection.add(ids, embeddings, documents, metadatas)`：`ids` 唯一标识每个 Chunk，`metadatas` 可用于过滤与溯源。

**`config.ini` 模型配置**：

```ini
[main]
embedmodel=milkey/dmeta-embedding-zh:f16
mainmodel=qwen:32b
```

**索引正确性自测**（在索引代码后追加）：

```python
if __name__ == "__main__":
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'quit':
            break
        else:
            # 从 Chroma 查询与向量相似的知识块
            results = collection.query(
                query_embeddings=[ollama.embeddings(model=embedmodel, prompt=query)['embedding']],
                n_results=3
            )
            for result in results["documents"][0]:
                print("----------------------------------------------------")
                print(result)
```

#### (4) 检索与生成

实现交互式查询：输入问题 → 语义检索相关 Chunk → 作为上下文组装 Prompt → 大模型生成答案。

```python
import ollama, sys, chromadb
from load import getconfig

# 嵌入模型与大模型
embedmodel = getconfig()["embedmodel"]
llmmodel = getconfig()["llmmodel"]

# 向量库
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection("ragdb")

while True:
    query = input("Enter your query: ")
    if query.lower() == 'quit':
        break
    else:
        # 1) 生成查询向量
        queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']

        # 2) 用查询向量检索上下文
        relevantdocs = collection.query(query_embeddings=[queryembed], n_results=5)["documents"][0]
        docs = "\n\n".join(relevantdocs)

        # 3) 生成 Prompt（含防幻觉约束）
        modelquery = f"""
        请基于以下的上下文回答问题，如果上下文中不包含足够的回答问题的信息，请回答'我暂时无法回答该问题'，不要编造。

        上下文：
        ====
        {docs}
        ====

        我的问题是：{query}
        """

        # 4) 交给大模型生成（流式）
        stream = ollama.generate(model=llmmodel, prompt=modelquery, stream=True)
        for chunk in stream:
            if chunk["response"]:
                print(chunk['response'], end='', flush=True)
```

**检索与生成阶段的 4 步处理逻辑（必背）**：

| 步骤 | 动作 | 关键 API |
|---|---|---|
| ① | 用嵌入模型把问题转成查询向量 | `ollama.embeddings(model, prompt)` |
| ② | 用查询向量做语义检索，取回相关 Chunk | `collection.query(query_embeddings, n_results)` |
| ③ | 把 Chunk 与原问题组装进 Prompt | 字符串格式化 / 模板 |
| ④ | 调大模型生成接口输出 | `ollama.generate(model, prompt, stream)` |

> `collection.query` 常用参数：检索向量、返回数量 `n_results`、近似算法参数等；大模型 `generate` 常用参数：模型名、prompt、是否流式、temperature 等。

#### (5) 原型应用的局限性——通往生产级应用的 7 个真实问题

朴素 RAG 原型与"企业级应用"差距巨大，需思考：

1. **索引与使用分离/异步**：知识索引准备往往是离线、异步过程，而非顺序执行——如何实现？
2. **可视化管理**：知识获取/加载/索引是否需要功能完整的可视化管理工具？
3. **文档↔Chunk 对应**：原始文档与 Chunk 如何对应？后续知识维护与更新如何同步？
4. **外部系统集成**：检索/生成过程中可能需通过 API 对接外部系统——如何实现？
5. **并发**：管理与使用需支持多用户并发而非单用户——如何实现？
6. **UI 化与历史记录**：用户基于 UI 页面而非命令行交互；如何保存交互历史？
7. **溯源**：如何从生成结果溯源到参考的 Chunk / 文档？

> 原生代码的价值在于**从底层看清 RAG 基本原理**，便于后续理解框架与高阶优化；但其鲁棒性、可扩展性、可维护性都太初级，无法真实投产——这正是引入 LlamaIndex 等框架的理由。

---

### 3.1.2 使用 LlamaIndex 框架开发

#### (1) 经典 5 行代码 RAG 入门应用

```python
# 经典的 5 行代码的 RAG 应用

# 加载文档
documents = SimpleDirectoryReader("../data").load_data()

# 构造向量存储索引
index = VectorStoreIndex.from_documents(documents)

# 构造查询引擎
query_engine = index.as_query_engine()

# 对查询引擎提问
response = query_engine.query('这里放入 data 目录中知识相关的问题')

# 输出答案
print(response)
```

**封装组件分析（屏蔽了大量细节）**：

| 组件 | 作用 |
|---|---|
| `SimpleDirectoryReader` | 从目录（`../data`）加载读取知识 |
| `VectorStoreIndex` | 对加载的知识做嵌入与索引 |
| `query_engine`（`index.as_query_engine()`） | 自动完成检索与生成 |

**默认配置（无显式配置时）**：

- 大模型：OpenAI GPT-3.5-Turbo
- 向量库：内存
- 嵌入模型：OpenAI `text-embedding-3-small`

> 默认走 OpenAI 系列，因此必须配置 `OPENAI_API_KEY`。若使用 **ONE-API** 这类统一接口分发平台：把 `OPENAI_API_BASE` 指向分发服务地址，`OPENAI_API_KEY` 设为平台提供的 Key。

#### (2) 用 LlamaIndex 重写 3.1.1 的应用（完整版）

```python
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 1) 设置模型（Settings 是 LlamaIndex 全局设置组件；不设默认用 OpenAI）
Settings.llm = Ollama(model="qwen:14b")
Settings.embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")

# 2) 加载与读取文档（Document 是 LlamaIndex 中代表知识文档的对象类型）
reader = SimpleDirectoryReader(input_files=["../../data/yiyan.txt", "../../data/HR.txt"])
documents = reader.load_data()

# 3) 分割文档（SentenceSplitter 输出 Node 对象——LlamaIndex 中代表知识块的对象类型）
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

# 4) 准备向量存储（Chroma collection 类似关系库的一个库，再包成 ChromaVectorStore）
chroma = chromadb.HttpClient(host="localhost", port=8000)
chroma.delete_collection(name="ragdb")
collection = chroma.get_or_create_collection(name="ragdb", metadata={"hnsw:space": "cosine"})
vector_store = ChromaVectorStore(chroma_collection=collection)

# 5) 准备向量存储索引（LlamaIndex 会自动对 Node 嵌入并存入 Chroma，无须额外编码）
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# 6) 构造查询引擎（一行完成检索+生成）
query_engine = index.as_query_engine()

while True:
    user_input = input("问题：")
    if user_input.lower() == "exit":
        break
    response = query_engine.query(user_input)
    print("AI 助手：", response.response)
```

**6 步详解**：

1. **设置模型**：通过 `Settings` 全局设置大模型与嵌入模型（不设则默认 OpenAI）。此处用 Ollama 本地部署的 `qwen:14b` 与 `dmeta-embedding-zh:f16`。
2. **加载读取**：`SimpleDirectoryReader` 加载 TXT，产出 `Document` 对象。
3. **分割**：`SentenceSplitter` 直接把文档切成 `Node` 对象，无需自写切分逻辑。`chunk_size=500`、`chunk_overlap=20`。
4. **准备向量存储**：构造 Chroma `collection`（可指定距离度量 `hnsw:space: cosine`），再用 `ChromaVectorStore` 包装。
5. **构造索引**：用 `Node` + `ChromaVectorStore` 构造 `VectorStoreIndex`，框架自动完成嵌入与入库。
6. **构造查询引擎**：`index.as_query_engine()` 一行搞定检索+生成。

**核心数据抽象对照**：

| 概念 | LlamaIndex 类型 | 含义 |
|---|---|---|
| 知识文档 | `Document` | 一次加载的完整文档 |
| 知识块 | `Node` | 切分后的最小检索单元 |

**框架价值（远超节省 80% 代码量）**：带来灵活性、可扩展性、可维护性。可灵活变更：加载格式、大模型/向量存储、切分方式与参数——而不会因频繁变更产生难以维护的"空心粉"式代码。

---

### 3.1.3 使用 LangChain 框架开发

本书主用 LlamaIndex，但 LangChain 同为主流框架，对照学习有助于理解框架共性能力。

```python
import chromadb
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# 模型
llm = Ollama(model="qwen:14b")
embed_model = OllamaEmbeddings(model="milkey/dmeta-embedding-zh:f16")

# 加载与读取文档
loader = DirectoryLoader('../../data/',
                         glob="*.txt", exclude="*tips*.txt", loader_cls=TextLoader)
documents = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
splits = text_splitter.split_documents(documents)

# 准备向量存储
chroma = chromadb.HttpClient(host="localhost", port=8000)
chroma.delete_collection(name="ragdb")
collection = chroma.get_or_create_collection(name="ragdb", metadata={"hnsw:space": "cosine"})
db = Chroma(client=chroma, collection_name="ragdb", embedding_function=embed_model)

# 存储到向量库中，构造索引
db.add_documents(splits)

# 使用检索器
retriever = db.as_retriever()

# 用 LCEL（LangChain 表达语言）构造 RAG"链"
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    user_input = input("问题：")
    if user_input.lower() == "exit":
        break
    response = rag_chain.invoke(user_input)
    print("AI 助手：", response)
```

**LangChain vs LlamaIndex 差异（重点）**：

二者在"加载读取 / 分割 / 设置向量库"部分大同小异，主要差异：

1. **构造索引方式不同**：LangChain 可直接用向量库对象的 `add_documents` 构造索引，**无须独立的 `VectorStoreIndex` 对象**。
2. **核心组件 Chain + LCEL**：LangChain 最核心的组件之一是 **Chain**。检索与生成阶段用 LCEL（LangChain Expression Language，LangChain 表达语言）构造 `rag_chain`：从原始问题 → 检索 → 组装提示 → 调用大模型 → 输出解析，一系列链式动作用一个表达式完成，其余交给框架。

---

## 3.2 如何跟踪与调试 RAG 应用

> **问题背景**：LlamaIndex 高度抽象与封装，隐藏了大量运行时底层细节，给排障与调优带来不便——例如需要直接观察大模型的真实输入/输出，以判断检索精确性或模型输出能力。需要简单易用的框架内部跟踪机制。

### 3.2.1 借助 LlamaDebugHandler

#### 原理机制

- LlamaIndex 允许在构造组件时指定多个**回调类（Callback）**。
- 回调类在框架定义的**关键事件（开始/结束）**发生时被调用，记录跟踪信息：执行步骤、时间戳、输入信息、输出信息等。
- `LlamaDebugHandler` 是**专用于记录调试信息**的回调类。
- 所有回调类通过全局 `Settings.callback_manager` 集中管理——构造 `LlamaDebugHandler` 后挂到 `callback_manager`，即可全局或针对特定组件生效。

#### (1) 设置调试处理器

```python
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)

# 构造 LlamaDebugHandler
# print_trace_on_end：是否在每次事件结束时立即打印简单跟踪信息
llama_debug = LlamaDebugHandler(print_trace_on_end=True)

# 加入 CallbackManager 并设为全局
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager
```

也可在**组件级别**设置，例如构造索引时：

```python
index = VectorStoreIndex.from_documents(
    docs, callback_manager=callback_manager
)
```

#### (2) LlamaIndex 关键事件类型（CBEventType）

| 事件 | 含义 |
|---|---|
| `CHUNKING` | 文本分割事件 |
| `NODE_PARSING` | Node 解析事件 |
| `EMBEDDING` | 文本嵌入事件 |
| `LLM` | 调用大模型事件 |
| `QUERY` | 通过 query 引擎调用 query 方法事件 |
| `RETRIEVE` | 语义检索相关知识事件 |
| `SYNTHESIZE` | 组装 Prompt 并用大模型生成结果事件 |
| `TREE` | 生成文本摘要信息事件 |
| `SUB_QUESTION` | 生成子问题事件 |

> 注意：事件类型随版本演进可能调整，需查官方文档。

#### (3) 使用跟踪与调试信息

```python
import pprint

# 获得某事件的"时间信息"（含时间戳、耗时）
pprint.pprint(llama_debug.get_event_time_info(CBEventType.QUERY))

# 获得详细的事件跟踪信息（事件开始/结束时框架记录的输入输出）
pprint.pprint(llama_debug.get_event_pairs(CBEventType.QUERY))

# 打印完整的事件发生堆栈及耗时信息
llama_debug.print_trace_map()
```

- `get_event_time_info` → 事件时间信息（图 3-4）。
- `get_event_pairs` → 事件开始/结束时的详细跟踪信息（图 3-5）。
- `print_trace_map` → 完整事件发生堆栈与耗时（图 3-6）。

---

### 3.2.2 借助第三方的跟踪与调试平台

#### 背景

随着大模型应用涌现，出现了大量帮助应用"生产就绪"的**工程化平台**，主要用于：跟踪、调试、测试、评估、管理数据集。典型如 LangChain 公司的 **LangSmith**；LlamaIndex 也得到大量第三方平台支持。本节介绍开源平台 **Langfuse**。

#### Langfuse 是什么

- **开源**的大模型工程化平台，提供：跟踪、评估、提示（Prompt）管理等功能，帮助应用尽快投产。
- 包含**前后端的完整平台**：应用通过简单 SDK 无缝生成跟踪信息与性能指标，自动发送到 Langfuse 服务端；用户借助其前端 UI 直观查看。
- 可用 **Langfuse Cloud**（图 3-7）或**本地自建**（条件允许时推荐本地自建）。

#### (1) 申请 API Key

打开前端服务网站（在线或本地），注册登录后申请 API Key（图 3-8）。

#### (2) 应用开发集成（与 LlamaDebugHandler 思路一致）

```python
import os
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler

# 设置 Langfuse 平台的 API Key
os.environ["LANGFUSE_SECRET_KEY"] = "sk-****"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-****"
os.environ["LANGFUSE_HOST"] = "https://xx.xx.xx"

# 构造 Langfuse 平台的回调类
langfuse_callback_handler = LlamaIndexCallbackHandler()

# 设置到全局的 callback_manager
Settings.callback_manager = CallbackManager([langfuse_callback_handler])

# ......应用逻辑......

# 程序退出之前注意 flush，将缓存的跟踪信息发送到 Langfuse Server 端
langfuse_callback_handler.flush()
```

> **易错点**：必须在程序退出前调用 `flush()`，否则缓存的跟踪信息可能未发送到服务端而丢失。

#### (3) 使用 Langfuse UI 观察

完成集成后，应用各类事件（如 Index、Query）的跟踪信息与性能指标会被自动上报，可在 Langfuse UI 查看（图 3-9）。点击某次查询可看：

- 查询执行过程及每一步内部细节；
- 检索出的参考知识块（图 3-10）；
- 一次大模型调用的完整输入和输出（图 3-11）。

#### Langfuse 的其他能力

Prompt 模板管理、各类性能指标生成、大模型调用成本统计跟踪、评估数据集管理与在线评估等。

---

## 3.3 准备：基于 LlamaIndex 框架的 RAG 应用开发核心组件

### LlamaIndex 组件体系的设计哲学

- 预置大量封装良好、符合设计模式的 RAG 开发组件，覆盖**不同阶段与不同场景**。
- 组件**不是孤立模块的简单堆积**，而是通过组合、派生与集成形成**完整的应用集成框架**。
- 灵活性：既可用高度集成的上层 API 快速开发，也可用底层 API 实现灵活/复杂控制，或在内置组件上**派生自定义新组件**。
- 代价：强大灵活 → 一定的复杂性与学习门槛。

### 学习路线图（图 3-12）

本章按 **RAG 典型流程**对核心组件做分类，按此结构逐层学习，最终具备开发端到端 RAG 应用的能力。后续章节将依此路线图展开（数据加载与分割、模型与 Prompt、索引、检索、生成、评估与优化等）。

---

## 面试考点

### Q1：简述 RAG 应用的核心工作流程与必备模块。
**答**：① 加载与读取文档 → ② 分割文档为知识块（Chunk） → ③ 嵌入（Embedding）为向量 → ④ 向量存储与索引 → ⑤ 检索（基于查询向量做语义相似检索） → ⑥ 生成（把相关 Chunk 作为上下文组装 Prompt，交给大模型生成答案）。必备模块对应这六步。

### Q2：Chunking 为什么要做 overlap（重叠）？overlap 取值有什么约束？
**答**：overlap 让相邻知识块共享部分内容，避免在块边界处丢失上下文，提升语义覆盖和检索召回。约束：`0 <= overlap < chunk_size - 1`（按句/按 token 皆然），过大会退化为近似完全重复、增加存储与冗余。

### Q3：朴素 RAG 原型为什么不能直接上生产？至少说出 4 点。
**答**：① 索引与查询需分离/异步而非顺序；② 缺乏可视化的知识管理工具；③ 原始文档↔Chunk 缺少对应关系，知识更新难同步；④ 缺少与外部系统的 API 集成；⑤ 不支持多用户并发；⑥ 缺 UI 与历史记录持久化；⑦ 缺结果溯源（到 Chunk/文档）；⑧ 鲁棒性、可扩展性、可维护性差。

### Q4：LlamaIndex 的两个核心数据抽象是什么？
**答**：`Document`（代表一次加载的完整知识文档）与 `Node`（代表切分后的最小知识块/检索单元）。`SentenceSplitter.get_nodes_from_documents(documents)` 完成 `Document → Node` 的转换。

### Q5：LlamaIndex 默认使用什么模型/向量库？如何切换到本地 Ollama 模型？
**答**：默认大模型 OpenAI GPT-3.5-Turbo、嵌入 OpenAI text-embedding-3-small、向量库内存存储。切换：通过全局 `Settings` 设置——`Settings.llm = Ollama(model=...)`、`Settings.embed_model = OllamaEmbedding(model_name=...)`。也可在具体组件构造时通过 `llm=` / `embed_model=` 参数动态插入。

### Q6：LlamaIndex 与 LangChain 在构造 RAG 索引/链时的主要差异？
**答**：① LangChain 可直接用向量库对象 `add_documents` 构造索引，无须独立的 `VectorStoreIndex`；② LangChain 最核心组件是 Chain，检索与生成用 LCEL（LangChain Expression Language）构造 `rag_chain`，以管道表达式串联 retriever → prompt → llm → output_parser，声明式完成链式调用。LlamaIndex 则用 `index.as_query_engine()` 等高阶 API 封装。

### Q7：如何在不破坏框架的前提下观察 RAG 内部细节（检索结果、LLM 真实输入输出）？
**答**：① LlamaIndex 内置回调机制——构造 `LlamaDebugHandler(print_trace_on_end=True)`，挂到 `Settings.callback_manager`，再用 `get_event_pairs(CBEventType.RETRIEVE/LLM/QUERY)`、`print_trace_map()` 等查看；② 接入第三方平台如 Langfuse（开源）/ LangSmith，通过 `LlamaIndexCallbackHandler` 自动上报，UI 查看。注意退出前 `flush()`。

### Q8：列出 LlamaIndex 中至少 6 种 CBEventType 事件。
**答**：`CHUNKING`（分割）、`NODE_PARSING`（Node 解析）、`EMBEDDING`（嵌入）、`LLM`（大模型调用）、`QUERY`（query 引擎查询）、`RETRIEVE`（语义检索）、`SYNTHESIZE`（组装 Prompt 并生成）、`TREE`（摘要）、`SUB_QUESTION`（子问题生成）。

---

## 易错 / 陷阱

1. **`overlap` 边界条件**：`overlap` 必须 `>= 0` 且 `< sentences_per_chunk - 1`，否则抛 `ValueError`；写错会让索引阶段整批失败。
2. **原生代码 `collection.add` 参数顺序**：`add(ids, embeddings, documents, metadatas)` 是位置参数，顺序错乱会导致向量与文档/元数据错位，检索结果全错且无报错。
3. **LlamaIndex 默认走 OpenAI**：不配置 `Settings.llm` / `Settings.embed_model` 时默认调用 OpenAI，必须设 `OPENAI_API_KEY`；用 ONE-API 类分发平台时别忘了改 `OPENAI_API_BASE`。最常见的"第一次跑就报 401/连不上"问题。
4. **`SimpleDirectoryReader` 默认非递归 / 对格式敏感**：放错目录或文件类型不被识别会得到空 `documents`，后续索引为空但程序不报错，检索永远召不回。
5. **`chunk_size` 与上下文窗口不匹配**：块过大 → 单 Chunk 超嵌入/生成窗口；块过小 → 语义破碎、检索召回差。中文还要注意 `SentenceSplitter` 的 token 计数方式（按字符近似）。
6. **Chroma collection 残留**：重复运行若不先 `delete_collection` 或换名，会与旧数据混叠，造成"明明改了文档结果却没变"。
7. **`hnsw:space` 距离度量**：Chroma 默认 `l2`，若嵌入模型推荐 `cosine`（多数文本嵌入如此）而未在 `metadata={"hnsw:space":"cosine"}` 指定，相似度排序会失真。
8. **Langfuse 忘记 `flush()`**：程序提前退出会导致缓存的 trace 未上报，UI 上看不到任何记录，误以为集成失败。
9. **流式输出未 `flush=True`**：原生 `print(..., end='', flush=True)` 漏掉 `flush` 在某些终端会"卡住"看不到逐字输出。
10. **把"原生代码能跑"当成"能上线"**：朴素原型缺少并发、异步、溯源、UI、持久化等工程化能力，生产化必须用框架 + 补齐工程基础设施。

---

## 本章小结

- 用**原生代码**走通 RAG 六步，建立底层直觉；
- 用 **LlamaIndex**（主框架）把代码量降到约 20%，并获得灵活/可扩展/可维护；
- 用 **LangChain** 对照理解 Chain + LCEL 的声明式编排；
- 用 **LlamaDebugHandler** 与 **Langfuse** 解决"框架封装过深、看不见内部"的调试痛点；
- 第 3.3 节给出后续章节的学习路线图：沿 RAG 典型流程逐层展开 LlamaIndex 核心组件。
