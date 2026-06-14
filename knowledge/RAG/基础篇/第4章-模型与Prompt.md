# 第4章 模型与 Prompt

> 来源：严灿平《基于大模型的 RAG 应用开发与优化——构建企业级 LLM 应用》【基础篇】第4章
> 定位：RAG 应用涉及的两类核心模型（大模型、嵌入模型）及其统一接口、单独/集成使用、参数、自定义组件；以及贯穿全程的 Prompt 模板机制。
> 适用读者：希望系统掌握 LlamaIndex 模型抽象与 Prompt 工程的程序员。

---

## 0. 本章导览

典型 RAG 应用用到**两类模型**：

- **大模型（LLM）**：智能推理与生成的核心引擎，类似"大脑"，是开发任何大模型应用首先考虑的基础设施。
- **嵌入模型（Embedding Model）**：实现文档向量化与后续语义检索。

> 复杂 RAG 流程中可能使用一个或多个模型，开发者可按模型特点、擅长任务、资源要求灵活组合。

**Prompt 的角色**：Prompt 是赋予大模型能力的**基本输入**。大模型与 Prompt **不只用于最后生成**，在 RAG 的多个阶段都需要：生成摘要、查询转换、查询路由、智能体推理、响应评估等。

> **为什么必须关注 Prompt 修改**：不同模型、不同场景、甚至不同语言下，对 Prompt 的响应都**并非完全一致且可预测**，框架内置默认模板并不总是最合适的（如默认英文模板对国内中文模型不友好）。因此"使用与修改 Prompt"是基础技术之一。

---

## 4.1 大模型

> 几乎所有大模型应用开发框架都会对底层大模型 API 做**抽象与封装**，以提供更简洁的使用与灵活的模型切换能力。

### 4.1.1 大模型在 RAG 应用中的作用

大模型具有强大的自然语言理解与生成能力，在 RAG 全流程的**四个阶段**都能派上用场：

| 阶段 | 大模型可承担的工作 |
|---|---|
| **(1) 数据前期准备** | 文档整理与总结、抽取知识问题、生成问答对、排除重复知识、知识格式的结构化与规范化 |
| **(2) 加载与索引** | ① 生成补充元数据（标题、假设性问答对等）；② 生成知识文本摘要用于构造"基于摘要的索引"；③ 对复杂文档中的表格数据生成描述/总结文本；④ 用大模型判断数据相关性、决定是否索引 |
| **(3) 检索与生成** | ① 查询路由（判断输入意图路由到不同检索路径）；② 查询扩展/重写（提高检索准确率）；③ 答案生成（基于问题与上下文生成结果）；④ 响应合成（多个子查询答案的合成） |
| **(4) 应用评估** | ① 借助大模型生成评估用的结构化数据集；② 对生成结果做多维度评估（相关性评估、正确性评估等） |

---

### 4.1.2 大模型组件的统一接口

LlamaIndex 中的大模型组件：既可作为**独立模块**简化对大模型的访问，也可作为**参数插入**其他核心模块（索引、检索器、查询引擎等）使用。框架定义了统一接口 `BaseLLM`：

```python
class BaseLLM(ChainableMixin, BaseComponent):
    """BaseLLM interface."""

    @abstractmethod
    def metadata(self) -> LLMMetadata: ...

    # ===== 同步接口 =====
    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse: ...

    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse: ...

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen: ...

    @abstractmethod
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen: ...

    # ===== 异步接口 =====
    @abstractmethod
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse: ...

    @abstractmethod
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse: ...

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen: ...

    @abstractmethod
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen: ...
```

**一个具体大模型组件必须实现的接口（4 类）**：

1. **获取元数据** `metadata` → `LLMMetadata`（大模型的描述信息）；
2. **两种调用形态**：文本预测 `complete` 与 对话 `chat`；
3. **两种输出形态**：流式 `stream_*` 与 非流式；
4. **两种并发形态**：同步 与 异步 `a*`。

> 设计含义：4 个维度（complete/chat × 流式/非流式 × 同步/异步）构成完整接口矩阵，具体模型组件按需实现。

**OpenAI 组件 `_complete` 实现示例（底层用 OpenAI 官方 SDK）**：

```python
def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
    client = self._get_client()
    all_kwargs = self._get_model_kwargs(**kwargs)
    self._update_max_tokens(all_kwargs, prompt)

    response = client.completions.create(
        prompt=prompt,
        stream=False,
        **all_kwargs,
    )
    text = response.choices[0].text
    ......
    return CompletionResponse(
        text=text,
        raw=response,
        logprobs=logprobs,
        additional_kwargs=self._get_response_token_counts(response),
    )
```

---

### 4.1.3 大模型组件的单独使用

常用于**模型测试、验证模型可用性**。测试 OpenAI GPT（需设 `OPENAI_API_KEY`）：

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

# 测试 complete 接口（一次性文本预测）
llm = OpenAI(model='gpt-3.5-turbo-1106')
resp = llm.complete("白居易是")
print(resp)

# 测试 chat 接口（多轮对话，带 system 角色）
messages = [
    ChatMessage(role="system", content="你是一个聪明的 AI 助手"),
    ChatMessage(role="user", content="你叫什么名字？"),
]
resp = llm.chat(messages)
print(resp)
```

**切换到本地 Ollama 模型**（只需替换两行）：

```python
from llama_index.llms.ollama import Ollama
llm = Ollama(model='qwen:14b')
```

> **框架价值**：借助良好的设计模式快速更换模型，实现**模型配置化**，无需关注不同模型 API 差异。

---

### 4.1.4 大模型组件的集成使用

实际开发中，大部分时候**不直接**用大模型生成/对话，而是把大模型访问对象**动态插入其他模块**（索引、检索器、查询引擎等），由这些模块按需调用。

#### (1) 更改默认的大模型

通过 `Settings` 改默认大模型（不改则默认 OpenAI GPT）：

```python
# 通过设置 Settings 组件更改使用的默认的大模型
llm = OpenAI(model='gpt-3.5-turbo-1106')
Settings.llm = llm
```

#### (2) 将大模型组件插入其他模块

在构造其他模块时插入大模型。**判断规则**：组件初始化/构造方法中有 `llm` 参数 → 代表可动态指定/更改大模型。

```python
llm = OpenAI(temperature=0.1, model="gpt-4")
index = KeywordTableIndex.from_documents(documents, llm=llm)
query_engine = index.as_query_engine()  # 后面查询将使用这里定义的大模型
```

> 这是"局部覆盖全局"的典型用法：即便 `Settings.llm` 已设全局模型，仍可在单个组件上用 `llm=` 覆盖。

---

### 4.1.5 了解与设置大模型的参数

通用参数：模型名称、温度（temperature，随机性）、上下文窗口大小（token 数）等；有的大模型有特殊参数。

**参数分两类**：

| 类型 | 含义 | 举例 |
|---|---|---|
| **模型定义参数** | 关于模型部署/接入的配置 | 模型名称、模型文档路径、服务端口 |
| **模型生成参数** | 影响生成行为的参数 | temperature、上下文窗口大小 |

**LlamaIndex 中有两个地方可设置参数**：

#### (1) 在大模型初始化时设置

```python
from llama_index.llms.ollama import Ollama

_MODEL_KWARGS = {
    "base_url": "http://localhost:11434",
    "model": "qwen:14b",
    "context_window": 4096,
    "request_timeout": 60.0
}
llm = Ollama(**_MODEL_KWARGS)
```

> 不同接入方式（直连 / Ollama / Llama_cpp / vLLM）和不同模型支持的参数不同，需查官方文档。

**LlamaCPP 模型服务的参数设置示例**（区分 `model_kwargs` 与 `generate_kwargs`）：

```python
from llama_index.llms.llama_cpp import LlamaCPP

_MODEL_KWARGS = {"logits_all": True, "n_ctx": 2048, "n_gpu_layers": -1}
_GENERATE_KWARGS = {
    "temperature": 0.0, "top_p": 1.0, "max_tokens": 500,
    "logprobs": 32016,
}
model_path = Path(download_dir) / "selfrag_llama2_7b.q4_k_m.gguf"
llm = LlamaCPP(
    model_path=str(model_path),
    model_kwargs=_MODEL_KWARGS,
    generate_kwargs=_GENERATE_KWARGS,
    verbose=False
)
```

#### (2) 在 Settings 组件中设置

少量参数可在 `Settings` 设置（如上下文窗口），**后续构造具体组件时可被覆盖**：

```python
from llama_index.core import Settings
Settings.context_window = 4096
Settings.num_output = 256
```

> `Settings` 中的设置是"全局默认"，组件构造参数（如 `Ollama(context_window=...)`）优先级更高，会覆盖 `Settings`。

---

### 4.1.6 自定义大模型组件

> Llama_cpp、Ollama 等接入用的大模型组件本质上都是自定义组件。

**实现方式**：从基础模型类 **`CustomLLM`** 派生，实现相应接口；在接口实现中自由访问本地模型、调用自定义 API。

```python
from typing import Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

class MyLLM(CustomLLM):
    model_name: str = "custom"
    dummy_response = "你好，我是一个正在开发中的大模型 ......"

    # 实现 metadata 接口
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
        )

    # 实现 complete 接口（注意装饰器）
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.dummy_response)

    # 实现 stream_complete 接口
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)
```

**测试**：

```python
llm = MyLLM()
resp = llm.complete('你好！')
print(resp)
# 输出：你好，我是一个正在开发中的大模型 ......
```

**替换全局默认大模型**：

```python
Settings.llm = MyLLM()
```

> **关键点**：
> - `@llm_completion_callback()` 装饰器不能漏，它负责把回调/事件机制接入；
> - `stream_complete` 用 `yield` 逐 token 产出，`delta` 是本次新增 token、`text` 是累计文本；
> - 自定义组件后，可像内置组件一样被任何模块使用。

---

### 4.1.7 使用 LangChain 框架中的大模型组件

LlamaIndex 封装了大量大模型组件，但为兼容更多模型，提供了对 **LangChain** 大模型组件的**适配器**，扩大可用模型范围。

**示例：通过适配器使用百度千帆大模型**：

```python
from llama_index.llms.langchain import LangChainLLM
from langchain_community.llms import QianfanLLMEndpoint

llm = LangChainLLM(llm=QianfanLLMEndpoint(model='ERNIE-Bot-4'))
Settings.llm = llm
```

> **机制**：用 `LangChainLLM` 把 LangChain 中声明的大模型适配成 LlamaIndex 的大模型组件接口即可正常使用——这是跨框架复用模型的桥梁。

---

## 4.2 Prompt

> RAG 应用和大模型**通过 Prompt 沟通**；Prompt 是获得大模型输出的基本输入。RAG 各阶段都会用到 Prompt，因此必须先掌握 Prompt 的概念与使用。

### 4.2.1 使用 Prompt 模板

**是什么**：PromptTemplate 是 LlamaIndex/LangChain 这类框架的基础组件，用于构造**包含参数变量**的 Prompt；这些变量在运行时通过代码格式化，形成真正的 Prompt。

**典型 RAG 增强 Prompt 模板**：

```python
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

template = (
    "以下是提供的上下文信息：\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "根据这些信息，请回答以下问题：{query_str}\n"
)
qa_template = PromptTemplate(template)

# format：模板 → 普通字符串（用于一次性提问/查询）
prompt = qa_template.format(
    context_str='小麦 15 PRO 是小麦公司最新推出的 6.7 寸大屏旗舰手机。',
    query_str='小麦 15pro 的屏幕尺寸是多少？'
)
print(prompt)

# format_messages：模板 → ChatMessage 封装类型（含 role 与 content，用于对话多轮上下文）
messages = qa_template.format_messages(
    context_str='小麦 15 PRO 是小麦公司最新推出的 6.7 寸大屏旗舰手机。',
    query_str='小麦 15pro 的屏幕尺寸是多少？'
)
print(messages)
```

**两种格式化方法对比**：

| 方法 | 返回类型 | 用途 |
|---|---|---|
| `format(...)` | 普通字符串 | 简单一次性提问/查询（complete 风格） |
| `format_messages(...)` | `ChatMessage` 列表（至少含 role、content） | 对话模型的连续上下文多轮对话（chat 风格） |

---

### 4.2.2 更改默认的 Prompt 模板

#### 为什么大多数时候不关心 Prompt 模板？

1. 开发框架内置大量默认且经过测试的 Prompt 模板，简化工作量；
2. 模板格式化通常由框架**在使用时自动完成**（如响应时自动注入上下文与查询）。

> LlamaIndex 的默认模板位于 `llama-index-core-prompts` 模块目录（如图 4-1 的常见问答模板）。

#### 何时需要更改？典型场景

把默认**英文** Prompt 模板改为**中文** Prompt 模板，以更好适配国内大模型。

#### 更改流程（三步）

1. **确定阶段**：先确定要改 Prompt 模板的阶段；
2. **定位组件**：找到该阶段涉及 Prompt 模板的组件；
3. **用组件接口设置/更改**。

**关键接口**：用到 Prompt 模板的组件都可调用 `get_prompts` 与 `update_prompts` 接口。

**第 1 步——查看组件使用了哪些模板**（以查询引擎为例）：

```python
query_engine = index.as_query_engine()
prompts_dict = query_engine.get_prompts()
pprint.pprint(prompts_dict.keys())
```

输出是一个**字典对象**（图 4-2）——因为一个组件可能使用多个 Prompt 模板。

**第 2 步——查看某个具体模板内容**（如 `text_qa_template`）：

```python
pprint.pprint(prompts_dict["response_synthesizer:text_qa_template"].get_template())
```

> 模板在字典里的 key 形如 `<子组件名>:<模板名>`，例如 `response_synthesizer:text_qa_template`（图 4-3）。

**第 3 步——更改模板**（改为中文）：

```python
my_qa_prompt_tmpl_str = (
    "以下是上下文信息。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "根据上下文信息回答问题，不要依赖预置知识，不要编造。\n"
    "问题: {query_str}\n"
    "回答: "
)
my_qa_prompt_tmpl = PromptTemplate(my_qa_prompt_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": my_qa_prompt_tmpl}
)
```

> **⚠ 关键陷阱**：`context_str` 与 `query_str` 这两个**变量名称不能修改**，否则运行时绑定变量失败！

#### 替代方式：通过初始化参数直接传入

```python
query_engine = index.as_query_engine(text_qa_template=my_qa_prompt_tmpl)
```

效果与 `update_prompts` 相同。**通用规律**：LlamaIndex 中使用组件时，查看其初始化方法即可决定要改哪个模板，通过对应参数传入。

---

### 4.2.3 更改 Prompt 模板的变量

4.2.2 强调"通常不能随意改模板变量名"，但如果**必须改**（如想用更有意义的变量名），有两种机制：

#### (1) `template_var_mappings`：变量名映射

给自己的变量与框架要求的模板变量建立映射关系：

```python
my_qa_prompt_tmpl_str = (
    "以下是上下文信息。\n"
    "---------------------\n"
    "{my_context_str}\n"
    "---------------------\n"
    "根据上下文信息回答问题，不要依赖预置知识，不要编造。\n"
    "问题: {my_query_str}\n"
    "回答: "
)
template_var_mappings = {"context_str": "my_context_str", "query_str": "my_query_str"}
my_qa_prompt_tmpl = PromptTemplate(my_qa_prompt_tmpl_str)

# 使用自定义变量来格式化 Prompt 模板
print(my_qa_prompt_tmpl.format(my_context_str="......", my_query_str="......"))
```

#### (2) `function_mappings`：变量映射到函数

把 `context_str` 通过 `function_mappings` 映射到一个函数——格式化时 `context_str` 的实际值由调用该函数获得，函数输入是调用 `format` 时携带的关键词参数：

```python
# kwargs 为调用 format 方法时携带的关键词参数
def fn_context_str(**kwargs):
    # ......自定义逻辑......
    return fmtted_context

prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str,
    function_mappings={"context_str": fn_context_str}
)

# format 参数传入 fn_context_str 变量中
prompt_tmpl.format(context_str="...", query_str="...")
```

> **使用场景**：当 `context_str` 需要动态加工（如重排序、压缩、拼接、清洗）时，用 `function_mappings` 把加工逻辑封装进函数，模板格式化时自动执行。

---

## 4.3 嵌入模型

### 4.3.1 嵌入模型在 RAG 应用中的作用

**核心作用**：把分割后的知识块转换为向量——把自然语言转换为更易被计算机理解、存储与计算，并能表示语义的多个数值。

**为什么能做语义检索**：

- 嵌入模型生成的向量能**捕获文本语义**（本书以文本嵌入为主，但理论上**所有模态知识都可被嵌入**，需借助特殊模型）；
- 与关键词检索不同，基于向量的相似检索**不是字符串匹配**，而是按向量"相似程度"检索；
- 相似度算法：**余弦相似度、点积、欧几里得距离**等；这种海量向量相似语义检索能力是后面要讲的向量库的核心能力之一。

**嵌入模型在 RAG 的两个阶段都被使用**：

| 阶段 | 作用 |
|---|---|
| **索引阶段** | 将知识块转成向量，借向量库存储 |
| **生成阶段** | 将输入问题转成向量，借向量库检索相似语义的相关知识块，实现增强生成 |

> **选型参考**：想了解哪个嵌入模型更适合、查看基准测试数据或评估自有模型，参考 Hugging Face 的 **MTEB（Massive Text Embedding Benchmark，大文本嵌入基准）** 及其 GitHub 开源项目。

---

### 4.3.2 嵌入模型组件的接口

嵌入模型组件主要接口应是"生成文本向量"。看 LlamaIndex 嵌入模型基础类 `BaseEmbedding` 及相关定义：

```python
# 用于保存嵌入后的向量
Embedding = List[float]

# 相似度计算的 3 种算法：余弦相似度、点积、欧几里得距离
class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""
    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"

# 辅助方法：两个向量相似度比较
def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    ......

# 嵌入模型基础类
class BaseEmbedding(TransformComponent):
    ......
    @abstractmethod
    def _get_query_embedding(self, query: str) -> Embedding: ...

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> Embedding: ...

    @abstractmethod
    def _get_text_embedding(self, text: str) -> Embedding: ...

    @abstractmethod
    async def _aget_text_embedding(self, text: str) -> Embedding: ...
    ......
```

> **关键设计**：`_get_query_embedding`（问题向量化）与 `_get_text_embedding`（文档向量化）**分离**——这是因为部分嵌入模型对 query 与 document 采用不同编码前缀/策略（非对称检索），分离接口可分别优化。

**OpenAI 嵌入模型具体实现**（模块 `llama-index-embedding-openai`）：

```python
class OpenAIEmbedding(BaseEmbedding):
    ......
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        client = self._get_client()
        return get_embedding(
            client,
            text,
            engine=self._text_engine,
            **self.additional_kwargs,
        )
```

`get_embedding` 实现：

```python
def get_embedding(client: OpenAI, text: str, engine: str, **kwargs: Any) -> List[float]:
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=engine, **kwargs).data[0].embedding
    )
```

> **细节**：实现逻辑就是借助 OpenAI 官方 SDK 构造访问对象，用 `create` 方法生成向量。注意 `text.replace("\n", " ")`——OpenAI 建议把换行替换为空格以提升嵌入质量。

---

### 4.3.3 嵌入模型组件的单独使用

#### (1) OpenAI 嵌入模型

LlamaIndex 默认嵌入模型是 OpenAI `text-embedding-ada-002`，可不设置直接使用：

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

embed_model = OpenAIEmbedding()
embeddings = embed_model.get_text_embedding("中国的首都是北京")
print(embeddings)
# 输出：浮点数值向量数组，如 [0.007398069836199284, -0.011682811193168163, ...]
```

**比较两个向量的相似度**：

```python
embeddings1 = embed_model.get_text_embedding("中国的首都是北京")
embedding2 = embed_model.get_text_embedding("中国的首都是哪里？")
embedding3 = embed_model.get_text_embedding("苹果是一种好吃的水果")
print(embed_model.similarity(embeddings1, embedding2))  # 0.9324... 语义更相似
print(embed_model.similarity(embeddings1, embedding3))  # 0.7942... 语义差异更大
```

> 实测：`embedding1`（"中国的首都是北京"）与 `embedding2`（"中国的首都是哪里？"）相似度更高（0.93），与 `embedding3`（"苹果是一种好吃的水果"）相似度较低（0.79）——证明嵌入能捕获语义。

#### (2) Ollama 的本地嵌入模型

启动本地 Ollama 服务后，简单替换部分代码即可：

```python
from llama_index.embeddings.ollama import OllamaEmbedding
embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")
```

#### (3) TEI 的本地嵌入模型

借助 **Text Embeddings Inference（TEI）** 嵌入模型部署工具，可使用 Hugging Face 上的著名嵌入模型（如优秀中文嵌入模型 `bge-large-zh`）。

**启动 TEI 服务**：

```bash
model=BAAI/bge-large-zh-v1.5
text-embeddings-router --model-id $model --port 8080
```

**用本地 TEI 服务做嵌入**：

```python
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-large-zh-v1.5",
    timeout=60,            # timeout in seconds
    embed_batch_size=10,   # batch size for embedding
)
```

> **结论**：借助 LlamaIndex 切换嵌入模型非常便捷——同一套接口，背后可对接 OpenAI / Ollama / TEI 等多种后端。

---

### 4.3.4 嵌入模型组件的集成使用

与大模型组件一样，嵌入模型组件也可**插入其他模块**（如索引模块）在构造知识索引时自动嵌入。

#### (1) 更改默认的嵌入模型

通过 `Settings.embed_model` 更改：

```python
Settings.embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")
```

#### (2) 将嵌入模型组件插入其他模块

构造具体模块时插入，例如构造向量存储索引：

```python
embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")
index = VectorStoreIndex(nodes, embed_model=embed_model)
```

---

### 4.3.5 了解与设置嵌入模型的参数

**常见参数**：

| 参数 | 含义 |
|---|---|
| `model_name` | 需要使用的模型名称 |
| `embed_batch_size` | 批量送入模型进行处理的窗口大小 |
| `timeout` | 处理超时时间 |
| `max_retries` | 最大尝试次数 |
| `dimensions` | 生成的向量维度（如 2048） |

> **重要原则**：不同嵌入模型支持的参数**取决于模型本身而非 LlamaIndex**，使用时需查具体模型文档或查看 LlamaIndex 对应组件的初始化代码。

**构造时设置参数示例**：

```python
_MODEL_KWARGS = {
    "model_name": "milkey/dmeta-embedding-zh:f16",
    "embed_batch_size": 50
}
embed_model = OllamaEmbedding(**_MODEL_KWARGS)
```

---

### 4.3.6 自定义嵌入模型组件

> 公开发布的商业/开源嵌入模型有上百个，并非所有都在 LlamaIndex 中封装；机构也可能发布专用嵌入模型。此时需**自定义嵌入模型组件**。

**实现方式**：继承 `BaseEmbedding`，实现相应接口（逻辑需参考对应模型的调用说明）。

```python
from llama_index.core.embeddings import BaseEmbedding

# 导入自己的嵌入模型提供的模块，实现 embed 方法
from ... import MyModel

class MyEmbeddng(BaseEmbedding):
    def __init__(
        self,
        model_name: str = 'MyEmbeddingModel',
        **kwargs: Any,
    ) -> None:
        # 构造一个模型调用对象（模拟）
        self._model = MyModel(model_name)
        super().__init__(**kwargs)

    # 生成向量（模拟）
    def _get_text_embedding(self, text: str) -> List[float]:
        embedding = self._model.embed(text)
        return embedding

    # 批量生成向量（模拟）
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.embed([text for text in texts])
        return embeddings

    # ......实现其他必需的接口（_get_query_embedding 等）......
```

> 自定义完成后即可像内置组件一样使用（设到 `Settings.embed_model` 或作为参数插入其他模块）。

---

## 面试考点

### Q1：RAG 中用到哪两类模型？分别承担什么职责？
**答**：① 大模型（LLM）——推理与生成的核心引擎，用于数据预处理、元数据生成、查询路由/重写、答案生成/合成、应用评估等；② 嵌入模型——把知识块/查询转成向量，支撑索引（文档向量化入库）与检索（查询向量化做相似检索）。

### Q2：LlamaIndex 大模型组件统一接口 `BaseLLM` 必须实现哪些方法？背后的设计维度是什么？
**答**：必须实现 `metadata`，以及 `complete`/`chat`、`stream_complete`/`stream_chat` 及其异步版本 `acomplete`/`achat`/`astream_*`。设计维度是 4 个正交方向的组合：① complete vs chat（文本预测 vs 对话）；② 流式 vs 非流式；③ 同步 vs 异步；④ 元数据。这保证任何模型组件都能在统一调用形态下被使用与替换。

### Q3：如何在 LlamaIndex 中切换大模型？有哪些层次？
**答**：三个层次——① 全局默认：`Settings.llm = OpenAI/Ollama(...)`；② 组件级覆盖：构造组件时传 `llm=` 参数（有 `llm` 参数的组件都支持，优先级高于 `Settings`）；③ 自定义：继承 `CustomLLM` 实现接口（注意 `@llm_completion_callback()` 装饰器）。

### Q4：嵌入模型为什么要把 `_get_query_embedding` 和 `_get_text_embedding` 分离？
**答**：因为部分嵌入模型（如 bge 系列）对查询（query）和文档（document）采用不同的编码前缀/指令（非对称检索，asymmetric），分离接口可分别优化两者编码，提升检索精度。

### Q5：LlamaIndex 中如何修改默认 Prompt 模板？关键约束是什么？
**答**：通过组件的 `get_prompts()` 查看所用模板（返回字典，key 形如 `response_synthesizer:text_qa_template`），再用 `update_prompts({key: new_template})` 更新，或在组件构造时通过参数（如 `text_qa_template=`）传入。**关键约束**：`context_str` 与 `query_str` 等模板变量名不能随意改，否则运行时绑定失败；若必须改名，用 `template_var_mappings` 建立映射，或用 `function_mappings` 把变量映射到函数实现动态加工。

### Q6：`PromptTemplate.format` 与 `format_messages` 的区别？
**答**：`format` 返回普通字符串，用于一次性提问/查询（complete 风格）；`format_messages` 返回 `ChatMessage` 列表（含 role 与 content），用于对话模型的连续上下文多轮对话（chat 风格）。

### Q7：如何接入 LangChain 体系的大模型到 LlamaIndex？
**答**：用适配器组件 `LangChainLLM(llm=<langchain_model>)` 把 LangChain 中声明的大模型适配成 LlamaIndex 的 `BaseLLM` 接口即可正常使用，例如 `LangChainLLM(llm=QianfanLLMEndpoint(model='ERNIE-Bot-4'))`。

### Q8：向量相似度有哪些计算方式？默认是哪种？
**答**：余弦相似度（cosine，默认 `SimilarityMode.DEFAULT`）、点积（dot_product）、欧几里得距离（euclidean）。LlamaIndex 提供辅助方法 `similarity(e1, e2, mode=...)`。

### Q9：嵌入模型的常见参数有哪些？为什么不同模型支持的参数不一样？
**答**：`model_name`、`embed_batch_size`（批处理窗口）、`timeout`、`max_retries`、`dimensions`（向量维度）。不同模型支持的参数**取决于模型本身**（不同后端 API 能力不同）而非 LlamaIndex 框架，使用时需查具体模型文档。

---

## 易错 / 陷阱

1. **默认模型都是 OpenAI**：不设 `Settings.llm` / `Settings.embed_model` 时，LlamaIndex 默认调 OpenAI（大模型 GPT-3.5-Turbo、嵌入 text-embedding-ada-002 / text-embedding-3-small），漏配 `OPENAI_API_KEY` 会直接报错。
2. **`Settings` vs 组件参数的覆盖关系**：`Settings` 是"全局默认"，**组件构造参数优先级更高**会覆盖 `Settings`。调试"为什么我改了 Settings 没生效"时，先检查组件是不是传了同名参数。
3. **`complete` vs `chat` 用混**：`complete` 接收纯字符串 prompt（无角色区分）；`chat` 接收 `Sequence[ChatMessage]`（需带 `role`）。把多轮上下文塞进 `complete` 会丢失角色信息。
4. **自定义 `CustomLLM` 漏装饰器**：`complete`/`stream_complete` 必须加 `@llm_completion_callback()`，否则回调/事件追踪机制失效（与第3章的 LlamaDebugHandler / Langfuse 联动会丢数据）。
5. **`stream_complete` 的 `delta` 与 `text` 混淆**：`delta` 是本次新增 token，`text` 是累计文本。下游若误把 `text` 当增量拼接会出现重复输出。
6. **Prompt 模板变量名硬约束**：自定义模板时 `context_str`、`query_str` 等变量名不能改，否则运行时变量绑定失败（报 KeyError 或留空）。要改必须配 `template_var_mappings`。
7. **`get_prompts()` 返回字典的 key 格式**：是 `<子组件名>:<模板名>`（如 `response_synthesizer:text_qa_template`），漏写子组件前缀会导致 `update_prompts` 静默不生效。
8. **嵌入模型 query/document 接口用错**：把文档塞进 `_get_query_embedding` 或反过来，在非对称嵌入模型上会显著降低检索精度，且无明显报错。
9. **`SimilarityMode` 与向量库 `hnsw:space` 不一致**：嵌入模型/框架默认按 cosine 算相似度，但向量库（如 Chroma）默认 `l2`，两边度量不一致会导致检索排序偏差。需保持一致（多数文本嵌入用 cosine）。
10. **`embed_batch_size` 设过大**：超出嵌入服务并发/请求体上限会触发超时或 413，生产环境需按后端能力调优；`timeout` 与 `max_retries` 也要相应设置。
11. **OpenAI 嵌入的换行处理**：官方实现会把 `\n` 替换为空格——自定义嵌入后端时若不做类似处理，长文档嵌入质量可能下降。
12. **以为改了 Prompt 就一定更好**：不同模型对 Prompt 响应不同且不可完全预测，改模板后必须配合评估（第3章 Langfuse / 第6章评估）回归验证，不能凭感觉。
13. **TEI/Ollama 嵌入服务未启动**：代码切到本地嵌入模型但服务没起，报错信息常被框架包装，容易误判为组件 bug；先 `curl` 验证服务可达。

---

## 本章小结

- **大模型**：理解 `BaseLLM` 统一接口（complete/chat × 流式/非流式 × 同步/异步）、单独使用（测连通）、集成使用（`Settings.llm` 或组件 `llm=` 参数）、参数（定义参数 vs 生成参数）、自定义（`CustomLLM`）、跨框架适配（`LangChainLLM`）。
- **Prompt**：掌握 `PromptTemplate` 及 `format`/`format_messages` 两种格式化、`get_prompts`/`update_prompts` 改默认模板、`template_var_mappings`/`function_mappings` 改变量。
- **嵌入模型**：理解 `BaseEmbedding` 接口（query/text 分离）、三种相似度算法、OpenAI/Ollama/TEI 多后端切换、参数（model_name/embed_batch_size/timeout/dimensions 等）、自定义（继承 `BaseEmbedding`）。
- **核心思想**：框架通过统一接口 + 组件化 + 全局 `Settings` 实现"模型/Prompt 配置化"，让模型切换、Prompt 调整、组件替换都低成本可控——这是企业级 RAG 可维护性的基石。
