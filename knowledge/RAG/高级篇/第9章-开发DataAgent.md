# 第 9 章 开发 Data Agent

> 来源：严灿平《基于大模型的 RAG 应用开发与优化——构建企业级 LLM 应用》【高级篇】
> 定位：将 RAG 应用模块拓展到 Agent，开发以数据为中心的 Data Agent。

## 本章总览

**Agent** 是一种更高级的应用形式，普遍被认为是生成式 AI 的终极形式。Agent = AI 模型驱动 + 自主理解、规划、执行、完成任务。

OpenAI 应用研究主管 Lilian Weng 的经典定义：
```
Agent = 大模型 + 记忆 + 规划技能 + 使用工具
```

**Agent 与大模型的区别 ≈ 人与大脑的区别**：大脑指挥行动，但只有人才是执行任务的完整体。

**本章重点：**
1. Data Agent 的概念与能力（9.1）
2. 构造与使用 Agent 的工具（9.2）：FunctionTool / QueryEngineTool / RetrieverTool / QueryPlanTool / OnDemandLoaderTool
3. 基于函数调用直接开发 Agent（9.3）
4. 用框架组件开发 Agent（9.4）：OpenAIAgent / ReActAgent / 底层 API / 工具检索 / 上下文检索
5. 更细粒度地控制 Agent 运行（9.5）：分步运行 / 人类交互

---

## 9.1 初步认识 Data Agent

### 9.1.1 概念

**Data Agent** 在 RAG 基础上引入 **自我规划 + 使用工具** 能力，具备完成大模型驱动的、更丰富的 **数据读写任务** 能力。Data Agent 不仅能查询数据，还能使用工具执行真正的 **数据操作任务**，扩大了 RAG 应用场景。

### 9.1.2 Data Agent 相比 RAG 的 4 项能力

1. 兼具 RAG 应用的数据检索与查询生成能力。
2. 通过观察环境与任务目标 **推理出** 完成下一个数据任务的步骤。
3. 通过调用 **外部服务工具（API）** 完成复杂任务，并返回执行结果。
4. 具备 **长期记忆**（如向量库）与 **短期记忆**（一次任务的交互历史）能力。

### 9.1.3 Data Agent 相比 RAG 的 2 大增强

#### 增强 1：规划与推理任务步骤的能力
- **LlamaIndex 实现：** `Agent` 组件。借助大模型，使用 **循环推理** 规划任务步骤、使用的工具、工具输入参数，并调用工具完成任务。
- **常见推理范式：** **ReAct（Reasoning & Acting，推理-行动范式）**。

#### 增强 2：定义与使用工具的能力
- **LlamaIndex 实现：** `Tool` 相关组件。
- 每个工具通常是具备规范 **请求参数 + 响应参数** 的 API 或函数。请求参数通常是结构化参数，响应参数可以是文本字符串或任何格式。
- 框架提供便捷方法定义工具，支持把已有的可运行组件（如查询引擎）包装成工具。

### 9.1.4 框架支持的组件类型

**Agent 组件：**
| 类型 | 适用场景 |
|------|---------|
| `OpenAIAgent` | 支持带有函数调用功能的大模型 |
| `ReActAgent` | 支持其他大模型，通过 ReAct 推理范式规划推理 |

**Tool 相关组件：**
| 组件 | 用途 |
|------|------|
| `FunctionTool` | 将函数转换为 Agent 可用的工具 |
| `BaseTool` / `ToolMetadata` | 基础工具抽象 + 工具元数据定义 |
| `QueryEngineTool` | 将查询引擎转换为 Agent 工具 |
| `SlackToolSpec` | 工具规格定义组件，可直接转换为工具 |
| 工具库（LlamaHub） | 大量开箱即用的外部工具 |

---

## 9.2 构造与使用 Agent 的工具

### 9.2.0 工具可封装的对象

工具可以是对以下组件的封装：
1. 一个本地自定义函数（Function）
2. 一个查询引擎（QueryEngine）
3. 一个查询管道（QueryPipeline）
4. 一个检索器（Retriever）
5. 一个其他的 Agent
6. 企业内部其他应用的开放接口
7. 第三方公开的 API

### 9.2.1 深入了解工具类型

**基础抽象：`BaseTool`**，重要属性与接口：
- **`metadata`**：工具元数据（名称、描述、接口规范、是否直接返回等）。**元数据非常关键** —— 是帮助大模型理解工具用途并推理使用工具的重要信息。部分元数据会被自动生成。
- **`call()`**：工具必须实现的核心调用接口。

**几种工具类型：**
| 类型 | 作用 |
|------|------|
| `FunctionTool` | 把本地函数直接转换成工具，最简单但最灵活，可实现任意无状态逻辑 |
| `QueryEngineTool` | 把已构造好的查询引擎发布成工具 |
| `RetrieverTool` | 把检索器发布成工具，支持同时插入节点后处理器，输出 Node 内容 |
| `QueryPlanTool` | 根据传入的查询引擎工具和执行计划调用工具完成任务 |
| `OnDemandLoaderTool` | 借助指定的文档加载器加载数据 + 构造索引 + 查询 |

### 9.2.2 函数工具（FunctionTool）

```python
from llama_index.core.tools import FunctionTool

def add(a: int, b: int) -> int:
    return a + b

tool_add = FunctionTool.from_defaults(
    fn=add,
    name="tool_add",
    description="用于两个整数相加",
)

# 调用
output = tool_add.call(a=1, b=3)
print(type(output))            # ToolOutput
print(f"Output: {output.__dict__}")
```

**输出结构 `ToolOutput`：**
```python
{
    'content': '4',                    # 函数调用返回内容
    'tool_name': 'add',                # 工具名称
    'raw_input': {'args': (), 'kwargs': {'a': 1, 'b': 3}},   # 输入参数
    'raw_output': 4,                   # 原始输出
    'is_error': False                  # 是否出错
}
```

**本质：** 把自定义函数打包成 Tool 对象，调用时把请求转为函数调用，以标准形式（ToolOutput）输出返回值。

### 9.2.3 查询引擎工具（QueryEngineTool）

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine = index.as_query_engine(
    response_mode="compact", verbose=True, text_qa_template=qa_prompt
)

tool_xiaomai = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="tool_xiaomai",
    description="用于小麦手机信息查询",
    return_direct=False
)

print(tool_xiaomai.call(query_str="小麦手机采用了什么型号的 CPU？"))
```

**重要特性：** LlamaIndex 中的 **Agent 本身继承自 QueryEngine**，即 Agent 也是一种查询引擎，因此 Agent 也可包装成工具供其他 Agent 使用 → 实现 **Agent 之间互相调用**。

### 9.2.4 检索工具（RetrieverTool）

**适用场景：** 需要直接使用检索器的场景（如带有路由功能的检索器本质就是简单 Agent，需通过检索工具打包）。

```python
from llama_index.core.tools import RetrieverTool

vector_index = VectorStoreIndex(nodes)
retriever_xiaomai = vector_index.as_retriever(similarity_top_k=2)

tool_retriever_xiaomai = RetrieverTool.from_defaults(
    retriever=retriever_xiaomai,
    description="用于检索小麦手机的信息",
)

print(tool_retriever_xiaomai.call(query_str="小麦手机采用了什么型号的 CPU？"))
```

**输出：** 检索出的 Node 的元数据与内容（不是最终答案）。

### 9.2.5 查询计划工具（QueryPlanTool）

**作用：** 在其他工具之上的上层工具。接收一系列其他工具作为输入，被调用时根据传入的 **执行计划**（工具调用顺序及关系）完成任务。

**两种用法：**
1. Agent 配合：Agent 根据已有工具信息 **生成执行计划** → 交给 QueryPlanTool 执行。
2. 预设计划：预先设计复杂执行计划 → 交给 QueryPlanTool 执行（**减少大模型自行规划的不确定性**）。

```python
from llama_index.core.tools import QueryPlanTool
from llama_index.core import get_response_synthesizer
from llama_index.core.tools.query_plan import QueryPlan, QueryNode

query_xiaomai = index1.as_query_engine(response_mode="compact")
query_ultra = index2.as_query_engine(response_mode="compact")

# 两个子工具
query_tool_xiaomai = QueryEngineTool.from_defaults(
    query_engine=query_xiaomai,
    name="query_tool_xiaomai",
    description="提供小麦手机普通型号 Pro/Max 的信息")

query_tool_ultra = QueryEngineTool.from_defaults(
    query_engine=query_ultra,
    name="query_tool_ultra",
    description="提供小麦手机 Ultra 的信息")

response_synthesizer = get_response_synthesizer()
query_plan_tool = QueryPlanTool.from_defaults(
    query_engine_tools=[query_tool_xiaomai, query_tool_ultra],
    response_synthesizer=response_synthesizer,
)

# 构造执行计划：由多个 QueryNode 组成，dependencies 指定依赖关系
nodes = [
    QueryNode(id=1, query_str="查询小麦手机普通型号 Pro 的信息",
              tool_name="query_tool_xiaomai", dependencies=[]),
    QueryNode(id=2, query_str="查询小麦手机 Ultra 的信息",
              tool_name="query_tool_ultra", dependencies=[1]),
    QueryNode(id=3, query_str="对比小麦手机普通型号 Pro 与小麦手机 Ultra 的配置区别",
              tool_name="vs_tool", dependencies=[1, 2])
]

output = query_plan_tool(nodes=nodes)
print(output)
```

**关键点：** `QueryNode` 通过 `id`、`query_str`、`tool_name`、`dependencies`（依赖的 node id 列表）描述执行计划的 DAG 结构。

### 9.2.6 按需加载工具（OnDemandLoaderTool）

**作用：** 按需调用数据加载器自动读取数据 → 自动构造索引 → 构造查询引擎 → 查询输出。**在被调用时即时构造一个查询引擎**。

**注意（陷阱）：** 包含即时构造查询引擎的全部阶段，实时加载索引数据量大时会有较长时间响应延迟，**只适合少量"按需"加载场景**。

```python
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core.tools import OnDemandLoaderTool

def _baidu_reader(soup, url, include_url_in_text=True):
    # ... Web 网页解析逻辑
    return text, {"title": soup.find(class_="post__title").get_text()}

web_loader = BeautifulSoupWebReader(website_extractor={"cloud.baidu.com": _baidu_reader})

tool_xiaomai = OnDemandLoaderTool.from_defaults(
    web_loader,
    name="tool_xiaomai",
    description="用于查询本地文档中的小麦手机信息",
)

# web_loader 需要 URL 列表，因此 call 时需同时传 urls 和 query_str
output = tool_xiaomai.call(
    urls=["https://cloud.baidu.com/doc/AppBuilder/s/6lq7s8lli"],
    query_str='百度云千帆 appbuilder 是什么？'
)
```

---

## 9.3 基于函数调用功能直接开发 Agent

### 9.3.0 原理

**函数调用（Function Calling）**：大模型能根据输入参数中携带的函数调用信息，自动判断是否需要函数调用，并返回函数调用要求（函数名、函数输入等）。把函数看作工具，大模型本身就具备使用"推理"工具的能力 → 可基于函数调用直接开发 Agent。

### 9.3.1 准备工具

```python
from llama_index.core.tools import FunctionTool

# 工具 1：搜索天气
def search_weather(query: str) -> str:
    """用于搜索天气情况"""
    return "明天晴转多云，最高温度 30℃，最低温度 23℃。天气炎热，注意防晒哦。"
tool_search = FunctionTool.from_defaults(fn=search_weather)

# 工具 2：发送电子邮件
def send_email(subject: str, recipient: str, message: str) -> None:
    """用于发送电子邮件"""
    print(f"邮件已发送至 {recipient}，主题为 {subject}，内容为 {message}")
tool_send_mail = FunctionTool.from_defaults(fn=send_email)

# 工具 3：查询客户信息
def query_customer(phone: str) -> str:
    """用于查询客户信息"""
    return f"该客户信息为:\n 姓名: 张三\n 电话: {phone}\n 地址: 北京市海淀区"
tool_generate = FunctionTool.from_defaults(fn=query_customer)
```

### 9.3.2 开发 Agent（核心实现）

```python
class MyOpenAIAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-3.5-turbo"),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))

        # 关键 1：传入工具规格
        tools = [tool.metadata.to_openai_tool() for _, tool in self._tools.items()]
        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        # 关键 2：获取工具调用要求
        tool_calls = additional_kwargs.get("tool_calls", None)

        # 关键 3：如果调用工具，依次调用
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                # 继续对话（带工具结果）
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, tool_call) -> ChatMessage:
        id_ = tool_call.id
        function_call = tool_call.function
        tool = self._tools[function_call.name]
        output = tool(**json.loads(function_call.arguments))   # 解析参数并调用
        return ChatMessage(
            name=function_call.name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call.name,
            },
        )
```

**核心流程（重要）：**
1. **传入工具规格**：`tool.metadata.to_openai_tool()` 把工具元数据转为 OpenAI 函数调用格式。
2. **获取工具调用要求**：从 `ai_message.additional_kwargs["tool_calls"]` 读取。
3. **依次调用工具**：解析 `function_call.arguments`（JSON）→ 调用对应工具 → 把结果作为 `role="tool"` 消息加入历史。
4. **继续对话**：把工具结果送回大模型生成最终答案。

### 9.3.3 连续对话

```python
while True:
    user_input = input("请输入您的消息：")
    if user_input.lower() == "quit":
        break
    response = agent.chat(user_input)
    print(response)
```

---

## 9.4 用框架组件开发 Agent

LlamaIndex 中 Agent 组件分两类：
- **基于函数调用：`OpenAIAgent`**（支持其他有函数调用功能的大模型）
- **基于 ReAct 推理范式：`ReActAgent`**

### 9.4.1 使用 OpenAIAgent

```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
agent = OpenAIAgent.from_tools(
    [tool_search, tool_send_mail, tool_generate],
    llm=llm,
    verbose=True
)
```

**优势：** 与 9.3 原生开发效果一致，但代码大幅简化。通过 `verbose=True` 观察内部跟踪信息（是否调用函数、输入输出参数、最终结果）。

### 9.4.2 使用 ReActAgent

**适用：大模型不支持函数调用功能时**，使用基于 ReAct 推理范式的 Agent。

**ReAct（Reasoning & Acting）** 结合了 **思维链（CoT）提示工程 + 行动规划**，使大模型能进行任务推理、规划、完成。

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(
    [tool_search, tool_send_mail, tool_generate],
    llm=llm,
    verbose=True
)
```

**ReAct 推理循环 Prompt 模板（重要）：**

```
您可以使用以下工具：
{tool_desc}

要回答问题，请使用以下格式。

思考：我需要使用一个工具来帮助我回答这个问题。
行动：工具名称（{tool_names} 之一）
行动输入：工具的输入，采用表示 kwargs 的 JSON 格式（例如 {"text": "hello world", "num_beams": 5}）

如果使用此格式，您将收到以下格式的响应：
观察：工具响应
```

**ReAct 循环：思考（Thought）→ 行动（Action）→ 观察（Observation）→ 思考...** 直至完成任务。

**OpenAIAgent vs ReActAgent 核心区别：**
| 维度 | OpenAIAgent | ReActAgent |
|------|------------|-----------|
| 依赖 | 大模型函数调用功能 | ReAct Prompt 推理范式 |
| 工具选择机制 | 大模型原生 function calling | 通过 Thought/Action/Observation 文本推理 |
| 适用模型 | 支持函数调用的模型（如 GPT） | 任意大模型 |

### 9.4.3 使用底层 API 开发 Agent（AgentRunner + AgentWorker）

**架构（图 9-9）：**
- **`AgentRunner`**：顶级协调器。构造任务、运行任务每一步或端到端运行、保存状态、跟踪任务。
- **`AgentWorker`**：在 AgentRunner 内部构造，**真正执行任务步骤（step）**，返回每步输出。**AgentWorker 不保存任务状态，只负责执行**；AgentRunner 负责调用 AgentWorker 并收集聚合结果。

**底层 API 的好处：**
- 分离任务的构造与执行
- 获得更精细的观察结果
- 控制与调试任务的每一步
- 及时取消任务
- 定制 AgentWorker

```python
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker

llm = OpenAI(model="gpt-3.5-turbo")
openai_step_engine = OpenAIAgentWorker.from_tools(
    tools, llm=llm, verbose=True
)
agent = AgentRunner(openai_step_engine)
```

**等价性：** 这种方式开发完全等价于直接用 `OpenAIAgent` 开发。

### 9.4.4 开发带有工具检索功能的 Agent（重要）

**问题：** 工具集过大 → 推理能力下降甚至工具使用错乱。

**解决思路：** 使用工具前，根据输入任务语义对工具 **检索与过滤**，缩小工具集，降低推理错误概率。

**实现：给工具对象构造对象索引（ObjectIndex）**，运行时动态检索相关工具。

```python
from llama_index.core import VectorStoreIndex, ObjectIndex

tools = [tool_search, tool_send_mail, tool_customer]

# 构造对象索引，底层用向量检索
obj_index = ObjectIndex.from_objects(
    tools,
    index_cls=VectorStoreIndex,
)

llm = OpenAI(model="gpt-3.5-turbo")

# 注意：提供 tool_retriever（工具检索器），而不是工具集
agent = OpenAIAgent.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=2),
    verbose=True
)

# 测试：用 agent_worker 检索工具
tools = agent.agent_worker.get_tools('发送电子邮件')
for tool in tools:
    print(f'Tool name: {tool.metadata.name}')
```

**验证：**
- 输入"发送电子邮件" → 检索出 `send_email`（第一个）
- 输入"查询北京明天的天气" → 检索出 `search_weather`（第一个）

**核心模式（重要）：ObjectIndex 又一次出现** —— 与 SQL 查询引擎检索表（8.4.2）相同模式，对任意对象（这里是对工具对象）建向量索引。

### 9.4.5 开发带有上下文检索功能的 Agent

**适用场景：** 工具数量不多，但各工具工作内容易混淆/相似，仅用 description 无法区分 → 工具选择出错。

**解决思路：** 给 Agent 提供独立的 **上下文检索器（Context Retriever）**，推理行动时检索与输入问题相关的上下文（Context），辅助 Agent 选择正确工具。

**关键 API：** `ContextRetrieverOpenAIAgent`

```python
# 上下文：财务术语缩写解释
texts = [
    "Abbreviation: X = Revenue",
    "Abbreviation: YZ = Risk Factors",
    "Abbreviation: Z = Costs",
]
docs = [Document(text=t) for t in texts]
context_index = VectorStoreIndex.from_documents(docs)

context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    query_engine_tools,                                    # 正常工具列表
    context_index.as_retriever(similarity_top_k=1),        # 上下文检索器
    verbose=True
)

response = context_agent.chat("What is the YZ of March 2022?")
```

**工具检索器 vs 上下文检索器（重要对比）：**
| 检索器 | 作用 |
|--------|------|
| 工具检索器（9.4.4） | 从多个工具中 **筛选** 出可用相关工具（缩小选择范围） |
| 上下文检索器（9.4.5） | 检索出 **有助于选择工具的相关上下文知识**（增加提示信息） |

**示例效果：** 输入"What is the YZ of March 2022?"，上下文检索器检索出"Abbreviation: YZ = Risk Factors"插入 Prompt，帮助大模型正确选择工具。

---

## 9.5 更细粒度地控制 Agent 运行

### 9.5.0 问题

Agent 任务执行被高度封装，缺乏透明度与可控性，有"摸盲盒"感，难以观察、纠正错误、介入任务步骤。需要"分步调试"功能让 Agent 更可控。

### 9.5.1 分步可控地运行 Agent

**Agent 内部任务运行机制：**
- `AgentRunner` 先 `create_task`（构造任务）
- 进入 `run_step`（运行步骤）循环
- 具体步骤由 `AgentWorker` 主导执行
- 直至 `AgentWorker` 告诉 `AgentRunner` 已结束（`is_last=True`）

**分步运行核心代码：**

```python
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

# 1. 构造任务
task = agent.create_task("明天南京天气如何？")

# 2. 运行第一步
step_output = agent.run_step(task.task_id)
pprint.pprint(step_output.__dict__)

# 3. 循环直到 is_last=True
while not step_output.is_last:
    step_output = agent.run_step(task.task_id)
    pprint.pprint(step_output.__dict__)

# 4. 最终输出
response = agent.finalize_response(task.task_id)
print(str(response))
```

**关键 API：**
- `create_task(message)`：构造任务，返回带 `task_id` 的 task。
- `run_step(task_id)`：运行下一步，返回 `step_output`（含 `is_last` 标志）。
- `finalize_response(task_id)`：获取最终结果。

**运行过程观察：**
1. AgentWorker 推理出使用工具 `search_weather_with_args`。
2. 调用工具获得输出，`is_last=False`，继续运行。
3. AgentWorker 根据结果判断可回答，做出最终响应，`is_last=True`。
4. AgentRunner 结束循环，`finalize_response` 获取最终结果。

**查看所有步骤详情：**

```python
steps = agent.get_completed_steps(task.task_id)
for i, step in enumerate(steps):
    print(f'\nStep {i+1}:')
    print(step.__dict__)
```

### 9.5.2 在 Agent 运行中增加人类交互（Human-in-the-loop）

**核心：** 在分步运行基础上，允许在任务执行过程中接收人类反馈，控制与修改任务步骤。

**完整案例（城市信息查询 Agent）：**

```python
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker

# 准备多个城市工具
citys_dict = {'北京市':'beijing', '南京市':'nanjing', '广州市':'guangzhou',
              '上海市':'shanghai', '深圳市':'shenzhen'}

def create_city_tool(name: str):
    # 根据城市名构造查询引擎并包装成工具
    ...

query_engine_tools = [create_city_tool(city) for city in citys_dict.keys()]

# 用底层 API 构造 Agent
openai_step_engine = OpenAIAgentWorker.from_tools(query_engine_tools, verbose=True)
agent = AgentRunner(openai_step_engine)

# 交互式分步运行
task_message = None
while task_message != "exit":
    task_message = input(">> 你: ")
    if task_message == "exit":
        break

    task = agent.create_task(task_message)
    response = None
    step_output = None
    message = None

    # 任务执行中允许人类反馈
    # message="exit" → 任务被取消并退出
    # is_last=True → 任务正常退出
    while message != "exit" and (not step_output or not step_output.is_last):
        if message is None or message == "":
            step_output = agent.run_step(task.task_id)
        else:
            # 关键：允许把人类反馈信息传入中间任务步骤
            step_output = agent.run_step(task.task_id, input=message)

        if not step_output.is_last:
            message = input(">> 请补充任务反馈信息（留空继续，exit 退出）: ")

    if step_output.is_last:
        print(">> 任务运行完成。")
        response = agent.finalize_response(task.task_id)
        print(f"Final Answer: {str(response)}")
    elif not step_output.is_last:
        print(">> 任务未完成，被丢弃。")
```

**关键点：** `run_step(task_id, input=message)` —— 把人类反馈作为输入传入中间步骤，实现任务调整。

**示例效果：** 初始问题"北京的人口是多少" → 工具调用获得结果 → 人类反馈"与上海的人口做对比" → 任务调整，重新获取北京与上海信息 → 给出对比。

**核心价值（重要）：**
- 任务观察、任务取消、人工反馈、任务调整。
- 可定制个性化 AgentWorker：实现不同于 ReAct 的推理范式，只需实现对应的 `run_step` 接口并处理输出。

---

## 面试考点

### Q1：什么是 Agent？Agent 与大模型的关系？
**要点：**
- Lilian Weng 定义：`Agent = 大模型 + 记忆 + 规划技能 + 使用工具`。
- Agent 与大模型的关系 ≈ 人与大脑：大模型是"智慧大脑"，Agent 才是执行任务的完整体。
- Agent 是 AI 模型驱动、能自主理解、规划、执行、完成任务的 AI 程序，被普遍认为是生成式 AI 的终极形式。

### Q2：Data Agent 相比 RAG 增强了什么？
**要点：**
- 两大增强：① 规划与推理任务步骤的能力（循环推理 + ReAct 范式）；② 定义与使用工具的能力。
- 4 项能力：RAG 检索/生成 + 推理任务步骤 + 调用外部 API + 长短期记忆。

### Q3：OpenAIAgent 与 ReActAgent 的区别？何时用哪个？
**要点：**
- OpenAIAgent：依赖大模型 **函数调用功能**，大模型原生判断是否调工具。
- ReActAgent：基于 **ReAct 推理范式**（Thought-Action-Observation 循环），通过 Prompt 推理，**适用于不支持函数调用的大模型**。
- 模型支持函数调用时优先 OpenAIAgent（更高效、更可靠）。

### Q4：工具检索 vs 上下文检索的区别？
**要点：**
- 工具检索器（`ObjectIndex` + `tool_retriever`）：工具集过大时，**缩小工具选择范围**。
- 上下文检索器（`ContextRetrieverOpenAIAgent`）：工具易混淆/相似时，**检索相关上下文知识辅助选工具**（增加提示信息）。
- 两者都用 ObjectIndex / 向量检索，但目的不同。

### Q5：AgentRunner 与 AgentWorker 的职责划分？
**要点：**
- **AgentRunner**：顶级协调器，构造任务、调用 Worker、收集聚合结果、保存状态、跟踪任务。**有状态**。
- **AgentWorker**：真正执行任务每一步（step），返回每步输出。**无状态，只负责执行**。
- 分离构造与执行 → 支持分步调试、取消任务、定制 Worker、人工干预。

### Q6：如何分步可控地运行 Agent？
**要点：**
- `create_task` → 循环 `run_step(task_id)`（检查 `is_last`）→ `finalize_response`。
- `run_step(task_id, input=message)` 可在中间步骤传入人类反馈（human-in-the-loop）。
- `get_completed_steps` 查看所有步骤详情。

### Q7：Agent 工具的本质是什么？输出结构是什么？
**要点：**
- 工具是具备规范请求/响应参数的 API 或函数，可封装函数、查询引擎、检索器、查询管道、其他 Agent、外部 API。
- 输出统一为 `ToolOutput`：含 `content`、`tool_name`、`raw_input`、`raw_output`、`is_error`。
- 工具元数据（metadata）是帮助大模型理解工具用途、推理使用工具的关键。

### Q8：QueryPlanTool 解决什么问题？
**要点：**
- 在多个工具之上提供执行计划能力。接收工具列表 + 执行计划（DAG，由 QueryNode 组成，含 dependencies）按计划调度执行。
- 用途：① Agent 生成计划后交给它执行；② 预设计划减少大模型自行规划的不确定性。

### Q9：函数调用的完整流程是什么？
**要点：**
1. 工具元数据转函数规格（`to_openai_tool()`）传入大模型。
2. 大模型返回 `additional_kwargs["tool_calls"]`（含函数名 + JSON 参数）。
3. 解析参数 → 调用对应工具 → 结果作为 `role="tool"` 消息加入历史。
4. 把工具结果送回大模型生成最终答案。

---

## 易错 / 陷阱

1. **Agent 继承自 QueryEngine**：Agent 本身是查询引擎，可被包装成 QueryEngineTool 供其他 Agent 调用（Agent 间互调）。容易忽略这一特性。

2. **ReActAgent 不依赖函数调用**：ReActAgent 通过 **Prompt 推理**（Thought/Action/Observation）选择工具，适用于不支持函数调用的大模型。OpenAIAgent 才依赖函数调用。

3. **工具集过大风险**：工具过多会导致大模型推理能力下降甚至工具使用错乱。解决方案：用 `ObjectIndex` 给工具建索引，运行时通过 `tool_retriever` 动态检索缩小范围。

4. **OnDemandLoaderTool 性能陷阱**：即时加载+索引+查询，数据量大时响应延迟严重，只适合少量"按需"加载场景。

5. **RetrieverTool 输出 Node 而非答案**：RetrieverTool 调用输出的是检索出的 Node 内容（元数据 + 文本），不是最终答案。

6. **QueryNode 的 dependencies 字段**：构造执行计划时，`dependencies` 是 **依赖的 node id 列表**，描述 DAG 节点先后顺序，写错会导致执行顺序混乱。

7. **底层 API 开发 Agent 的等价性**：`AgentRunner(OpenAIAgentWorker(...))` 等价于直接用 `OpenAIAgent`，但前者支持分步控制。不要误以为底层 API 功能更弱。

8. **AgentWorker 无状态**：任务状态保存在 AgentRunner，AgentWorker 只负责执行。这与"Worker 应保存中间状态"的直觉相反。

9. **run_step 的 input 参数**：`run_step(task_id, input=message)` 的 `input` 是 **人类反馈信息**，传入中间步骤用于任务调整，不是新的查询。

10. **is_last 判断**：`step_output.is_last=False` 时必须继续 `run_step`，直到 `is_last=True` 才能 `finalize_response`。提前 finalize 会出错。

11. **工具元数据 description 的关键性**：description 是大模型选择工具的核心依据，描述模糊或不准确会导致工具选错。ContextRetrieverOpenAIAgent 正是用于 description 不足以区分工具时补充上下文。

12. **自定义 AgentWorker 实现要点**：实现不同于 ReAct 的推理范式时，只需实现对应的 `run_step` 接口并处理输出，无需重写整个 Agent。
