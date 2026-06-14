# 第13章 新型 RAG 范式原理与实现

> 出处：严灿平《基于大模型的 RAG 应用开发与优化——构建企业级 LLM 应用》【高级篇】第13章
> 定位：从经典 RAG 向模块化 RAG 演进的三种实验性范式——C-RAG、Self-RAG、RAPTOR。重点讲透动机、原理、实现。

> 重要前提：这些范式是**实验性探索**，自身仍在演进，非成熟产品。学习目的是理解其设计思想，在开发中灵活使用，不可生搬硬套，使用时需充分测试评估。

---

## 总览：三种范式的核心思想对比

| 范式 | 全称 | 核心机制 | 解决的核心痛点 | 关键模块 |
| --- | --- | --- | --- | --- |
| **C-RAG** | Corrective-RAG | 检索后**自纠错**：评估→过滤→查询重写→网络补充 | 检索出的上下文**相关性不足** | 评估器、查询转换、搜索 |
| **Self-RAG** | Self-RAG | 模型层**微调**输出"自省 token"，按需检索+多次生成评估择优 | **过度检索**+**输出与知识不一致** | 自省 token、logprobs 量化评估 |
| **RAPTOR** | Recursive Abstractive Processing for Tree-Organized Retrieval | 构造**从概要到细节的多层树状知识库** | 经典 RAG 只能局部细节、无法宏观综合 | 嵌入→聚类→摘要→递归 |

---

## 13.1 自纠错 RAG：C-RAG（Corrective-RAG）

> 论文："Corrective Retrieval Augmented Generation"（中科大 + Google 研究院等）

### 13.1.1 C-RAG 诞生的动机

一句话：**尽可能提高检索出的上下文相关性**。

无论索引选择、检索算法还是查询分析重写多用心，经典 RAG 在运行时仍难保证检索出的知识完全相关。C-RAG 的"C"提供的是一种**事后纠错与调整**策略。

**本质**：检索出相关知识后，**自我评估相关性，并根据评估结果自我纠错**——包括删除不相关知识、查询转换后重新检索、借助搜索引擎补充外部知识。

### 13.1.2 C-RAG 的原理（图 13-1）

相比经典 RAG，C-RAG 增加了 **3 个模块**：评估器、查询转换、搜索。

**流程**：
1. 正常语义检索出相关知识块。
2. 用**轻量级评估器**（通常是大模型）评估召回知识质量，分为三类：**相关知识 / 存疑知识 / 不相关知识**。
3. 按评估结果分流：
   - **相关知识**（至少一个 Node 相关）：直接交给大模型生成。
   - **存疑/不相关知识**：先**重写输入问题**（期望更好搜索结果），再用**网络搜索或其他方式**补充相关知识。

> 总结：评估检索知识→去除不相关→借其他途径补充相关→提高输入知识质量→回答更准确。

### 13.1.3 C-RAG 的实现

#### 1. 准备 3 个 Prompt

```python
# 生成答案
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "以下是上下文\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请仅根据上面的上下文，回答以下问题，不要编造其他内容。\n"
    "如果上下文中不存在相关信息，请拒绝回答。\n"
    "问题: {query_str}\n"
    "答案: "
)
text_qa_prompt = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)

# 评估相关性（只输出 yes/no）
EVALUATE_PROMPT_TEMPLATE="""您是一个评分人员，评估检索出的文档与用户问题的相关性。
       以下是检索出的文档：
       ----------------
       {context}
       ----------------
       以下是用户问题：
       ----------------
       {query_str}
       ----------------
       如果文档中包含与用户问题相关的关键词或语义，且有助于解答用户问题，请将其评为相关。
       请给出 yes 或 no 来表明文档是否与问题相关。
       注意只需要输出 yes 或 no，不要有多余解释。
       """
evaluate_prompt = PromptTemplate(EVALUATE_PROMPT_TEMPLATE)

# 重写输入问题
REWRITE_PROMPT_TEMPLATE= """你需要生成对检索进行优化的问题。请根据输入内容，尝试推理其中的语义意图/含义。
       这是初始问题：
       ----------------
       {query_str}
       ----------------
       请提出一个改进的问题："""
rewrite_prompt = PromptTemplate(REWRITE_PROMPT_TEMPLATE)
```

#### 2. 构造检索器

```python
def create_retriever(file):
    docs = SimpleDirectoryReader(input_files=[file]).load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index.as_retriever(similarity_top_k=3), index.as_query_engine()
```

#### 3. 相关性评估器（逐 Node 评估，返回相关的 + 是否需搜索）

```python
def evaluate_nodes(query_str: str, retrieved_nodes: List[Document]):
    # 用查询管道组装 evaluate_prompt + llm（直接用大模型也行）
    evaluate_pipeline = QueryPipeline(chain=[evaluate_prompt, llm_openai])
    filtered_nodes = []
    need_search = False
    for node in retrieved_nodes:
        relevancy = evaluate_pipeline.run(context=node.text, query_str=query_str)
        if relevancy.message.content.lower() == 'yes':
            filtered_nodes.append(node)
        else:
            need_search = True
    return filtered_nodes, need_search
```

#### 4. 重写输入问题

```python
def rewrite(query_str: str):
    new_query_str = llm_openai.predict(rewrite_prompt, query_str=query_str)
    return new_query_str
```

#### 5. 搜索工具（Tavily 网络搜索）

```python
def web_search(query_str: str):
    tavily_tool = TavilyToolSpec(api_key="tvly-***")
    search_results = tavily_tool.search(query_str, max_results=5)
    return "\n".join([result.text for result in search_results])
```

#### 6. 生成答案（因需自行组装检索+搜索的上下文，直接用大模型）

```python
def query(query_str, context_str):
    response = llm_openai.predict(text_qa_prompt, context_str=context_str, query_str=query_str)
    return response
```

#### 7. 主程序（C-RAG 完整流程）

```python
file_name = "../../data/citys/南京市.txt"
retriever, query_engine = create_retriever(file_name)
query_str = '南京市的人口数量是多少与分布情况如何？参加 2024 年中考的学生数量是多少？'

# 先测试经典 RAG 直接生成（对比）
response = query_engine.query(query_str)

# === C-RAG 流程 ===
# 1. 检索
retrieved_nodes = retriever.retrieve(query_str)

# 2. 评估，仅保留相关上下文
filtered_nodes, need_search = evaluate_nodes(query_str, retrieved_nodes)
filtered_text = "\n".join([node.text for node in filtered_nodes])

# 3. 存在不相关知识 → 重写问题 + 网络搜索
search_text = ""
if need_search:
    new_query_str = rewrite(query_str)
    search_text = web_search(new_query_str)

# 4. 组合新上下文生成
context_str = filtered_text + "\n" + search_text
response = query(query_str, context_str)
```

**测试效果**：故意加入一个无法在知识库找到的实时信息问题（"参加 2024 年中考的学生数量"）。经典 RAG 无法回答；C-RAG 过滤掉 2 个不相关 Node，重写问题后网络搜索补充，最终完整回答。

### C-RAG 的局限

- 大模型评估相关性有不确定性与模型依赖性，存在**错误过滤**可能。
- 网络搜索在严格企业场景可能引入**安全风险**。

---

## 13.2 自省式 RAG：Self-RAG

> 论文："Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"（华盛顿大学 + IBM AI 研究院等）。原型项目测试中显著优于传统 RAG。

### 13.2.1 Self-RAG 诞生的动机

即使不考虑上下文长度与响应时间，经典 RAG 仍有两大负面问题：

#### 1. 过度检索
经典 RAG 不加区分地对每个问题都 top_K 检索，可能引入无用甚至偏离内容，影响输出。

#### 2. 输出一致性问题
经典 RAG 无法确保输出结果与检索知识事实一致（大模型不能绝对遵循提示，也无法绝对保证知识相关性）。

**通俗类比**：RAG 是允许优秀学生（大模型）考试时查参考书，两个问题：
- 不管题目如何都查参考书（效率低；应快速答熟悉的，不熟悉的才查）。
- 查了很多参考书却不严格按它们回答（仍答错）。

工程上的折中（检索前判断是否需检索、Prompt 严格要求、迭代评估）会带来复杂度上升、性能下降、不可控因素增多。**Self-RAG 的解法**：在模型层微调。

### 13.2.2 Self-RAG 的原理

> Self-RAG 与 RAG 最大不同：**通过模型层微调，让大模型具备判断检索与按需检索能力**，再与应用层配合提高生成准确性与质量。

#### 1. 基本流程（图 13-4）

1. **判断检索**：大模型判断按需检索还是直接输出答案（经典 RAG 是直接检索）。
2. **按需检索**：
   - 不需检索（如"创作一首歌颂母爱的诗"）→ 大模型直接输出。
   - 需要检索（如"介绍公司最受欢迎的产品"）→ 检索器检索 top-K 块。
3. **增强生成**：逐个用检索出的 K 个块与问题组装 Prompt，生成 **K 个输出答案**。
4. **评估、选择与输出**：对 K 个输出评估，选最佳作为最终输出。

#### 2. 评估指标（表 13-1）

两个阶段需大模型评估：①是否需检索 ②如何对多个输出评估。共设计 **4 种评估指标**：

| 指标 | 输入 | 输出 | 定义 |
| --- | --- | --- | --- |
| **Retrieve** | x / x,y | {yes, no, continue} | 决定何时检索 |
| **IsREL** | x, d | {relevant, irrelevant} | d 是否提供解决 x 的有用信息 |
| **IsSUP** | x, d, y | {fully supported, partially supported, no support} | y 中可验证陈述被 d 支持的程度 |
| **IsUSE** | x, y | {5,4,3,2,1} | y 对 x 是否有用 |

**详解**：

- **Retrieve（是否需要检索）**：3 取值
  - `[No Retrieval]`：无须检索，直接生成。
  - `[Retrieval]`：需要检索。
  - `[Continue to Use Evidence]`：无须检索，继续用之前的检索内容。
- **IsREL（知识相关性）**：2 取值
  - `[Relevant]` / `[Irrelevant]`。
- **IsSUP（响应支持度）**：3 取值
  - `[Fully supported]` / `[Partially supported]` / `[No support / Contradictory]`（即编造）。
  - 例：知识只有"中国首都是北京"，输出"北京是中国首都，北京最受欢迎的景点是长城"→后半部分无支持→部分支持。
- **IsUSE（响应有效性）**：按有用程度 1~5 分，`[Utility:1]` ~ `[Utility:5]`。

#### 3. 指标生成：自省 token（关键创新）

**朴素方法**（应用层用 Prompt 判断）缺点：
- 过多大模型交互 → 响应性能下降 + token 成本升高。
- Prompt 只能定性评估，不利于量化。

**Self-RAG 方法**：**微调大模型**，让其在推理过程中**自我反省，直接输出代表指标的标记性 token**——称为"**自省 token（reflection tokens）**"。

例 1（输出携带相关性等指标）：
```
Response: [Relevant] 字节调动的 Coze 是一个大模型的应用开发平台... [Partially supported] [Utility:5]
```

例 2（输出 [Retrieval] 表示需"求助"外部知识）：
```
Response: 当然![Retrieval]<paragraph>
```

> 通过微调引入自省 token，让大模型更智能并适应工作流。该模型需特殊训练。Self-RAG 开源项目提供训练数据与过程，并提供微调后测试模型 `selfrag_llama2_7b`（借 Hugging Face Hub 使用）。

#### 4. 输出评估：基于 logprobs 量化

如何量化比较多个输出并打分选最佳？自省 token 本身非量化指标，需借助大模型推理结果中的 **`logprobs`（对数概率）** 字段。

##### 1) 了解对数概率

大模型输出本质是根据提示预测下一个 token 并循环，直到结束 token。每步并非确定下一个 token，而是输出**含多个可能 token 及其概率的列表**，最后选一个输出。`logprobs` 字段保存每步预测的多个可能 token 的输出概率（取对数）。

##### 2) 评估算法（3 种指标，Retrieve 无须量化）

设 `p(token)` 为该 token 在对应位置的输出概率（`exp(logprob)`）：

- **【IsREL】知识相关性**：
  `score = p([Relevant]) / (p([Relevant]) + p([Irrelevant]))`

- **【IsSUP】响应支持度**：
  `score = (p([Fully supported]) + 0.5 * p([Partially supported])) / (p([Fully supported]) + p([Partially supported]) + p([No support / Contradictory]))`

- **【IsUSE】响应有效性**：
  分别计算 5 个 token 概率占比乘对应权重（-1, -0.5, 0, 0.5, 1）求和：
  `score = Σ weight[i] * (p([Utility:i+1]) / Σ p)`
  其中 `weights = [-1, -0.5, 0, 0.5, 1]`。

##### IsSUP 实现示例

```python
_IS_SUPPORTED_TOKENS = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
]

def _is_supported_score(pred_tokens: List[int], pred_log_probs_dict: List[Dict[str, float]]) -> float:
    is_supported_score = 0
    # 找到自省 token 出现位置（该指标只有一个 token）
    token_appear_id = -1
    for tok_idx, token in enumerate(pred_tokens):
        if token in _IS_SUPPORTED_TOKENS:
            token_appear_id = tok_idx
            break

    if token_appear_id > -1:
        issup_score_dict = {}
        for token in _IS_SUPPORTED_TOKENS:
            prob = pred_log_probs_dict[token_appear_id][token]
            issup_score_dict[token] = np.exp(float(prob))     # 对数概率 → 概率
        is_supported_score = (
            issup_score_dict["[Fully supported]"]
            + 0.5 * issup_score_dict["[Partially supported]"]
        ) / np.sum(list(issup_score_dict.values()))
    return is_supported_score
```

**注意**：
- `logprobs` 是对数概率，计算时用 `exp` 转正常概率。
- 实际使用需参考推理工具（如 `llama_cpp`）文档，从输出参数中找到 `pred_tokens` 与 `pred_log_probs_dict`。

### 13.2.3 Self-RAG 的实现

基于 `selfrag_llama2_7b` 构建符合 Self-RAG 的原型应用。开源项目主要介绍微调，不提供应用层框架，需参考其测试/推理代码自行构建。

#### 1. 模型测试（先感受自省 token）

用 `llama_cpp` 作为本机推理工具（Ollama 当时还不支持该模型）。

```bash
pip install llama_cpp_python
pip install huggingface-hub

# 下载 gguf 版本
huggingface-cli download m4r1/selfrag_llama2_7b-GGUF \
  selfrag_llama2_7b.q4_k_m.gguf \
  --local-dir ./model --local-dir-use-symlinks False
```

```python
from llama_cpp import Llama

_MODEL_KWARGS = {"logits_all": True, "n_ctx": 2048, "n_gpu_layers": 200}
_GENERATE_KWARGS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "logprobs": 1000}

llm = Llama(model_path="./model/selfrag_llama2_7b.q4_k_m.gguf", **_MODEL_KWARGS)

# 格式化 Prompt（注意格式）
def format_prompt(input, paragraph=None):
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
    if paragraph is not None:
        prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return prompt

# 测试两个问题
query_1 = "写一首歌颂母爱的小诗"                       # 创作型，无须检索
query_2 = "能否介绍一下字节跳动的 AI 平台 Coze？"        # 事实型，需检索

for query in [query_1, query_2]:
    pred = llm(format_prompt(query), **_GENERATE_KWARGS)
    print(pred["choices"][0]["text"])        # 携带自省 token 的响应
    print(pred["choices"][0])                # 含 logprobs 详情
```

**输出观察**：
- query_1 → 含 `[No Retrieval]`（无须检索，创作问题）。
- query_2 → 含 `[Retrieval]`（需要补充知识）。

带入模拟知识后再推理：

```python
paragraph = """Coze 是字节跳动的大模型应用一站式开发平台。"""
# format_prompt 中 paragraph 默认带入
pred = llm(format_prompt(query_2), **_GENERATE_KWARGS)
# 输出: [Relevant]Coze is a platform...[Fully supported][Continue to Use Evidence]...[Utility:5]
```

带入知识后输出含 `[Relevant][Fully supported][Utility:5]` 等自省 token。

#### 2. 应用测试：自定义 Self-RAG 查询引擎

构造 `CustomQueryEngine` 实现完整 Self-RAG 工作流（重点展示多次生成后量化评估择优）。

##### 定义所有自省 token

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.response import Response
from llama_index.core.bridge.pydantic import Field
from llama_index.core.utils import print_text

_TOKENS = {
    "retrieval":  ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"],
    "relevance":  ["[Irrelevant]", "[Relevant]"],
    "support":    ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"],
    "utility":    ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"],
    "ctrl": [   # 所有控制 token（用于后处理清除）
        "[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]",
        "[Irrelevant]", "[Relevant]",
        "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
        "<paragraph>", "</paragraph>",
        "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
    ],
}
```

##### 初始化查询引擎

```python
class SelfRAGQueryEngine(CustomQueryEngine):
    def __init__(self, llm: LlamaCPP, retriever: BaseRetriever) -> None:
        super().__init__()
        self.llm = llm
        self.retriever = retriever
```

##### 核心查询方法 `query`

```python
    def query(self, query_str: str) -> str:
        # 1. 调用大模型
        response = self.llm.complete(_format_prompt(query_str))
        answer = response.text

        # 2. 判断是否需要检索
        if "[Retrieval]" in answer:
            print_text("需要检索知识，开始检索...\n", color="blue")
            documents = self.retriever.retrieve(query_str)
            # 3. 每个知识块 + 问题组装成 Prompt
            paragraphs = [_format_prompt(query_str, document.node.text) for document in documents]

            # 4. 重新生成并评估
            llm_response_per_paragraph, paragraphs_final_score = self._regen_then_eval(paragraphs)

            # 5. 选最高分
            best_paragraph_id = max(paragraphs_final_score, key=paragraphs_final_score.get)
            answer = llm_response_per_paragraph[best_paragraph_id]
        else:
            print_text("无须检索知识，直接输出答案\n", color="green")

        # 6. 后处理清除自省 token
        answer = _postprocess_answer(answer)
        return str(answer)
```

##### `_regen_then_eval`（重新生成 + 量化评估）

```python
    def _regen_then_eval(self, paragraphs: List[str]) -> Tuple[Dict[int, str], Dict[int, float]]:
        paragraphs_final_score = {}
        llm_response_text = {}
        for p_idx, paragraph in enumerate(paragraphs):
            response = self.llm.complete(paragraph)
            pred = response.raw                                # 关键：从 raw 取 logprobs
            llm_response_text[p_idx] = response.text

            logprobs = pred["choices"][0]["logprobs"]
            pred_log_probs = logprobs["top_logprobs"]

            # 计算三种得分
            isrel_score = _relevance_score(pred_log_probs[0])              # 相关性为第一个 token
            issup_score = _is_supported_score(logprobs["tokens"], pred_log_probs)
            isuse_score = _is_useful_score(logprobs["tokens"], pred_log_probs)

            # 总分（IsUSE 权重 0.5）
            paragraphs_final_score[p_idx] = isrel_score + issup_score + 0.5 * isuse_score
        return llm_response_text, paragraphs_final_score
```

##### 三个评分函数

```python
def _relevance_score(pred_log_probs: Dict[str, float]) -> float:
    rel_prob  = np.exp(float(pred_log_probs["[Relevant]"]))
    irel_prob = np.exp(float(pred_log_probs["[Irrelevant]"]))
    return rel_prob / (rel_prob + irel_prob)

def _is_useful_score(pred_tokens: List[int], pred_log_probs_dict: List[Dict[str, float]]) -> float:
    isuse_score = 0
    utility_token_appear_id = -1
    for tok_idx, tok in enumerate(pred_tokens):
        if tok in _TOKENS["utility"]:
            utility_token_appear_id = tok_idx
    if utility_token_appear_id > -1:
        ut_score_dict = {}
        for token in _TOKENS["utility"]:
            prob = pred_log_probs_dict[utility_token_appear_id][token]
            ut_score_dict[token] = np.exp(float(prob))
        ut_sum = np.sum(list(ut_score_dict.values()))
        ut_weights = [-1, -0.5, 0, 0.5, 1]
        isuse_score = np.sum([
            ut_weights[i] * (ut_score_dict[f"[Utility:{i + 1}]"] / ut_sum)
            for i in range(len(ut_weights))
        ])
    return isuse_score

def _postprocess_answer(answer: str) -> str:
    for token in _TOKENS["ctrl"]:
        answer = answer.replace(token, "")
    for sym in ["</s>", "\n", "<|endoftext|>"]:
        answer = answer.replace(sym, "")
    return answer
```

##### 主程序

```python
import os
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from pathlib import Path
from selfrag_queryengine import SelfRAGQueryEngine

_MODEL_KWARGS = {"logits_all": True, "n_ctx": 2048, "n_gpu_layers": -1}
_GENERATE_KWARGS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1000, "logprobs": 32016}
download_dir = "../../model"

# 构造测试文档（Xiaomi 14 系列事实）
documents = [Document(text="..."), ...]
index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

model_path = Path(download_dir) / "selfrag_llama2_7b.q4_k_m.gguf"
llm = LlamaCPP(model_path=str(model_path), model_kwargs=_MODEL_KWARGS, generate_kwargs=_GENERATE_KWARGS)

query_engine = SelfRAGQueryEngine(llm, retriever)

# 创作型问题（无须检索）
response = query_engine.query("write a poem about beautiful sunset")
# 事实型问题（需检索+多次生成评估）
response = query_engine.query("Tell me some truth about xiaomi 14 phone, especially about its battery and camera?")
```

**测试效果**：创作问题直接输出；事实问题检索后逐块重新生成评估，最终"脱颖而出"得分最高的答案。

### 13.2.4 Self-RAG 的优化

原型中多次生成基于一次检索的 top-K 块，实际有两个问题：
1. **检索块已按语义相似度排序**，生成得分排序常与之一致，丧失评估意义。
2. **实际知识结构复杂**，常需一次性把多个文档输入生成（保证完整+更多参考），而非一次一文档。

**优化方向（改变检索策略，多次检索 + 多次评估）**：
- 查询重写后再次检索（不同重写算法）。
- 不同检索算法获不同相关知识。
- 检索后用不同 Rerank 算法重排形成不同知识重点。

---

## 13.3 检索树 RAG：RAPTOR

> 论文："Raptor: Recursive Abstractive Processing for Tree-Organized Retrieval"（斯坦福大学）

### 13.3.1 RAPTOR 诞生的动机

经典 RAG 基于向量 top-K 召回，**只能关注局部与细节，无法关注整体与宏观语义**（除非全量输入依赖超长上下文大模型）。

例："孙悟空是如何从顽皮猴子成长为斗战胜佛的？"——需对整本《西游记》阅读理解总结，而非召回几章节就能解决。

**RAPTOR 要做的**：构造一个**从上至下、从概要到细节、从宏观层到微观层**的多层次树状知识库，帮助大模型既能答事实性细节，也能答需理解更高层知识的问题。

### 13.3.2 RAPTOR 的原理（图 13-10）

**基本思想**：
1. **叶子 Node（Leaf Node）**：原始文档解析后的知识块（图例 6 个）→ 嵌入生成向量。
2. **聚类**：用聚类算法对基础 Node 聚类（如分成 3 组），把"相关"文档分一组；给每组生成**摘要（Summary）**，基于摘要构造一组抽象程度与语义丰富度更高的新 Node（3 个）。
3. **递归**：对这 3 个新 Node 递归执行"嵌入→聚类→生成摘要 Node"，直到无法再分组（无新聚簇）。

**结果**：构造一棵完整的 Node 树。图例 10 个 Node：Node1~Node6 基础 Node，Node7~Node9 中间层，Node10 根 Node（不一定只有一个根）。高层 Node = 低层若干 Node 的总结精简版。所有 Node 的嵌入向量存向量库用于检索。

#### 两种检索模式

- **树遍历检索**（图 13-11）：从根 Node 开始，基于向量相似度与父子关系逐层向下检索，最后检索出全部相关 Node 作为上下文。
- **全量检索 / 折叠检索**（图 13-12，**推荐**）：将树完全展开成单层，直接对所有 Node 做向量相似度检索，更快且不遗漏。

#### RAPTOR 的意义
1. 不同层次多级别构造语义表示并嵌入，提高召回能力。
2. 有效回答不同层次问题（低阶/高阶 Node 各自解决）。
3. 适合解决需理解多个知识块才能回答的综合性问题。

### 13.3.3 RAPTOR 的实现

#### 1. 构造索引（核心循环）

RAPTOR 用普通向量存储索引，重点在 Node 的构造：除基础 Node 外，通过 **嵌入→聚类→生成摘要 Node** 循环迭代生成"父"Node，最后形成完整树，在所有 Node 上构造向量索引。

```python
# RaptorRetriever 的部分实现
async def insert(self, documents: List[BaseNode]) -> None:
    embed_model = self.index._embed_model
    transformations = self.index._transformations

    # 1. 对传入文档做 Node 分割（叶子 Node 基础）
    cur_nodes = run_transformations(documents, transformations, in_place=False)

    # 2. 按树层次循环构造
    for level in range(self.tree_depth):
        # 2.1 给当前 Node 生成向量
        embeddings = await embed_model.aget_text_embedding_batch(
            [node.get_content(metadata_mode="embed") for node in cur_nodes])
        id_to_embedding = {node.id_: emb for node, emb in zip(cur_nodes, embeddings)}

        # 2.2 聚类（语义相近 Node 聚到一簇）
        nodes_per_cluster = get_clusters(cur_nodes, id_to_embedding)

        # 2.3 给每个聚簇生成摘要
        summaries_per_cluster = await self.summary_module.generate_summaries(nodes_per_cluster)

        # 2.4 摘要构造新 Node（即当前 Node 的父 Node）
        new_nodes = [
            TextNode(
                text=summary,
                metadata={"level": level},
                excluded_embed_metadata_keys=["level"],
                excluded_llm_metadata_keys=["level"])
            for summary in summaries_per_cluster
        ]

        # 2.5 处理当前 Node：设 parent_id + embedding + 插入索引
        nodes_with_embeddings = []
        for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
            for node in cluster:
                node.metadata["parent_id"] = summary_doc.id_
                node.excluded_embed_metadata_keys.append("parent_id")
                node.excluded_llm_metadata_keys.append("parent_id")
                node.embedding = id_to_embedding[node.id_]
                nodes_with_embeddings.append(node)
        self.index.insert_nodes(nodes_with_embeddings)

        # 2.6 以父 Node 为新当前 Node 进入下次循环（此时父 Node 还未插入索引）
        cur_nodes = new_nodes

    # 3. 迭代结束后把最后一次的父 Node 插入索引
    self.index.insert_nodes(cur_nodes)
```

#### 2. 实现聚类与摘要

**生成摘要**：用 LlamaIndex `tree_summarize` 类型响应生成器：

```python
async def generate_summaries(self, documents_per_cluster: List[List[BaseNode]]) -> List[str]:
    responses = []
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", use_async=True, llm=llm)
    jobs = []
    for documents in documents_per_cluster:
        with_scores = [NodeWithScore(node=doc, score=1.0) for doc in documents]
        response = response_synthesizer.asynthesize(self.summary_prompt, with_scores)
        responses.append(response)
    return [str(response) for response in responses]
```

**聚类**：借助现成 Python 模块（如 `scikit-learn`，图 13-13）。

#### 3. 实现检索（推荐全量检索）

RAPTOR 索引无特殊之处，就是普通向量存储索引，检索非常简单：

```python
async def collapsed_retrieval(self, query_str: str) -> Response:
    return await self.index.as_retriever(similarity_top_k=3).aretrieve(query_str)
```

> LlamaIndex 官方 LlamaHub 的 LlamaIndex Packs 库中有这些范式的第三方实现代码包，是很好的学习研究材料。

---

## 【面试考点】

### Q1：三种新型 RAG 范式各自解决什么问题？
**要点**：
- **C-RAG**：检索后相关性不足 → 事后自纠错（评估→过滤→重写→网络补充）。
- **Self-RAG**：过度检索 + 输出与知识不一致 → 模型层微调输出自省 token，按需检索 + 多次生成评估择优。
- **RAPTOR**：经典 RAG 只能局部细节 → 构造多层树状知识库（嵌入→聚类→摘要→递归），兼顾宏观与细节。

### Q2：C-RAG 的完整流程？3 个新增模块？
**要点**：检索→评估器评估（相关/存疑/不相关）→分流：相关直接生成；存疑/不相关→查询转换重写→网络搜索补充→组合生成。3 个新增模块：评估器、查询转换、搜索。

### Q3：Self-RAG 的 4 种评估指标？取值？
**要点**：
- Retrieve：[No Retrieval] / [Retrieval] / [Continue to Use Evidence]
- IsREL：[Relevant] / [Irrelevant]
- IsSUP：[Fully supported] / [Partially supported] / [No support / Contradictory]
- IsUSE：[Utility:1]~[Utility:5]

### Q4：Self-RAG 为什么用"自省 token"而不是 Prompt 判断？
**要点**：Prompt 判断需多次大模型交互（性能下降+成本升高）且只能定性；微调让大模型推理时自我反省直接输出标记性 token，定量可评估、减少应用层复杂度、不降低大模型能力。

### Q5：Self-RAG 如何量化评估多个输出？logprobs 是什么？
**要点**：用大模型输出的 `logprobs`（对数概率）字段。大模型每步预测下一个 token 时输出多个可能 token 及其概率列表。在自省 token 出现位置取该指标的各 token 概率（用 `exp` 转正常概率），按公式加权计算得分（IsREL、IsSUP、IsUSE；Retrieve 无须量化）。总分 = IsREL + IsSUP + 0.5×IsUSE，选最高分输出。

### Q6：IsSUP 的得分公式？
**要点**：`(p([Fully supported]) + 0.5×p([Partially supported])) / (p([Fully supported]) + p([Partially supported]) + p([No support / Contradictory]))`。

### Q7：RAPTOR 怎么构造树？
**要点**：叶子 Node（原始块）→ 嵌入 → 聚类（语义相近分组）→ 每簇生成摘要 Node → 摘要 Node 作为父 Node → 递归执行直至无法再分簇。所有 Node 向量存向量库。

### Q8：RAPTOR 两种检索模式？
**要点**：①树遍历检索（从根逐层向下，基于相似度+父子关系）；②全量检索/折叠检索（树展开成单层，对所有 Node 向量相似度检索，**推荐**，更快不遗漏）。

### Q9：Self-RAG 实现中为什么用 CustomQueryEngine？
**要点**：需对检索与生成做个性化精确控制（判断 [Retrieval] → 检索 → 多次生成 → logprobs 评估 → 择优），框架默认查询引擎无法满足，故继承 `CustomQueryEngine` 实现 `query` 方法。

### Q10：这些范式的局限？
**要点**：
- C-RAG：大模型评估有不确定性（可能错误过滤）；网络搜索有企业安全风险。
- Self-RAG：需特殊微调模型；多次生成评估耗时；得分排序可能与语义相似度排序一致丧失评估意义。
- RAPTOR：聚类质量依赖嵌入与聚类算法；摘要可能失真；构造树成本（多次大模型摘要）高。
- 共同：均为实验性，需充分测试评估，不可生搬硬套。

---

## 【易错 / 陷阱】

1. **自省 token 是模型层能力，需微调模型**：不能在任意大模型上直接用 Self-RAG，必须用 `selfrag_llama2_7b` 这类微调模型；普通大模型不会输出 `[Retrieval]`/`[Relevant]` 等 token。
2. **`logits_all` 和 `logprobs` 必须开启**：`llama_cpp` 推理时必须 `_MODEL_KWARGS={"logits_all": True}` 且 `_GENERATE_KWARGS={"logprobs": N}`，否则无 logprobs 字段无法量化评估。
3. **`exp` 转换**：logprobs 是对数概率，计算得分前必须用 `np.exp(float(prob))` 转正常概率。
4. **自省 token 出现位置定位**：IsSUP/IsUSE 需先遍历 `pred_tokens` 找到自省 token 出现的位置 `token_appear_id`，再在该位置取该指标所有 token 的概率；IsREL 在第一个 token 位置直接取。
5. **IsUSE 权重是 [-1, -0.5, 0, 0.5, 1]**：注意负权重（[Utility:1][Utility:2] 为负），最终总分 = IsREL + IsSUP + 0.5×IsUSE。
6. **后处理必须清除所有自省 token**：`_postprocess_answer` 要清除 `_TOKENS["ctrl"]` 中所有 token 以及 `</s>`、`<|endoftext|>`、`\n` 等，否则答案带乱码 token。
7. **C-RAG 的"相关"判断是逐 Node 的**：`need_search` 只要有一个 Node 不相关就为 True，触发重写+搜索；不要整体判断。
8. **C-RAG 网络搜索结果与过滤后检索结果要拼接**：`context_str = filtered_text + "\n" + search_text`，注意 search_text 初始为空（无 need_search 时不拼接）。
9. **RAPTOR 父 Node 在循环内不立即插入索引**：循环中只插入当前层 Node（带 embedding），父 Node 作为下一轮的 cur_nodes；循环结束才把最后一批父 Node 插入。
10. **RAPTOR 推荐全量检索而非树遍历**：全量检索更快不遗漏；树遍历逐层向下易遗漏且慢。
11. **RAPTOR 聚类质量决定树质量**：聚类依赖嵌入质量与聚类算法（如 scikit-learn），聚类差则摘要失真、检索效果差。
12. **三种范式均为实验性**：不可直接上生产，必须充分评估；优先学习设计思想，结合实际灵活使用。
13. **Self-RAG 多次生成基于一次检索 top-K 有缺陷**：得分排序常与语义相似度排序一致；优化应改检索策略（查询重写、不同检索算法、不同 Rerank）做多次检索 + 多次评估。
14. **Ollama 不支持 selfrag_llama2_7b**：需用 `llama_cpp` 推理；模型需下载 gguf 版本（`m4r1/selfrag_llama2_7b-GGUF`）。
