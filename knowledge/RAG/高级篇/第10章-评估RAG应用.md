# 第 10 章 评估 RAG 应用

> 来源：严灿平《基于大模型的 RAG 应用开发与优化——构建企业级 LLM 应用》【高级篇】
> 定位：RAG 应用上线前的科学评估方法、指标体系与 LlamaIndex 评估组件实践。

## 本章总览

传统软件上线前需要软件测试（用例设计、单元/集成/压力测试）。对于 RAG 应用（含 Agent），这个阶段 **举足轻重，甚至比传统应用更重要**。

**本章重点：**
1. 为什么 RAG 应用需要评估（10.1）
2. 评估依据与指标体系（10.2）—— faithfulness / relevancy / context recall 等
3. 评估流程与方法（10.3）
4. 评估检索质量（10.4）—— RetrieverEvaluator
5. 评估响应质量（10.5）—— 单次评估 / 批量评估
6. 基于自定义标准的评估（10.6）—— GuidelineEvaluator

---

## 10.1 为什么 RAG 应用需要评估

将基于大模型的 RAG 应用投入生产前，必须提前考虑与应对 4 个问题：

1. **大模型输出的不确定性带来不可预知性**：投入生产前需科学测试衡量这种不可预知性。
2. **持续维护需科学、快速、可复用的衡量手段**：例如回答置信度上升了 10% 还是下降了 5%？
3. **知识库是动态的**：不断维护中可能产生新的 **知识干扰**，需定期检测与重新评估。
4. **基础大模型依赖**：如何在大量商业/开源大模型中选择最适合的？大模型升级一次版本对 RAG 应用影响多大？

---

## 10.2 RAG 应用的评估依据与指标

### 10.2.1 评估依据（评估器输入）

基于大模型的 RAG 应用与传统应用区别：传统应用输出确定且易衡量（如确定数值）；RAG 应用输入输出都是自然语言，需要更智能的工具与评估模型。

**评估模块输入 4 要素：**

| 要素 | 含义 | 来源 |
|------|------|------|
| 输入问题（question） | 用户的输入问题 | 用户 |
| 生成的答案（answer） | RAG 应用的输出，即待评估答案 | RAG 应用 |
| 上下文（context） | 增强 RAG 输出的参考上下文 | 检索阶段生成 |
| 参考答案（reference_answer） | 输入问题的真实正确答案 | 通常需人类标注 |

### 10.2.2 评估指标体系（表 10-1，重要）

| 名称 | 相关输入 | 解释 |
|------|---------|------|
| **正确性（Correctness）** | answer、reference_answer | 生成的答案与参考答案的匹配度，涵盖回答的语义相似度 + 事实相似度 |
| **语义相似度（Semantic Similarity）** | answer、reference_answer | 生成的答案与参考答案的语义相似度（基于 embedding） |
| **忠实度（Faithfulness）** | answer、context | 生成的答案与检索上下文的一致性，即答案能否从上下文推理出来。**是否存在幻觉问题** |
| **上下文相关性（Context Relevancy）** | context、question | 检索上下文与输入问题的相关性，即上下文中有多少内容与问题相关 |
| **答案相关性（Answer Relevancy）** | answer、question | 生成的答案与输入问题的相关性，即答案是否完整且不冗余地回答了问题（**不考虑答案正确性**） |
| **上下文精度（Context Precision）** | context、reference_answer | 检索上下文中与参考答案相关的条目是否排名较高 |
| **上下文召回率（Context Recall）** | context、reference_answer | 检索上下文与参考答案的一致性，即参考答案能否归因到上下文 |

**核心区分（重要记忆点）：**
- **faithfulness（忠实度）**：answer 是否忠于 context（**防幻觉**）。
- **context relevancy**：context 是否与 question 相关（**检索精度**）。
- **answer relevancy**：answer 是否完整回答了 question（**生成质量**，不看对错）。
- **correctness**：answer 是否与 reference_answer 一致（**对错**，需要参考答案）。

**两类指标：**
- **需要 reference_answer**：correctness、semantic similarity、context precision、context recall。
- **不需要 reference_answer**：faithfulness、context relevancy、answer relevancy。

---

## 10.3 RAG 应用的评估流程与方法

**评估流程（图 10-1）：**

1. **确定** 评估的目的、维度与指标。
2. **准备评估数据集**：可自行准备标注，也可用大模型生成。根据指标可能需要不同 question 与 reference_answer。
3. **输入 RAG 应用**：获得检索结果（context）与生成的答案（answer）。
4. **计算指标**：将评估依据输入评估器，计算各类指标，分析整体性能。

**组件级评估侧重：** 检索（Retrieval）与生成（Generation）两个最关键阶段，单独评估可细致观察分析问题，有针对性优化增强。

---

## 10.4 评估检索质量

### 10.4.0 概念与指标

利用检索评估组件 **`RetrieverEvaluator`** 对任何检索模块评估。**主要指标：**

| 指标 | 含义 |
|------|------|
| **命中率（hit_rate）** | 检索出的上下文对期望上下文的命中率 |
| **平均倒数排名（mrr, Mean Reciprocal Rank）** | 衡量检索出上下文的排名质量 |
| **cohere 重排相关性（cohere-rerank-relevancy）** | 用 Cohere Rerank 模型排名结果衡量检索排名质量 |

**评估依据：** 输入问题 + 期望的上下文（检索出的 Node）。

**两种评估方式：**
- `evaluate`：对单次查询评估。
- `evaluate_dataset`：对构造好的检索评估数据集批量评估。

### 10.4.1 生成检索评估数据集

利用 **`generate_question_context_pairs`** 借助大模型从已有数据生成一系列"输入问题 + 参考上下文"数据对（也可自行构造）。

```python
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.core.node_parser import SentenceSplitter

# 读取文档、构造 Node
documents = SimpleDirectoryReader(input_files=["../../data/citys/南京市.txt"]).load_data()
node_parser = SentenceSplitter(chunk_size=1024)
nodes = node_parser.get_nodes_from_documents(documents)
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

# 准备检索器（后面使用）
vector_index = VectorStoreIndex(nodes)
retriever = vector_index.as_retriever(similarity_top_k=2)

# 中文化 Prompt
QA_GENERATE_PROMPT_TMPL = """
以下是上下文:
---------------------
{context_str}
---------------------
你是一位专业教授。你的任务是基于以上的上下文，为即将到来的考试设置
{num_questions_per_chunk} 个问题。
这些问题必须基于提供的上下文生成，并确保上下文能够回答这些问题。确保每一行都只有
一个独立的问题。不要有多余解释。不要给问题编号。"
"""

qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm_ollama,
    num_questions_per_chunk=1,
    qa_generate_prompt_tmpl=QA_GENERATE_PROMPT_TMPL
)

# 保存到本地，避免重复生成
qa_dataset.save_json("retriever_eval_dataset.json")
```

**关键点：**
- `num_questions_per_chunk`：每个 Node 生成的问题数量。
- 自定义中文 `qa_generate_prompt_tmpl` 用于生成中文问题。
- 保存为 JSON 减少重复生成开销。

**加载并查看：**

```python
qa_dataset = EmbeddingQAFinetuneDataset.from_json("retriever_eval_dataset.json")
eval_querys = list(qa_dataset.queries.items())
for eval_id, eval_query in eval_querys[:10]:
    print(f"Query: {eval_query}")
```

**重要字段：** `qa_dataset.relevant_docs[eval_id]` —— 该输入问题 **期望关联的文档**（用于评估检索是否召回正确 Node）。

### 10.4.2 运行评估检索过程的程序

**基本流程：**
1. 加载检索评估数据集。
2. 构造 `RetrieverEvaluator`，设置评估指标。
3. 调用 `evaluate` 查看评估结果。

```python
from llama_index.core.evaluation import RetrieverEvaluator

qa_dataset = EmbeddingQAFinetuneDataset.from_json("retriever_eval_dataset.json")
querys = list(qa_dataset.queries.items())

# 构造检索评估器，设定评估指标
metrics = ["mrr", "hit_rate"]
retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)

# 逐个评估（前 10 个）
for eval_id, eval_query in eval_querys[:10]:
    expect_docs = qa_dataset.relevant_docs[eval_id]
    print(f"Query: {eval_query}, Expected docs: {expect_docs}")

    eval_result = retriever_evaluator.evaluate(
        query=eval_query,
        expected_ids=expect_docs
    )
    print(eval_result)
```

**关键 API：**
- `RetrieverEvaluator.from_metric_names(metrics, retriever=...)`：用指标名列表构造评估器。
- `evaluate(query=..., expected_ids=...)`：单次评估，输入问题 + 期望 Node id。
- `evaluate_dataset(qa_dataset)`：批量评估（与逐个循环 evaluate 等价但更简洁）。

```python
eval_results = retriever_evaluator.evaluate_dataset(qa_dataset)
```

---

## 10.5 评估响应质量

**响应质量是 RAG 应用评估的重点**，关系端到端客户体验。可对单指标/单次响应评估，也可在批量数据集上自动化运行多个响应评估器。

### 10.5.1 生成响应评估数据集

**与检索评估数据集的区别：** 响应评估除输入问题 + 检索上下文，还需大模型生成的答案，甚至参考答案。

用 **`RagDatasetGenerator`** 的 `generate_dataset_from_nodes` 批量生成响应评估数据集：

```python
from llama_index.core.evaluation import RagDatasetGenerator, LabelledRagDataset

docs = SimpleDirectoryReader(input_files=['../../data/citys/南京市.txt']).load_data()

dataset_generator = RagDatasetGenerator.from_documents(
    documents=docs,
    llm=llm_ollama,
    num_questions_per_chunk=1,
    show_progress=True,
    question_gen_query="您是一位老师。您的任务是为即将到来的考试设置"
        "{num_questions_per_chunk}个问题。这些问题必须基于提供的上下文生成，并确保"
        "上下文能够回答这些问题。确保每一行都只有一个独立的问题。不要有多余解释。"
        "不要给问题编号。"
)

# 只需运行一次
rag_dataset = dataset_generator.generate_dataset_from_nodes()
rag_dataset.save_json('./rag_eval_dataset.json')

# 加载并查看
rag_dataset = LabelledRagDataset.from_json('./rag_eval_dataset.json')
for example in rag_dataset.examples:
    print(f'query: {example.query}')
    print(f'answer: {example.reference_answer}')
```

**评估用例数据格式（重要）：**
- `query`：生成的问题。
- `reference_contexts`：检索出的参考上下文。
- `reference_answers`：参考答案。

### 10.5.2 单次响应评估

构造对应评估器（Evaluator）→ 输入必需数据 → 获得评估结果。

**单次响应评估输入参数：**
- `query`：输入问题。
- `response`：RAG 应用的响应结果。用 `evaluate_response` 可直接传 response；用 `evaluate` 需从 response 提取上下文与文本分别传入。
- `reference`：参考答案，**正确性与相似度评估需要**，对应 `reference_answer` 字段。

```python
query_engine = _create_doc_engine('Nanjing')
query = "南京的气候怎么样？"
response = query_engine.query(query)

# 1. 忠实度（防幻觉）—— 不需 reference
evaluator = FaithfulnessEvaluator()
eval_result = evaluator.evaluate_response(query=query, response=response)
print(f'faithfulness score: {eval_result.score}\n')

# 2. 相关性（综合上下文相关性 + 答案相关性）—— 不需 reference
evaluator = RelevancyEvaluator()
eval_result = evaluator.evaluate_response(query=query, response=response)
print(f'relevancy score: {eval_result.score}\n')

# 3. 上下文相关性 —— 不需 reference
evaluator = ContextRelevancyEvaluator()
eval_result = evaluator.evaluate_response(query=query, response=response)
print(f'context relevancy score: {eval_result.score}\n')

# 4. 答案相关性 —— 不需 reference
evaluator = AnswerRelevancyEvaluator()
eval_result = evaluator.evaluate_response(query=query, response=response)
print(f'answer relevancy score: {eval_result.score}\n')

# 5. 正确性 —— 需要 reference
evaluator = CorrectnessEvaluator()
eval_result = evaluator.evaluate_response(
    query=query, response=response,
    reference='南京的气候属于较典型的北亚热带季风气候...'   # 标准答案
)
print(f'correctness score: {eval_result.score}\n')

# 6. 语义相似度（基于 embedding）—— 需要 reference
evaluator = SemanticSimilarityEvaluator()
eval_result = evaluator.evaluate_response(
    query=query, response=response,
    reference='南京四季分明，冬夏温差较大...'
)
print(f'semantic similarity score: {eval_result.score}\n')
```

**核心评估器对照表：**

| 评估器 | 指标 | 是否需 reference | 输入 |
|--------|------|----------------|------|
| `FaithfulnessEvaluator` | 忠实度 | 否 | query, response |
| `RelevancyEvaluator` | 综合相关性 | 否 | query, response |
| `ContextRelevancyEvaluator` | 上下文相关性 | 否 | query, response |
| `AnswerRelevancyEvaluator` | 答案相关性 | 否 | query, response |
| `CorrectnessEvaluator` | 正确性 | **是** | query, response, reference |
| `SemanticSimilarityEvaluator` | 语义相似度 | **是** | query, response, reference |

**重要陷阱：** 需 reference 的评估器（correctness、semantic similarity）**不传 reference 会异常**。

### 10.5.3 批量响应评估

借助 **`BatchEvalRunner`** 在评估数据集上 **并行** 运行多个响应评估器，计算统计综合性能。

```python
from llama_index.core.evaluation import BatchEvalRunner
import asyncio
import pandas as pd

query_engine = _create_doc_engine('Nanjing')

# 构造多个响应评估器
faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()
correctness_evaluator = CorrectnessEvaluator()
similartiy_evaluator = SemanticSimilarityEvaluator()

# 加载数据集
rag_dataset = LabelledRagDataset.from_json('./rag_eval_dataset.json')

# 构造批量评估器（关键：workers 并行）
runner = BatchEvalRunner(
    {
        "faithfulness": faithfulness_evaluator,
        "relevancy": relevancy_evaluator,
        "correctness": correctness_evaluator,
        "similarity": similartiy_evaluator,
    },
    workers=4
)

# 异步并行评估
async def evaluate_queries():
    eval_results = await runner.aevaluate_queries(
        query_engine,
        queries=[example.query for example in rag_dataset.examples][:10],
        reference=[example.reference_answer for example in rag_dataset.examples][:10],
    )
    return eval_results

eval_results = asyncio.run(evaluate_queries())

# 结果展示
def display_results(eval_results):
    data = {}
    for key, results in eval_results.items():
        scores = [result.score for result in results]
        scores.append(sum(scores) / len(scores))   # 追加平均值
        data[key] = scores
    data["query"] = [result.query for result in eval_results["faithfulness"]]
    data["query"].append("【Average】")
    df = pd.DataFrame(data)
    print(df)

display_results(eval_results)
```

**关键点：**
- `BatchEvalRunner` 构造时 dict 的 key 是指标名，value 是评估器。
- `workers` 控制并行度，缩短评估时间。
- `aevaluate_queries`（异步）输入：query_engine、批量 queries、批量 reference。
- 结果可输出到表格（pandas）/ Excel 直观观察。

---

## 10.6 基于自定义标准的评估

前面的评估基于固定通用指标。企业级应用常有 **特殊响应要求**，希望制定评估指南/标准 → 用 **`GuidelineEvaluator`** 设置个性化标准。

```python
from llama_index.core.evaluation import GuidelineEvaluator

GUIDELINES = [
    "答案应该完全回答了输入问题。",
    "答案应该避免模糊或含糊不清的用词。",
    "答案应该在可能时使用明确的统计数据或数字。"
]

evaluators = [
    GuidelineEvaluator(
        guidelines=guideline,
        eval_template=myprompts.MY_GUILD_EVAL_TEMPLATE   # 可自定义 Prompt 模板
    )
    for guideline in GUIDELINES
]

for guideline, evaluator in zip(GUIDELINES, evaluators):
    eval_result = evaluator.evaluate_response(
        query="南京有多少人口？南京的气候怎么样？",
        response=response
    )
    print("====================================")
    print(f"Guideline: {guideline}")
    print(f"Pass: {eval_result.passing}")        # 是否通过
    print(f"Feedback: {eval_result.feedback}")   # 反馈
```

**关键点：**
- 每条 guideline 构造一个独立评估器。
- 输出含 `passing`（是否通过）和 `feedback`（反馈）。
- 可通过 `eval_template` 自定义评估 Prompt。
- **适用场景：** 企业级应用存在特殊响应要求（合规性、风格、数据使用等）时使用。

---

## 面试考点

### Q1：RAG 应用为什么需要评估？与传统软件测试有何不同？
**要点：**
- 4 大需求：① 大模型输出不确定性需衡量；② 持续维护需可复用衡量手段；③ 动态知识库产生知识干扰需定期重评；④ 大模型选型/版本升级影响需衡量。
- **与传统区别**：传统输出确定易衡量（如数值）；RAG 输入输出都是自然语言，需借助更智能的工具与评估模型。

### Q2：faithfulness、context relevancy、answer relevancy、correctness 的区别？（高频）
**要点：**
- **faithfulness（忠实度）**：answer 是否忠于 context（**防幻觉**），输入 answer+context。
- **context relevancy**：context 是否与 question 相关（**检索精度**），输入 context+question。
- **answer relevancy**：answer 是否完整回答了 question（**生成质量，不看对错**），输入 answer+question。
- **correctness（正确性）**：answer 是否与 reference_answer 一致（**对错**），输入 answer+reference_answer。
- **记忆口诀**：faithfulness 防幻觉、context relevancy 看检索、answer relevancy 看生成、correctness 看对错。

### Q3：哪些指标需要 reference_answer？
**要点：**
- 需要 reference_answer：correctness、semantic similarity、context precision、context recall。
- 不需要：faithfulness、context relevancy、answer relevancy。
- 实践中：无参考答案时只能评 faithfulness/relevancy 类指标；有标注答案才能评 correctness。

### Q4：检索质量评估有哪些指标？mrr 是什么？
**要点：**
- 检索评估指标：hit_rate（命中率）、mrr（平均倒数排名）、cohere-rerank-relevancy。
- **mrr（Mean Reciprocal Rank）**：衡量检索结果排名质量。对每个查询取第一个相关结果的倒数排名（如排第 1 = 1.0，第 2 = 0.5，第 3 = 0.33），再对所有查询取平均。mrr 越接近 1 表示相关结果排名越靠前。

### Q5：RetrieverEvaluator 如何使用？评估依据是什么？
**要点：**
- `RetrieverEvaluator.from_metric_names(["mrr","hit_rate"], retriever=retriever)` 构造。
- `evaluate(query=..., expected_ids=...)` 单次评估，输入问题 + 期望 Node id 列表。
- `evaluate_dataset(qa_dataset)` 批量评估。
- 评估依据：输入问题 + 期望的上下文（relevant_docs）。

### Q6：如何生成评估数据集？
**要点：**
- 检索评估数据集：`generate_question_context_pairs(nodes, llm=..., num_questions_per_chunk=...)`，输出 `EmbeddingQAFinetuneDataset`。
- 响应评估数据集：`RagDatasetGenerator.from_documents(...)` + `generate_dataset_from_nodes()`，输出 `LabelledRagDataset`（含 query / reference_contexts / reference_answers）。
- 均可自定义中文 Prompt，保存为 JSON 避免重复生成。

### Q7：单次响应评估 vs 批量响应评估？
**要点：**
- 单次：构造单个评估器（如 `FaithfulnessEvaluator`）→ `evaluate_response(query, response, [reference])`。
- 批量：`BatchEvalRunner({指标名: 评估器}, workers=N)` → `aevaluate_queries(query_engine, queries, reference)` 异步并行。
- 批量评估用 `workers` 控制并行度，缩短评估时间，结果可统计汇总（如平均值）。

### Q8：自定义标准评估解决什么问题？
**要点：**
- `GuidelineEvaluator` 允许制定个性化评估指南（如"答案应完全回答问题""避免模糊用词""使用明确数据"）。
- 输出 `passing`（是否通过）+ `feedback`（反馈）。
- 适用企业级应用的特殊响应要求（合规、风格、数据使用规范等）。
- 每条 guideline 构造独立评估器，可自定义 `eval_template`。

---

## 易错 / 陷阱

1. **faithfulness 与 correctness 混淆**：faithfulness 看的是 answer vs **context**（防幻觉），correctness 看的是 answer vs **reference_answer**（对错）。两者完全不同，correctness 必须有标注答案。

2. **answer relevancy 不考虑答案正确性**：answer relevancy 只看 answer 是否完整、不冗余地回答了 question，**不看答案对错**。答案可能完全错误但 answer relevancy 仍高（只要它"看起来"回答了问题）。

3. **需 reference 的评估器不传 reference 会异常**：`CorrectnessEvaluator` 和 `SemanticSimilarityEvaluator` 必须传 `reference` 参数，否则报错。

4. **RelevancyEvaluator 是综合指标**：`RelevancyEvaluator` 综合了上下文相关性与答案相关性，不是单一指标。如需单独评估要用 `ContextRelevancyEvaluator` / `AnswerRelevancyEvaluator`。

5. **evaluate vs evaluate_response**：`evaluate_response` 直接传 response 对象；`evaluate` 需从 response 提取上下文与文本分别作为参数传入。两者输入方式不同。

6. **检索评估的 expected_ids**：`evaluate(query, expected_ids=...)` 的 `expected_ids` 是 **期望检索出的 Node id 列表**（来自 `qa_dataset.relevant_docs[eval_id]`），不是问题文本。

7. **num_questions_per_chunk 的含义**：是 **每个 Node** 生成的问题数量，不是总问题数。Node 多则问题多。

8. **批量评估 reference 列表顺序**：`aevaluate_queries` 的 `queries` 和 `reference` 列表必须 **一一对应**（顺序、数量一致），错位会导致评估错误。

9. **mrr 的计算方向**：mrr 取 **倒数排名**（reciprocal），排第 1 = 1.0 最高，排越后值越小。不要误以为是"排名值"。

10. **评估数据集生成是消耗 LLM 调用的**：用大模型生成评估数据集会消耗大量 token，建议生成后保存 JSON 复用，避免每次重新生成。

11. **hit_rate 与 mrr 关注点不同**：hit_rate 关注"是否召回"（二值），mrr 关注"召回后排名好不好"（位置质量）。两者互补。

12. **GuidelineEvaluator 每条标准独立评估**：一条 guideline 对应一个评估器实例，多个标准要构造多个评估器分别评估（zip 遍历），不是一个评估器评估多条标准。
