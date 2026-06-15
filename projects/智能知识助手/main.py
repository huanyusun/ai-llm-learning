# -*- coding: utf-8 -*-
"""
================================================================================
智能知识助手 Intelligent Knowledge Assistant
================================================================================
【综合实战项目】把前面 01-06 模块学到的三块核心能力整合成一个完整应用：

    01-Transformer / 02-LLM基础  --> 提供了"模型"概念（这里用一个脚本化 LLM 模拟）
    03-Prompt工程                --> 结构化 Prompt（系统/记忆/上下文/指令分层）
    04-RAG                       --> 知识检索（纯 numpy 向量库 + 余弦相似度）
    05-Agent                     --> ReAct 推理循环（Thought → Action → Observation）
    06-微调与部署                 --> 这里是"如何接真实模型/部署"的入口（见 README）

最终形态：一个能"多轮对话 + 按需检索 + 调用工具推理"的知识助手。
    - 用户提问
    - Agent 判断：要不要查知识库？要不要算？
    - 检索（RAG）→ 观察
    - 计算（工具）→ 观察
    - 综合给出回答，并记住上下文，支持追问

【为什么是"模拟版"】
    目标是零外部依赖、直接 uv run 跑通、把整合思路看得清清楚楚。
    三个"可替换点"用注释标出：
        LLM        : mock_llm_step   -> 换 client.chat.completions.create(...)
        Embedding  : embed (哈希词袋) -> 换 sentence-transformers
        向量库     : NumpyVectorStore -> 换 chromadb / faiss
    换掉这三处，整套骨架立刻变成真实可用的 Agent。

【运行】
    uv run --directory /Users/sunhuanyu/ai-llm-learning python "projects/智能知识助手/main.py"
================================================================================
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ============================================================================
# 配置
# ============================================================================
EMBED_DIM = 256          # toy embedding 维度（哈希桶数）
TOP_K = 3                # RAG 检索返回的 chunk 数
MAX_STEPS = 6            # ReAct 单轮最大步数（论文 HotpotQA 设 7）
RELEVANCE_THRESHOLD = 0.20  # toy embedding 余弦阈值，低于此认为"未检索到相关内容"


# ============================================================================
# 模块一：RAG —— 知识库构建与检索（对应 04-RAG/mini_rag）
# ============================================================================

# ---- 1.1 内置知识库（替代外部文档；真实场景从文件/数据库加载）------------------
KNOWLEDGE_DOCS: list[str] = [
    # 来源：模拟一份"AI 入门常识"文档
    "RAG 的全称是 Retrieval-Augmented Generation，即检索增强生成。"
    "它通过在生成前先从外部知识库检索相关片段，来缓解大模型的幻觉问题。"
    "RAG 的典型流水线包含六步：加载、分块、嵌入、存储、检索、生成。",

    "ReAct 是 2022 年由 Yao 等人提出的 Agent 范式，让模型交替进行推理 Reasoning 和行动 Acting。"
    "ReAct 的循环是 Thought → Action → Observation → ... → Finish。"
    "ReAct 论文发表在 arXiv，编号 2210.03629。",

    "Prompt 工程的核心是用结构化文本引导模型行为。"
    "常见技巧包括：角色设定、few-shot 示例、思维链 Chain-of-Thought、输出格式约束。"
    "好的 Prompt 能显著提升模型在特定任务上的表现。",

    "余弦相似度是衡量两个向量方向接近程度的指标，取值范围 [-1, 1]。"
    "在 RAG 中通常把文本嵌入向量做 L2 归一化，归一化后点积等于余弦相似度。"
    "它比欧氏距离更适合文本检索，因为它关注方向而非绝对长度。",

    "Transformer 是 2017 年 Google 提出的神经网络架构，核心是自注意力机制 Self-Attention。"
    "它摆脱了 RNN 的序列依赖，可以高度并行训练，是大语言模型的基础。"
    "GPT 系列只用了 Transformer 的 Decoder 部分。",

    "大模型的幻觉 Hallucination 是指模型生成看似合理但实际错误的内容。"
    "缓解幻觉的方法包括：RAG 检索外部知识、提高温度参数的稳定性、添加事实核查、约束输出。",
]


# ---- 1.2 文本分块（复用 04-RAG 的分句思路）------------------------------------
_PUNCT = re.compile(r"[，。！？；：、”“‘’（）《》【】\s]+")


def _split_by_sentences(text: str, n: int = 2, overlap: int = 1) -> list[str]:
    """按中文句末标点切句，每 n 句一个 chunk，相邻重叠 overlap 句。"""
    sentences = [s.strip() for s in re.split(r"(?<=[。！？\n])\s*", text) if s.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    step = max(1, n - overlap)
    i = 0
    while i < len(sentences):
        chunks.append("".join(sentences[i : i + n]))
        i += step
    return chunks


# ---- 1.3 toy embedding：词袋 + 哈希桶 + L2 归一化（复用 04-RAG 思路）----------
def _tokenize(text: str) -> list[str]:
    cleaned = _PUNCT.sub("", text)
    unigrams = list(cleaned)
    bigrams = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
    return unigrams + bigrams


def _hash_bucket(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % EMBED_DIM


def embed(text: str) -> np.ndarray:
    """文本 -> 单位向量。归一化后点积 == 余弦相似度。"""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for tok in _tokenize(text):
        vec[_hash_bucket(tok)] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---- 1.4 向量存储（复用 04-RAG 的 NumpyVectorStore）---------------------------
class NumpyVectorStore:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self.matrix: np.ndarray = np.zeros((0, EMBED_DIM), dtype=np.float32)

    def add(self, texts: list[str], metadatas: list[dict]) -> None:
        vectors = np.stack([embed(t) for t in texts]) if texts else np.zeros((0, EMBED_DIM))
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.matrix = vectors if self.matrix.shape[0] == 0 else np.vstack([self.matrix, vectors])

    def query(self, query_vec: np.ndarray, top_k: int = TOP_K) -> list[tuple[str, float, dict]]:
        if not self.texts:
            return []
        scores = self.matrix @ query_vec
        k = min(top_k, len(self.texts))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.texts[i], float(scores[i]), self.metadatas[i]) for i in top_idx]


def build_knowledge_base(docs: list[str]) -> NumpyVectorStore:
    """把内置文档分块 + 嵌入 + 入库，返回构造好的向量库。"""
    store = NumpyVectorStore()
    chunk_texts: list[str] = []
    chunk_metas: list[dict] = []
    for doc_id, doc in enumerate(docs):
        for chunk in _split_by_sentences(doc, n=2, overlap=1):
            chunk_texts.append(chunk)
            chunk_metas.append({"doc_id": doc_id, "source": f"知识库#{doc_id}"})
    store.add(chunk_texts, chunk_metas)
    return store


# ============================================================================
# 模块二：Agent 的"工具"（对应 05-Agent/02-react_agent 的 ACTIONS 注册表）
# ============================================================================

@dataclass
class ToolResult:
    """工具执行结果。is_relevant 标记本次检索是否真的相关（防幻觉用）。"""
    text: str
    is_relevant: bool = True


def make_tools(store: NumpyVectorStore):
    """构造工具集。每个工具：名字 + 描述（给 LLM 看）+ 执行函数。

    真实 Agent 框架（LangChain/OpenAI function calling）里，这些描述会被塞进
    Prompt 让模型知道"我有哪些工具、各能干什么"。这里我们用脚本化 LLM，
    描述仅作展示，但结构上和真实环境一致。
    """

    def retrieve(query: str) -> ToolResult:
        """工具1：知识库检索（RAG）。"""
        hits = store.query(embed(query), top_k=TOP_K)
        if not hits:
            return ToolResult("知识库为空，未检索到任何内容。", is_relevant=False)
        best_score = hits[0][1]
        if best_score < RELEVANCE_THRESHOLD:
            return ToolResult(
                "检索到的内容与问题相关度很低，可能知识库中没有相关信息。",
                is_relevant=False,
            )
        # 把 top_k 结果拼成"观察"，附相似度便于追溯
        lines = []
        for i, (chunk, score, meta) in enumerate(hits):
            lines.append(f"[{i + 1}](score={score:.3f}, {meta['source']}) {chunk.strip()}")
        return ToolResult("\n".join(lines), is_relevant=True)

    def calculator(expression: str) -> ToolResult:
        """工具2：四则运算计算器（演示"非检索类"工具如何接入）。

        安全实现：只允许数字和 + - * / ( ) 空格，避免 eval 风险。
        """
        if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expression):
            return ToolResult(f"表达式不安全/不合法：{expression}", is_relevant=False)
        try:
            value = eval(expression, {"__builtins__": {}}, {})
            return ToolResult(f"{expression} = {value}", is_relevant=True)
        except ZeroDivisionError:
            return ToolResult("错误：除以零。", is_relevant=False)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"计算错误：{e}", is_relevant=False)

    # 注册表：(执行函数, 是否结束动作)
    tools: dict[str, tuple[Callable[[str], ToolResult], bool]] = {
        "检索": (retrieve, False),
        "计算": (calculator, False),
        "完成": (lambda ans: ToolResult(ans), True),
    }
    return tools


# ============================================================================
# 模块三：Prompt 工程（对应 03-Prompt工程）
# ============================================================================
# 结构化 Prompt = 系统角色 + 工具说明 + 对话记忆 + 当前问题
SYSTEM_PROMPT = """\
你是一个智能知识助手。你可以使用以下工具回答用户问题：

1. 检索[查询词]   : 从知识库中检索相关片段（适合事实类问题）
2. 计算[表达式]   : 执行四则运算（适合数学问题）
3. 完成[最终答案] : 给出最终答案并结束本轮

回答流程（ReAct）：每一步先 Thought（推理），再 Action（选一个工具），
看到 Observation 后继续，直到信息足够用 完成[...] 收尾。
如果知识库检索结果不相关，应诚实说明不知道，不要编造。
"""

# 给用户的"可见格式"：把 Agent 的结构化输出整理成自然语言


# ============================================================================
# 模块四：对话记忆（多轮上下文）—— 整合性体现
# ============================================================================
@dataclass
class Memory:
    """对话记忆：保留最近若干轮的 (用户, 助手) 对。

    真实场景会把 Memory 也塞进 Prompt（system + history + user）。
    这里我们用它做两件事：
      1) 追问消解：把"它/这个/那个"等指代用上一轮主题替换（toy 版）；
      2) 让用户看到"助手记得上下文"。
    """
    max_turns: int = 3
    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))
        # 只保留最近 max_turns 轮（滑动窗口，防止 Prompt 无限膨胀）
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def resolve_coreference(self, query: str) -> str:
        """toy 指代消解：把"它/这个"替换成上一轮用户问题的核心词。"""
        if not self.turns:
            return query
        last_user = self.turns[-1][0]
        # 取上一轮问题里第一个长度>=2的词当作"主题"
        core = None
        for kw in ["RAG", "ReAct", "Prompt", "Transformer", "Agent", "检索", "向量", "幻觉"]:
            if kw in last_user:
                core = kw
                break
        if core and ("它" in query or "这个" in query or "那个" in query):
            return query.replace("它", core).replace("这个", core).replace("那个", core)
        return query

    def render(self) -> str:
        if not self.turns:
            return "(无历史)"
        return "\n".join(f"用户: {u}\n助手: {a}" for u, a in self.turns)


# ============================================================================
# 模块五：脚本化 LLM（模拟大模型按 ReAct 输出 Thought/Action）
#         真实环境把这里换成一次模型调用即可（见文末真实版说明）
# ============================================================================

def _classify(query: str) -> str:
    """toy 意图识别：判断这个问题该走检索、计算，还是直接回答。"""
    # 数学问题：包含"算/等于/是多少"+ 数字 + 运算符
    if re.search(r"\d", query) and re.search(r"[\+\-\*/×÷]", query) and ("算" in query or "等于" in query or "多少" in query or "*" in query or "/" in query):
        return "calc"
    # 默认走检索（事实类）
    return "retrieve"


def _extract_expression(query: str) -> str:
    """从自然语言里抠出算式，如 '帮我算 3.5 * 12' -> '3.5 * 12'。"""
    m = re.search(r"([\d\s\+\-\*/\(\)\.]+)", query)
    return m.group(1).strip() if m else query


def mock_llm_step(query: str, history_text: str, step: int, intent: str) -> tuple[str, str]:
    """脚本化 LLM：根据问题类型 + 步数，返回 (thought, action_text)。

    真实环境：把 SYSTEM_PROMPT + history + query + 历史 Observation 喂给模型，
    让模型输出 "Thought N: ... Action N: ...", 再用正则解析。
    """
    # ---- 计算类问题：一步算 + 一步完成 ----
    if intent == "calc":
        expr = _extract_expression(query)
        if step == 1:
            return (
                f"这是一个数学计算问题。我需要计算 {expr}，调用计算器。",
                f"计算[{expr}]",
            )
        if step == 2:
            return (
                "已得到计算结果，给出最终答案。",
                f"完成[{expr} 的计算结果见上一步观察。]",
            )

    # ---- 检索类问题：先检索，若相关则完成；不相关则诚实回答 ----
    if intent == "retrieve":
        if step == 1:
            return (
                f"这是一个事实类问题，我需要从知识库检索「{query}」相关信息。",
                f"检索[{query}]",
            )
        if step == 2:
            return (
                "已检索到候选内容，下一步根据 Observation 的相关度给出最终答案。",
                "完成[根据检索到的上下文作答。]",
            )

    # 兜底
    return ("信息不足，结束本轮。", "完成[我暂时无法回答该问题。]")


# ============================================================================
# 模块六：ReAct 主循环 + 最终回答生成（整合 RAG 观察 + Prompt 规范化）
# ============================================================================

def parse_action(action_text: str) -> tuple[str, str]:
    """把 '检索[RAG 是什么]' 解析成 ('检索', 'RAG 是什么')。"""
    head, _, body = action_text.partition("[")
    return head.strip(), body.rstrip("]").strip()


def react_loop(
    query: str,
    tools: dict,
    memory: Memory,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """ReAct 推理循环。返回 (最终答案, 轨迹)。

    轨迹每一步：{thought, action, observation}
    """
    history_text = memory.render()
    intent = _classify(query)
    trace: list[dict] = []

    if verbose:
        print("\n" + "=" * 72)
        print(f"[ReAct 循环开始] 问题: {query}")
        print(f"[意图识别] -> {intent}")
        print(f"[对话记忆]\n{history_text}")
        print("=" * 72)

    for step in range(1, MAX_STEPS + 1):
        thought, action_text = mock_llm_step(query, history_text, step, intent)
        action_name, arg = parse_action(action_text)

        if action_name not in tools:
            trace.append({"thought": thought, "action": action_text, "observation": f"未知工具: {action_name}"})
            break

        func, is_finish = tools[action_name]

        if verbose:
            print(f"\nThought {step}: {thought}")
            print(f"Action  {step}: {action_name}[{arg}]")

        if is_finish:
            # 完成动作：最终答案由"综合观察"得到（见下方 compose_final_answer）
            final = compose_final_answer(query, trace, intent)
            if verbose:
                print(f"Observation {step}: (Finish) -> {final}")
            trace.append({"thought": thought, "action": action_text, "observation": final})
            return final, trace

        # 执行工具，得到 Observation
        result: ToolResult = func(arg)
        observation = result.text
        if verbose:
            print(f"Observation {step}: {observation}")
        trace.append(
            {
                "thought": thought,
                "action": action_text,
                "observation": observation,
                "tool": action_name,
                "is_relevant": result.is_relevant,
            }
        )

    # 到达最大步数仍未完成
    return "我暂时无法回答该问题（推理步数耗尽）。", trace


def compose_final_answer(query: str, trace: list[dict], intent: str) -> str:
    """把 ReAct 轨迹里的"观察"综合成给用户的自然语言答案。

    这是"整合性"的关键：Agent 的推理结果不是直接 dump 观察文本，
    而是按 Prompt 工程的规范，给出"有依据、防幻觉、带溯源"的回答。
    """
    if intent == "calc":
        # 找到计算工具的观察
        for t in trace:
            if t.get("tool") == "计算":
                if t.get("is_relevant"):
                    return f"计算结果：{t['observation'].split(' = ')[-1] if ' = ' in t['observation'] else t['observation']}"
                return t["observation"]
        return "无法完成该计算。"

    # 检索类：从轨迹里挑出相关的检索观察，拼成答案
    relevant_chunks: list[str] = []
    sources: set[str] = set()
    for t in trace:
        if t.get("tool") == "检索" and t.get("is_relevant"):
            # 解析每个 [n](score=..., 来源) 片段
            for line in t["observation"].splitlines():
                m = re.match(r"\[\d+\]\(score=([\d.]+), ([^)]+)\) (.+)", line)
                if m:
                    score = float(m.group(1))
                    src = m.group(2)
                    text = m.group(3)
                    if score >= RELEVANCE_THRESHOLD:
                        relevant_chunks.append(text)
                        sources.add(src)

    if not relevant_chunks:
        return "抱歉，知识库里似乎没有与该问题直接相关的内容，我暂时无法回答。"

    # 综合：取相关片段，并标注来源（防幻觉 / 可溯源）
    body = "".join(relevant_chunks[:TOP_K])
    src_str = "、".join(sorted(sources))
    return f"{body}\n（来源：{src_str}）"


# ============================================================================
# 模块七：助手封装（多轮对话入口）
# ============================================================================

class KnowledgeAssistant:
    """智能知识助手：把 RAG + ReAct + 记忆封装成一个可对话的对象。"""

    def __init__(self, docs: list[str] | None = None) -> None:
        self.store = build_knowledge_base(docs or KNOWLEDGE_DOCS)
        self.tools = make_tools(self.store)
        self.memory = Memory(max_turns=3)
        print(f"[助手初始化] 知识库已建立：{len(self.store.texts)} 个 chunk，"
              f"向量矩阵 {self.store.matrix.shape}")

    def chat(self, query: str, verbose: bool = True) -> str:
        """单轮对话：含指代消解 → ReAct → 写入记忆。"""
        resolved = self.memory.resolve_coreference(query)
        if verbose and resolved != query:
            print(f"\n[指代消解] 原问: {query}  ->  消解后: {resolved}")

        answer, _trace = react_loop(resolved, self.tools, self.memory, verbose=verbose)
        self.memory.add(query, answer)
        return answer


# ============================================================================
# main：跑一个多轮对话 demo
# ============================================================================

def main() -> None:
    print("=" * 72)
    print(" 智能知识助手 Intelligent Knowledge Assistant")
    print(" 整合 RAG 检索 + ReAct Agent + 多轮对话记忆（纯 numpy 模拟版）")
    print("=" * 72)

    assistant = KnowledgeAssistant()

    # 一个多轮对话脚本，覆盖三种能力：
    #   1) 事实类 -> RAG 检索
    #   2) 数学类 -> 计算器工具
    #   3) 追问    -> 记忆 + 指代消解
    #   4) 超纲问题 -> 诚实拒答（防幻觉）
    dialogue = [
        "RAG 是什么？",                 # 事实类 -> 检索
        "它和这个 ReAct 有什么区别？",   # 追问：指代消解"它/这个" -> RAG / ReAct
        "帮我算一下 3.5 * 12 等于多少？", # 数学类 -> 计算器
        "今天晚饭吃什么？",             # 超纲 -> 防幻觉拒答
    ]

    for i, q in enumerate(dialogue, 1):
        print("\n" + "#" * 72)
        print(f"# 第 {i} 轮对话")
        print("#" * 72)
        print(f"\n用户提问 >>> {q}")
        answer = assistant.chat(q, verbose=True)
        print("\n" + "-" * 72)
        print(f"最终回答 >>> {answer}")
        print("-" * 72)

    # 收尾
    print("\n" + "=" * 72)
    print(" 全流程跑通 ✓")
    print(" 整合点回顾：")
    print("   - 04-RAG      : embed + NumpyVectorStore + retrieve (知识库检索)")
    print("   - 05-Agent    : ReAct 循环 Thought→Action→Observation→Finish")
    print("   - 03-Prompt   : SYSTEM_PROMPT 结构化 + 防幻觉 + 溯源标注")
    print("   - 多轮记忆    : Memory 滑动窗口 + 指代消解")
    print(" 接真实模型只需替换三处（见 README.md「如何接真实模型」）：")
    print("   mock_llm_step -> OpenAI/Ollama 调用")
    print("   embed         -> sentence-transformers")
    print("   NumpyVectorStore -> chromadb/faiss")
    print("=" * 72)


if __name__ == "__main__":
    main()


# ============================================================================
#                              真实版说明
# ----------------------------------------------------------------------------
# 本文件三大"可替换点"，替换后即变成真实可用的 Agent：
#
# 1) LLM (mock_llm_step)
#    from openai import OpenAI
#    client = OpenAI()
#    resp = client.chat.completions.create(
#        model="gpt-4o-mini",
#        messages=[
#            {"role": "system", "content": SYSTEM_PROMPT + tools_description},
#            {"role": "user", "content": memory.render() + "\n问题: " + query},
#        ],
#    )
#    # 解析 resp 里的 "Thought N: ... Action N: ..." 或用 function calling
#
# 2) Embedding (embed)
#    from sentence_transformers import SentenceTransformer
#    _model = SentenceTransformer("BAAI/bge-small-zh")
#    def embed(text): return _model.encode(text, normalize_embeddings=True)
#
# 3) 向量库 (NumpyVectorStore)
#    import chromadb
#    client = chromadb.PersistentClient(path="./chroma")
#    col = client.get_or_create_collection("kb", metadata={"hnsw:space": "cosine"})
#    col.add(ids=..., embeddings=..., documents=..., metadatas=...)
#    col.query(query_embeddings=..., n_results=TOP_K)
#
# 其余骨架（ReAct 循环、记忆、Prompt 模板、工具注册表）完全不用改。
# ============================================================================
