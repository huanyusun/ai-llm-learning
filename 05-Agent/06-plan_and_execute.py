"""
06 · Plan-and-Execute Agent 模式演示
==============================================
面试考点：Plan-and-Execute vs ReAct 的区别、适用场景
运行：uv run python "05-Agent/06-plan_and_execute.py"
"""

print("=" * 60)
print("Plan-and-Execute Agent 模式演示")
print("=" * 60)

# ── 1. ReAct vs Plan-and-Execute ──────────────────────────
print("""
两种 Agent 模式对比：

┌─────────────────────────────────────────────────────────┐
│  ReAct 模式（边想边做）                                    │
│                                                         │
│  Thought → Action → Observation → Thought → Action → ...│
│                                                         │
│  特点：每步都重新思考，灵活但可能迷失方向                    │
│  适合：简单任务、步骤少、需要即时反馈                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Plan-and-Execute 模式（先规划再执行）                      │
│                                                         │
│  Plan: [Step1, Step2, Step3, ...]                       │
│  Execute: Step1 → Step2 → Step3 → ...                   │
│  Replan: 根据执行结果调整计划                               │
│                                                         │
│  特点：全局规划，执行更有条理                               │
│  适合：复杂任务、多步骤、需要全局视角                        │
└─────────────────────────────────────────────────────────┘
""")

# ── 2. 模拟 Plan-and-Execute ──────────────────────────────
print("=" * 60)
print("模拟 Plan-and-Execute：研究报告生成")
print("=" * 60)

task = "帮我写一份关于 RAG 技术在企业中应用的研究报告"
print(f"\n用户任务: \"{task}\"\n")

# Phase 1: Planning
print("--- Phase 1: Planning（规划阶段）---")
plan = [
    "搜索 RAG 技术的最新发展和企业应用案例",
    "整理 RAG 的核心技术栈（检索、生成、评估）",
    "收集 3-5 个企业级 RAG 应用的成功案例",
    "分析 RAG 在企业落地的常见挑战和解决方案",
    "撰写报告大纲并填充内容",
    "审校报告，确保数据准确、逻辑通顺",
]

print("LLM 生成的执行计划:")
for i, step in enumerate(plan):
    print(f"  Step {i+1}: {step}")

# Phase 2: Execute
print(f"\n--- Phase 2: Execute（执行阶段）---")

execution_results = [
    {"step": 1, "status": "✅", "result": "找到 15 篇相关论文和 8 个企业案例"},
    {"step": 2, "status": "✅", "result": "整理出 5 大技术模块：数据处理、嵌入、检索、重排、生成"},
    {"step": 3, "status": "✅", "result": "收集到：金融客服、法律检索、医疗问答、电商推荐、内部知识库"},
    {"step": 4, "status": "⚠️", "result": "发现数据安全和模型幻觉是最大挑战，需要补充解决方案"},
]

for r in execution_results:
    print(f"  Step {r['step']} {r['status']}: {r['result']}")

# Phase 3: Replan
print(f"\n--- Phase 3: Replan（重规划）---")
print("  Step 4 发现新问题 → 调整计划:")
print("  Step 4.1: 补充数据安全方案（私有化部署、数据脱敏）")
print("  Step 4.2: 补充幻觉缓解方案（Rerank、事实核查）")
print("  继续执行 Step 5, 6...")

# ── 3. 代码框架 ──────────────────────────────────────────
print(f"\n" + "=" * 60)
print("Plan-and-Execute 代码框架（伪代码）")
print("=" * 60)
print("""
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm    # 负责生成和调整计划
        self.executor = executor_llm  # 负责执行每一步
        self.tools = tools

    def run(self, task):
        # Phase 1: 生成计划
        plan = self.planner.generate_plan(task)

        results = []
        for step in plan:
            # Phase 2: 执行每一步
            result = self.executor.execute(step, self.tools)
            results.append(result)

            # Phase 3: 检查是否需要重规划
            if result.needs_replan:
                plan = self.planner.replan(task, results, plan)

        return self.synthesize(results)
""")

# ── 4. Agent 记忆机制 ──────────────────────────────────────
print("=" * 60)
print("Agent 记忆机制")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────┐
│  短期记忆（Working Memory）                               │
│                                                         │
│  - 实现：对话上下文窗口（chat history）                     │
│  - 容量：受 token 限制（如 128K tokens）                   │
│  - 特点：会话结束即丢失                                    │
│  - 优化：摘要压缩、滑动窗口                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  长期记忆（Long-term Memory）                             │
│                                                         │
│  - 实现：外部存储（向量数据库、关系数据库）                   │
│  - 容量：理论无限                                         │
│  - 特点：跨会话持久化                                      │
│  - 检索：语义检索（向量相似度）+ 时间衰减                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  反思记忆（Reflective Memory）                            │
│                                                         │
│  - 实现：定期对经验进行总结和反思                            │
│  - 代表：Generative Agents (Stanford, 2023)              │
│  - 流程：观察 → 存储 → 检索 → 反思 → 规划                  │
│  - 特点：能形成高层次的"认知"和"价值观"                     │
└─────────────────────────────────────────────────────────┘

记忆与 RAG 的关系：
  - Agent 的长期记忆本质上就是一个 RAG 系统
  - 存储：把对话/经验嵌入后存入向量数据库
  - 检索：新对话时检索相关历史记忆
  - 区别：RAG 检索外部知识，记忆检索自身经验

面试要点：
1. 短期记忆受 token 限制，长期记忆靠外部存储
2. 记忆的检索策略：相关性 + 时效性 + 重要性
3. Generative Agents 的反思机制是 Agent 记忆的里程碑
4. MemGPT 用操作系统的虚拟内存思想管理 LLM 上下文
""")
