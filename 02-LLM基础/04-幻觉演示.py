# 依赖（可选）：本文件优先用本地 ollama 实测；若无后端，则退化为「剧本演示」。
#   实测需要: ollama run qwen2.5  +  uv add openai
# 不要自动安装——避免和你的环境冲突。
"""
LLM 基础 第4课：幻觉（Hallucination）演示
==========================================
参考：知识库《02-LLM基础/系统知识.md》§6 幻觉：成因与缓解
      InstructGPT 论文（[S2] §1）：幻觉率 21% vs 41%
      GPT-4 段（[S1] §2.3）：red teaming / 安全 reward

什么是幻觉？模型生成【看似合理、但与事实或输入不符】的内容。
InstructGPT 论文用数据刻画（[S2] §1，必背）：
    "InstructGPT models make up information not present in the input
     about half as often as GPT-3 (a 21% vs. 41% hallucination rate)."
即：对齐把幻觉率从 GPT-3 的 41% 降到 21%（约减半），但【不为零】。

【本课重点】
  ① 用「超出知识边界」的问题触发幻觉（虚构人物/事件/文献）
  ② 用「知识截止」类问题触发幻觉（问最新事件）
  ③ 对比「裸问」vs「带缓解策略」的回答差异
  ④ 把 §6.3 的缓解策略连成可操作的工程 checklist

运行：
  uv run --directory /Users/sunhuanyu/ai-llm-learning python "02-LLM基础/04-幻觉演示.py"
  （无 ollama/openai 时自动用「剧本演示」模式，照样能学。）
"""
import os
import textwrap


# ============================================================
# 触发幻觉的「陷阱问题」集（都是模型容易编的内容）
# ============================================================
TRAP_QUESTIONS = [
    {
        "tag": "虚构人物",
        "question": "请详细介绍明代学者「张怀瑾」在《格物通考》中的主要贡献。",
        "why": "「张怀瑾」和《格物通考》都是虚构的。模型为了「答得像」，",
        "likely": "很可能会编造一个听起来很像真的生平和贡献。",
    },
    {
        "tag": "不存在的论文",
        "question": "请总结 Smith & Wang 2023 年发表在 NeurIPS 上的论文"
                    "《Adaptive Hyper-Spectral Tokenization》的核心方法。",
        "why": "这篇论文不存在。模型见过太多「NeurIPS + 两个姓 + 技术名词」组合，",
        "likely": "很容易拼凑出一篇假论文的摘要、方法、实验数字。",
    },
    {
        "tag": "知识截止",
        "question": "昨天（具体日期你自己判断）A股收盘后，央行发布了什么新政策？",
        "why": "训练数据有截止日期，模型根本「不知道昨天」。但为了显得有用，",
        "likely": "可能一本正经地编一条政策。",
    },
    {
        "tag": "过度自信的细节",
        "question": "圆周率小数点后第 1234 位数字是几？请直接给答案并解释。",
        "why": "模型没真正「背」到那么远；它在拟合「圆周率」的概率分布，",
        "likely": "会自信地给一个错答案，而不是说「我不确定」。",
    },
]


# ============================================================
# 缓解策略（对应系统知识库 §6.3）
# ============================================================
MITIGATION_PROMPTS = {
    "虚构人物": (
        "如果下面提到的人或书并不存在，请直接说「我不确定，"
        "这可能是一个虚构的人物/书名」，不要编造内容。\n\n问题："
        "请详细介绍明代学者「张怀瑾」在《格物通考》中的主要贡献。"
    ),
    "不存在的论文": (
        "请先判断这篇论文是否真实存在。如果不确定，请明确说「我不确定"
        "这篇论文是否存在」，不要编造作者、方法或实验结果。\n\n问题："
        "请总结 Smith & Wang 2023 年发表在 NeurIPS 上的论文"
        "《Adaptive Hyper-Spectral Tokenization》的核心方法。"
    ),
}


# ============================================================
# 调用模型（复用第3课的逻辑，独立成简单封装，避免循环依赖）
# ============================================================
def ask_model(prompt: str, temperature: float = 0.3):
    """尝试调用本地 ollama 或云端 openai；失败返回 None。"""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    # 先试 ollama（免费本地）
    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        resp = client.chat.completions.create(
            model="qwen2.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    # 再试 openai
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    return None


# ============================================================
# 剧本演示（无后端时的预期行为说明）
# ============================================================
SCRIPTED = {
    "虚构人物": (
        "【典型幻觉回答（剧本）】\n"
        "  张怀瑾（约 1450—1522），明代中期著名学者，所著《格物通考》共十二卷，\n"
        "  系统总结了当时的格物之学，提出了「物必有理，理必有验」的方法论，\n"
        "  对后世王廷相、方以智等影响深远……（以上全部为编造，听起来却很可信）\n"
        "【带缓解的回答（剧本）】\n"
        "  我不确定「张怀瑾」和《格物通考》是否真实存在，这可能是虚构的人物/书名，\n"
        "  建议核实后再追问。"
    ),
    "不存在的论文": (
        "【典型幻觉回答（剧本）】\n"
        "  Smith & Wang (NeurIPS 2023) 提出 Adaptive Hyper-Spectral Tokenization（AHT），\n"
        "  通过可学习的光谱带分组把高光谱图像切成自适应 token，在四个数据集上\n"
        "  提升 3.2% 的 mIoU……（论文不存在，方法/数字都是编的）\n"
        "【带缓解的回答（剧本）】\n"
        "  我不确定这篇论文是否真实存在，无法确认作者、会议和方法，请核实标题。"
    ),
}


# ============================================================
# 主流程
# ============================================================
def wrap(text, indent="    ", width=72):
    """把长文本按固定宽度缩进换行，方便阅读。"""
    return "\n".join(indent + line for line in textwrap.wrap(text, width) if line.strip())


def main():
    print("=" * 64)
    print("第4课：幻觉演示——问模型「超出它知识边界」的问题")
    print("=" * 64)
    print()
    print("【背景】InstructGPT 论文（[S2] §1）的标志性数据：")
    print("    InstructGPT 幻觉率 21%  vs  GPT-3 的 41%  —— 对齐让它约减半，但【不为零】。")
    print("幻觉根因（系统知识库 §6.2）：语言建模优化的是「下一个 token 的似然」，")
    print("    不是「真伪」。模型在拟合概率，而非事实。")
    print()

    have_model = ask_model("请只回复「OK」两个字。") is not None
    mode = "实测（连到了真实模型）" if have_model else "剧本演示（未检测到可用模型）"
    print(f"【运行模式】{mode}\n")

    for trap in TRAP_QUESTIONS:
        print("-" * 64)
        print(f"陷阱类型：{trap['tag']}")
        print(f"问题：{trap['question']}")
        print(f"为什么危险：{trap['why']}{trap['likely']}")

        tag = trap["tag"]
        if have_model:
            print("\n  ▶ 裸问（不做任何缓解）的回答：")
            ans = ask_model(trap["question"], temperature=0.3)
            print(wrap(ans or "(无返回)"))
            if tag in MITIGATION_PROMPTS:
                print("\n  ▶ 加缓解策略（系统知识库 §6.3）后的回答：")
                ans2 = ask_model(MITIGATION_PROMPTS[tag], temperature=0.3)
                print(wrap(ans2 or "(无返回)"))
        else:
            if tag in SCRIPTED:
                print()
                for line in SCRIPTED[tag].splitlines():
                    print("    " + line if line and not line.startswith("【") else line)
        print()

    # ============================================================
    # 缓解策略 checklist（连回面试）
    # ============================================================
    print("=" * 64)
    print("缓解幻觉的工程 checklist（系统知识库 §6.3）")
    print("=" * 64)
    checklist = [
        ("RAG（检索增强）",         "引入外部权威知识作 grounding，事实性场景的首选。⚠ 检索错则「带毒检索」。"),
        ("低温度 / 确定性解码",      "temperature 调低（0~0.3），减少「放飞」式随机幻觉。⚠ 不解决事实性幻觉。"),
        ("结构化 prompt",           "加「不确定就说不知道」「先核实再答」「给出处」等约束（本课演示）。"),
        ("Self-consistency",        "多次采样取一致答案（多数投票），适合有标准答案的任务。"),
        ("RLHF / DPO 对齐",         "用人类偏好直接降低编造输出（InstructGPT 21% vs 41% 的来源）。"),
        ("Red teaming（红队）",      "主动构造边界/恶意 prompt 找漏洞，迭代修复（GPT-4 采用，[S1] §2.3）。"),
        ("让模型「承认不知道」",     "训练模型识别自身知识边界（self-awareness），比硬编「我不知道」更鲁棒。"),
    ]
    for name, desc in checklist:
        print(f"  • {name}：{desc}")

    print()
    print("=" * 64)
    print("✅ 第4课要点（连回面试）")
    print("=" * 64)
    print("  • 幻觉 = 生成看似合理但与事实/输入不符的内容；根因是「拟合似然而非真伪」。")
    print("  • 标志数据：InstructGPT 21% vs GPT-3 41%，对齐让幻觉【减半但非零】（[S2] §1）。")
    print("  • 五大成因：训练目标、数据噪声、对齐讨好性、知识截止、长上下文遗忘（§6.2）。")
    print("  • 工程首选：RAG（grounding）+ 低温度 + 结构化 prompt + 让模型敢说「不知道」。")
    print("  • ⚠ 误区：RAG 不能根治幻觉（受检索质量制约）；低温度只压随机幻觉不压事实幻觉。")
    print("=" * 64)


if __name__ == "__main__":
    main()
