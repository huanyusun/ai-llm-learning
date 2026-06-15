# 依赖: uv add tiktoken
# （本文件用 tiktoken 演示 GPT 系的 BPE 分词。请先按上面命令安装依赖再运行。
#   不要自动安装——避免和你的环境冲突。）
"""
LLM 基础 第1课：Tokenizer 实验（BPE 分词）
==========================================
参考：知识库《02-LLM基础/系统知识.md》§3 Tokenizer
      tiktoken 官方文档 https://github.com/openai/tiktoken
      HuggingFace NLP Course（[S1] §3.2 推荐）

核心问题：模型不直接读「字符」也不直接读「词」，而是读 token。
          token 是介于字符和词之间的「子词（subword）」单元。
          把文本切成 token 的工具就叫 tokenizer。

GPT 系（GPT-3.5/4、LLaMA）用的都是 **Byte-level BPE**（字节级 BPE）：
  ① 以「字节」为基本符号（能覆盖任意字符，包括中文、emoji、罕见字符）；
  ② 迭代合并语料中【最频繁共现的相邻字节对】，直到词表达到预设大小。
  合并选择标准是「频率」——这是 BPE 与 WordPiece（BERT，选似然增益）的关键区别。

【本课重点】
  • 看中文/英文分别被切成几个 token（中文更费 token）
  • 统计 token 数（这是 API 计费和上下文窗口的基本单位）
  • 看同一个词在不同编码（cl100k_base / p50k_base）下 token 数不同——词表不通用

运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "02-LLM基础/01-tokenizer实验.py"
"""
import os

try:
    import tiktoken
except ImportError:
    # 友好提示：缺依赖时给出明确指令，而不是一长串 traceback
    print("=" * 60)
    print("❌ 缺少依赖 tiktoken")
    print("=" * 60)
    print("请先在项目根目录运行：")
    print("    uv add tiktoken")
    print("然后重新运行本文件。")
    print()
    print("为什么用 tiktoken？它是 OpenAI 开源的 BPE 分词器，")
    print("和 GPT-3.5/4 用的编码（cl100k_base）一致，适合学习 LLM 分词。")
    raise SystemExit(1)


# ============================================================
# 【1】最基本：看一句话被切成几个 token、每个 token 是什么
# ============================================================
print("=" * 64)
print("【1】BPE 切词：看一句话被切成几个 token")
print("=" * 64)

# cl100k_base 是 GPT-4 / GPT-3.5 用的编码（词表）
enc = tiktoken.get_encoding("cl100k_base")

text_en = "Hello, world! I love learning large language models."
tokens_en = enc.encode(text_en)

print(f"\n英文句子：{text_en}")
print(f"  字符数  : {len(text_en)}")
print(f"  token 数: {len(tokens_en)}")
print(f"  每个 token（id → 还原文本）:")
for tid in tokens_en:
    # decode 把单个 token id 还原成字节串，能直观看到「子词」长什么样
    piece = enc.decode([tid])
    # 把不可见字符（空格显示成 ␣，换行显示成 ↵）方便观察
    display = piece.replace(" ", "␣").replace("\n", "↵")
    print(f"    id={tid:>6}  →  {display!r}")
print(f"  → 观察：常见的整词（Hello/love/models）往往是一个 token；")
print(f"     少见的词会被拆成多个子词 token（这就是 BPE 的「子词」精髓）。")


# ============================================================
# 【2】中文 vs 英文：同样语义，中文更费 token
# ============================================================
print("\n" + "=" * 64)
print("【2】中英文对比：同样意思，token 数差很多")
print("=" * 64)

pairs = [
    ("Hello world", "你好，世界"),
    ("I love you", "我爱你"),
    ("Large language models are powerful", "大语言模型很强大"),
    ("The quick brown fox jumps over the lazy dog",
     "敏捷的棕色狐狸跳过了懒狗"),
]

print(f"\n{'英文':<42} {'tok':>4}  | {'中文':<22} {'tok':>4}  倍数")
print("-" * 70)
for en, zh in pairs:
    t_en = len(enc.encode(en))
    t_zh = len(enc.encode(zh))
    ratio = t_zh / t_en
    print(f"{en:<42} {t_en:>4}  | {zh:<22} {t_zh:>4}  {ratio:.2f}x")

print("\n  → 结论：GPT-4 的词表（cl100k_base）对英文友好，对中文不友好。")
print("     中文一个字常被切成 1~2 个 token，导致：")
print("     ① 同样内容的中文，token 数更多，API 计费更贵；")
print("     ② 中文更早撑爆上下文窗口。")
print("     这就是为什么国产模型（Qwen/GLM/DeepSeek）会专门优化中文词表。")


# ============================================================
# 【3】不同编码词表不同 → token 数不同（tokenizer 不通用）
# ============================================================
print("\n" + "=" * 64)
print("【3】不同模型用不同词表：token id 不通用！")
print("=" * 64)

sample = "Tokenizer 把文本切成 token，是 LLM 的入口。"
# cl100k_base: GPT-4 / GPT-3.5
# p50k_base  : text-davinci-003 / Codex（老 GPT-3.5）
# o200k_base  : GPT-4o（更新的词表，对多语更友好）
encodings = ["cl100k_base", "o200k_base"]
print(f"\n文本：{sample}")
for name in encodings:
    e = tiktoken.get_encoding(name)
    n = len(e.encode(sample))
    print(f"  {name:<14}: {n} 个 token")
print("  → 同一段文字在不同词表下 token 数不同。")
print("     把 GPT 的 token id 直接喂给 LLaMA 是错的——必须用各自模型的 tokenizer。")


# ============================================================
# 【4】估算「上下文窗口 / 费用」：token 是计量单位
# ============================================================
print("\n" + "=" * 64)
print("【4】实战估算：token 是上下文窗口和计费的单位")
print("=" * 64)

article = (
    "大语言模型（LLM）是参数量通常达数百亿的 Transformer 语言模型。"
    "它通过预训练习得语言理解和世界知识，再经 SFT 和 RLHF 对齐人类偏好。"
    "推理时采用自回归生成：每一步根据前文预测下一个 token，"
    "用 temperature、top-k、top-p 等采样策略从概率分布中选词。"
)
n_tokens = len(enc.encode(article))
print(f"\n一段 {len(article)} 字的中文：")
print(f"  → 约占 {n_tokens} 个 token")
print(f"  → GPT-4o 上下文窗口 128000 token，这段文字约占 {n_tokens/128000*100:.3f}%")
print(f"  → 若按 $5 / 1M input token 计费，这段约花 ${n_tokens/1e6*5:.5f}")
print("  → 工程意义：写 RAG/Agent 时，token 数直接决定能塞多少上下文、花多少钱。")


# ============================================================
# 【5】手写一个迷你 BPE：感受「合并最频繁的相邻对」
# ============================================================
print("\n" + "=" * 64)
print("【5】手写迷你 BPE：感受「合并最频繁的相邻字节对」")
print("=" * 64)

def mini_bpe(corpus: str, num_merges: int):
    """极简 BPE 演示：在「字符」层面迭代合并最频繁的相邻对。

    注意：这只是教学版，省略了字节级、特殊 token、正则预切分等细节。
    tiktoken 的真实实现要复杂得多，但核心思想就是这个「贪心合并」。
    """
    # 把语料拆成单个字符（含空格标记 ␣ 便于观察词边界）
    word = list(corpus.replace(" ", "␣"))
    merges = []  # 记录每次合并了哪一对

    for step in range(num_merges):
        # 统计所有相邻对的频次
        from collections import Counter
        pairs = Counter(zip(word[:-1], word[1:]))
        if not pairs:
            break
        # 选频次最高的一对合并
        best = pairs.most_common(1)[0][0]
        freq = pairs[best]
        if freq < 2:
            break  # 只剩出现一次的对，合并没意义

        # 执行合并：把相邻的 (a, b) 拼成 "ab"
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == best:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
        merged = (best[0] + best[1]).replace("␣", " ")
        merges.append((best, freq, merged))

    return word, merges

demo = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
final, merges = mini_bpe(demo, num_merges=6)
print(f"\n输入语料：{demo}")
print(f"\n迭代合并过程（每次合并语料中最频繁的相邻对）：")
for i, (pair, freq, merged) in enumerate(merges, 1):
    print(f"  第{i}次合并: {pair}  (出现 {freq} 次)  → 生成新子词 [{merged}]")
print(f"\n最终子词序列（前 15 个）: {final[:15]} ...")
print("  → 看：'low' 因为频繁出现被合并成一个子词；这正是 BPE 构建词表的方式。")


# ============================================================
# 【6】连回面试
# ============================================================
print("\n" + "=" * 64)
print("✅ 第1课要点（连回面试）")
print("=" * 64)
print("  • 模型不读字符也不读词，读的是 token（子词单元）；tokenizer 负责「切」。")
print("  • GPT/LLaMA 用 Byte-level BPE：以字节为基本符号，")
print("    迭代合并【最频繁共现的相邻字节对】，选对标准是「频率」。")
print("  • BPE vs WordPiece：BPE 选频率，WordPiece（BERT）选似然增益（[S1] §3.2）。")
print("  • 词表训练前固定、跨模型不通用：GPT 的 token id 不能喂给 LLaMA。")
print("  • 中文更费 token：同语义中文 token 数多 → 更贵、更早撑爆上下文窗口。")
print("  • token 是计量单位：上下文窗口（如 128k）、API 计费都以 token 计。")
print("=" * 64)
