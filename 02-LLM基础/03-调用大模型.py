# 依赖（二选一，按你有的环境选）：
#   方案A（本地，免费）：先装 ollama，再拉模型  →  brew install ollama && ollama run qwen2.5
#   方案B（云端，需 API key）：pip/uv add openai  →  export OPENAI_API_KEY="sk-..."
# 不要自动安装——避免和你的环境冲突。
"""
LLM 基础 第3课：调用大模型（Ollama 本地 / OpenAI 云端）
=====================================================
参考：知识库《02-LLM基础/系统知识.md》§4 上下文窗口 / §5 采样 / §6 幻觉
      Ollama 官方文档 https://ollama.com
      OpenAI API Reference https://platform.openai.com/docs/api-reference

本文件演示「如何用代码调用一个真正的大模型」，并对比本地 vs 云端两条路线。
学完本课后你会明白：
  • API 请求 = 「messages 列表」+ 「采样参数」+ 「模型名」
  • role 的含义（system / user / assistant）
  • temperature / top_p 怎么通过 API 传（对应第2课手写的那些策略）
  • 调用失败要 try/except 给出友好提示（生产必备）

运行：
  方案A（ollama）：
    uv run --directory /Users/sunhuanyu/ai-llm-learning python "02-LLM基础/03-调用大模型.py"
  方案B（OpenAI）：先 export OPENAI_API_KEY="sk-..."，再同上运行。
"""
import os
import sys


# ============================================================
# 方案A：调用本地 Ollama（http://localhost:11434，兼容 OpenAI 接口）
# 需要: ollama run qwen2.5   （先在终端跑一次拉模型）
# 依赖: uv add openai        （用 openai SDK 连 ollama 的 OpenAI 兼容端点）
# ============================================================
def call_ollama(prompt: str, model: str = "qwen2.5", temperature: float = 0.7,
                top_p: float = 0.9, max_tokens: int = 256) -> str:
    """调用本地 Ollama 模型。

    Ollama 在本地启动后，会在 http://localhost:11434 提供「OpenAI 兼容」的
    /v1/chat/completions 接口，所以可以直接用 openai SDK，只是 base_url 不同。
    好处：换到云端 OpenAI 只需改 base_url + 加 API key。
    """
    from openai import OpenAI  # 延迟导入：只在实际调用时才需要
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # ollama 不校验 key，随便填，但 SDK 要求非空
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个简洁的中文助手，回答尽量精炼。"},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,  # 对应第2课的 temperature 采样
        top_p=top_p,              # 对应第2课的 top-p 采样
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# 方案B：调用 OpenAI 云端 API
# 需要: export OPENAI_API_KEY="sk-..."   （https://platform.openai.com/api-keys 申请）
# 依赖: uv add openai
# ============================================================
def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7,
                top_p: float = 0.9, max_tokens: int = 256) -> str:
    """调用 OpenAI 官方 API。需要环境变量 OPENAI_API_KEY。"""
    from openai import OpenAI
    client = OpenAI()  # 默认读环境变量 OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个简洁的中文助手，回答尽量精炼。"},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# 统一入口：自动选择可用的后端，try/except 友好提示
# ============================================================
def chat(prompt: str, backend: str = "auto", **kw) -> str:
    """统一调用入口。

    backend:
      "auto"    —— 先试 ollama，连不上就试 openai，再不行给提示
      "ollama"  —— 只用本地 ollama
      "openai"  —— 只用云端 openai
    """
    backends = []
    if backend == "auto":
        backends = ["ollama", "openai"]   # 默认先试免费的本地
    else:
        backends = [backend]

    errors = []
    for b in backends:
        try:
            if b == "ollama":
                print(f"  [尝试] 本地 Ollama（http://localhost:11434）...")
                return call_ollama(prompt, **kw)
            elif b == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    errors.append("openai: 未设置环境变量 OPENAI_API_KEY")
                    continue
                print(f"  [尝试] 云端 OpenAI API...")
                return call_openai(prompt, **kw)
        except Exception as e:
            msg = str(e)
            # 只取错误的前 120 字，避免长 traceback
            errors.append(f"{b}: {msg[:120]}")
            print(f"  [失败] {b} → {msg[:100]}")

    # 全部失败 → 给出友好的、可操作的提示（而不是抛异常吓到新手）
    print()
    print("=" * 64)
    print("❌ 所有后端都不可用。请按以下任一方案配置：")
    print("=" * 64)
    print()
    print("方案A（推荐·免费·本地）：")
    print("  1. 安装 ollama：     brew install ollama")
    print("  2. 启动并拉模型：    ollama run qwen2.5")
    print("  3. 装 SDK：          uv add openai")
    print("  4. 重跑本文件")
    print()
    print("方案B（云端·需付费 API key）：")
    print("  1. 申请 key：        https://platform.openai.com/api-keys")
    print("  2. 设置环境变量：    export OPENAI_API_KEY=\"sk-...\"")
    print("  3. 装 SDK：          uv add openai")
    print("  4. 重跑本文件")
    print()
    print("具体错误信息：")
    for e in errors:
        print(f"  - {e}")
    print("=" * 64)
    return ""


# ============================================================
# 主流程：用同一组 prompt 跑几个对比实验
# ============================================================
def main():
    print("=" * 64)
    print("【1】基础对话：问模型一个问题")
    print("=" * 64)
    answer = chat("用一句话解释什么是「大语言模型」。", temperature=0.3)
    if answer:
        print(f"\n  模型回答：{answer}\n")

    print("=" * 64)
    print("【2】温度对比：同一问题，低温 vs 高温（看创造性差异）")
    print("=" * 64)
    prompt2 = "给「月亮」写一句比喻句。"
    print(f"  prompt: {prompt2}")
    for T in [0.2, 1.0]:
        print(f"\n  --- temperature={T} ---")
        a = chat(prompt2, temperature=T, max_tokens=80)
        if a:
            print(f"  {a}")

    print()
    print("=" * 64)
    print("【3】system prompt 的作用：约束模型角色")
    print("=" * 64)
    # 通过改 messages 里的 system 角色来控制模型行为（不是在本文件里改，
    # 而是演示「同一模型，不同 system prompt → 不同表现」这个工程要点）。
    print("  （本演示默认 system = '简洁的中文助手'；生产中常把角色/规则写在 system 里）")
    print("  role 三种：system(设定角色/规则) / user(用户提问) / assistant(模型回答)")

    print()
    print("=" * 64)
    print("✅ 第3课要点（连回面试）")
    print("=" * 64)
    print("  • 调用 LLM = 发 {messages, model, temperature, top_p, max_tokens} 给 API")
    print("  • messages 用 role 区分：system(设定) / user(提问) / assistant(历史回答)")
    print("  • temperature/top_p 就是第2课手写的采样参数，API 原样透传给模型")
    print("  • 本地(ollama) vs 云端(openai)：ollama 提供 OpenAI 兼容接口，切换只需改 base_url")
    print("  • 生产必备：try/except + 友好降级提示，别让用户看到裸 traceback")
    print("=" * 64)


if __name__ == "__main__":
    main()
