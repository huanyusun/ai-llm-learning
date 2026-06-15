"""
微调与部署 第3课：本地部署与调用（ollama / vLLM）
====================================================
本文件演示两种主流本地部署方案：
  (A) ollama —— 本地/端侧一键运行（GGUF，Mac/CPU 友好）
  (B) vLLM   —— 高吞吐 GPU 服务引擎（PagedAttention、Continuous Batching）

参考论文：
  PagedAttention / vLLM, Kwon et al., 2023, arXiv:2309.06180

⚠️ 本文件【需要真实模型】，无法纯 numpy 跑通。请按下方注释准备环境后再运行。
   - 方案 A 需要：已安装 ollama，并 `ollama pull qwen2.5`
   - 方案 B 需要：GPU + 安装 vllm（`uv pip install vllm`）
   没装环境时，import 会失败 → 走 except 分支打印环境提示，不会崩。

运行：
  uv run python "06-微调与部署/03-本地部署调用.py"
"""

# ============================================================
# 方案 A：ollama —— 本地一键部署 + 调用（含流式输出）
# ============================================================
# 准备步骤：
#   1. 安装 ollama：https://ollama.com （macOS/Linux/Windows 都有一键包）
#   2. 拉模型（在终端执行）：
#        ollama pull qwen2.5                # 需要: ollama pull qwen2.5
#      也可换更小/更大：qwen2.5:0.5b / qwen2.5:7b / llama3.1 / gemma2 ...
#   3. 启动后默认监听 http://localhost:11434
#   4. 运行本文件。

def demo_ollama():
    """ollama 调用：先非流式，再流式（逐 token 返回）。"""
    import urllib.request, json

    OLLAMA_URL = "http://localhost:11434/api/chat"
    MODEL = "qwen2.5"                       # 需要: ollama pull qwen2.5
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一名资深 AI 工程师，回答简洁。"},
            {"role": "user", "content": "用一句话解释 LoRA 微调的原理。"},
        ],
        "stream": False,                    # 先演示非流式
        "options": {"temperature": 0.3},
    }

    # ---- (1) 非流式：一次性返回完整结果 ----
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        print("【ollama 非流式】回答：")
        print("  ", data["message"]["content"])
        print(f"  (eval_count={data.get('eval_count')}, "
              f"eval_duration={data.get('eval_duration',0)/1e9:.2f}s)")
    except Exception as e:
        print(f"  [ollama 非流式调用失败] {e}")
        print("  提示：确认 ollama 已启动并已 `ollama pull qwen2.5`")
        return

    # ---- (2) 流式：逐 token 返回（SSE-like，每行一个 JSON）----
    print("\n【ollama 流式】逐 token 输出：")
    payload["stream"] = True
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:                # 逐行读取
                if not raw.strip():
                    continue
                chunk = json.loads(raw.decode())
                delta = chunk.get("message", {}).get("content", "")
                print(delta, end="", flush=True)   # 流式打印
                if chunk.get("done"):
                    break
        print("\n  [流式结束]")
    except Exception as e:
        print(f"\n  [ollama 流式调用失败] {e}")


# ============================================================
# 方案 B：vLLM —— 高吞吐 GPU 服务引擎（标注，按需启用）
# ============================================================
# vLLM 论文核心 (arXiv:2309.06180)：
#   - PagedAttention：KV cache 按块非连续存放，显存近零碎片
#   - Continuous Batching：每步重组 batch，吞吐 2~4×
#   - OpenAI 兼容 API：直接用 openai SDK 调用
#
# 准备步骤（需要 GPU）：
#   uv pip install vllm                      # 安装
#
# 方式 1：用 Python 直接跑离线推理（OpenAI 兼容 API 之外的最简用法）
def demo_vllm_offline():
    """
    vLLM 离线推理（不启服务，直接在进程内推理）。
    需要：GPU + `uv pip install vllm`
    """
    from vllm import LLM, SamplingParams

    # 加载模型（vLLM 会自动用 PagedAttention 管理 KV 显存）
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")   # 换成你有的模型
    sampling = SamplingParams(temperature=0.3, max_tokens=128)

    prompts = ["用一句话解释 PagedAttention 的原理。"]
    outputs = llm.generate(prompts, sampling)
    for o in outputs:
        print("【vLLM 离线】回答：", o.outputs[0].text)


# 方式 2：启动 OpenAI 兼容服务，再用 openai SDK 调用
r"""
# 终端启动 vLLM 服务（OpenAI 兼容 API）：
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --enable-prefix-caching \           # 前缀缓存（复用公共前缀 KV）
  --speculative-model "[ngram]" \     # 投机解码（可选）
  --quantization awq                  # 量化（可选：awq/gptq/fp8/gguf）

# 然后用标准 openai SDK 调用（含流式）：
"""
def demo_vllm_service():
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": "用一句话解释 Continuous Batching。"}],
        stream=True,                                  # 流式输出
    )
    print("【vLLM 服务流式】：", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


# ============================================================
# 主入口：默认只跑 ollama（最易得），vLLM 按需取消注释
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("方案 A：ollama 本地部署调用")
    print("=" * 60)
    try:
        demo_ollama()
    except Exception as e:
        print(f"  [ollama 环境不可用] {e}")
        print("  请先安装 ollama 并 `ollama pull qwen2.5`，再运行本文件。")

    print()
    print("=" * 60)
    print("方案 B：vLLM（需要 GPU，本机默认跳过；环境就绪时取消下方注释）")
    print("=" * 60)
    print("  vLLM 是生产 GPU 服务的首选：PagedAttention + Continuous Batching，")
    print("  吞吐比 FasterTransformer/Orca 高 2~4×（arXiv:2309.06180）。")
    print("  用法见上方 demo_vllm_offline() / demo_vllm_service()。")
    # 取消下面两行注释来实际运行（需先 `uv pip install vllm`）：
    # demo_vllm_offline()
    # demo_vllm_service()
