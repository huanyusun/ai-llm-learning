"""
微调与部署 第4课：推理优化 —— KV Cache 原理演示（纯 numpy）
============================================================
参考论文：
  FlashAttention: Fast and Memory-Efficient Exact Attention
  Dao et al., 2022, arXiv:2205.14135
  PagedAttention / vLLM: Efficient Memory Management for LLM Serving
  Kwon et al., 2023, SOSP'23, arXiv:2309.06180

核心一句话：自回归生成时，前面 token 的 K、V 不会变 → 【缓存起来】，
  下一步只算新 token 的 Q 去查全部 K/V，避免每步重算整段前缀。

本文件做什么（全部 numpy 可跑）：
  【1】手写 attention：带/不带 KV cache 两种实现
  【2】跑一段「生成」，对比两种方式的【总浮点运算量】和【墙钟时间】
  【3】可视化：每步计算量随 token 数的变化（cache 是 O(1)/步，无 cache 是 O(t)/步）
  【4】串讲：KV cache 的显存代价 → PagedAttention 的分页管理

运行：uv run python "06-微调与部署/04-推理优化演示.py"
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(0)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT_DIR, exist_ok=True)

# 用一个「单层单头」attention 做原理演示（真实是多头 + 多层，规模等比放大）
d_model = 128
seq_len_demo = 64

# 随便造一组 Q/K/V 投影矩阵（真实场景从基座加载）
W_q = rng.standard_normal((d_model, d_model)) * 0.05
W_k = rng.standard_normal((d_model, d_model)) * 0.05
W_v = rng.standard_normal((d_model, d_model)) * 0.05
scale = 1.0 / np.sqrt(d_model)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================
# 【1】两种 attention 实现
# ============================================================
print("=" * 60)
print("【1】手写 attention：无 cache vs 有 KV cache")
print("=" * 60)

def attention_no_cache(all_tokens, W_q, W_k, W_v, scale):
    """
    无 cache：每生成一个新 token，都把【整段历史】重新算一遍 K、V。
    输入 all_tokens: (t, d) —— 到目前为止所有 token 的表示
    返回：最后一个 token 的 attention 输出 (d,)
    本步浮点运算量 ∝ t（线性于已生成长度）
    """
    Q = all_tokens @ W_q                    # (t, d)
    K = all_tokens @ W_k                    # (t, d)  ← 每次都重算历史 K
    V = all_tokens @ W_v                    # (t, d)  ← 每次都重算历史 V
    scores = (Q @ K.T) * scale              # (t, t)
    attn = softmax(scores, axis=-1)
    out = attn @ V                          # (t, d)
    return out[-1], Q, K, V                 # 返回最后 token 输出 + 中间量（供计数）


def attention_with_cache(new_token, K_cache, V_cache, W_q, W_k, W_v, scale):
    """
    有 KV cache：历史 K、V 已缓存，本步只算【新 token】的 K、V 并追加。
    输入 new_token: (d,) 当前新 token；K_cache/V_cache: (t_old, d)
    返回：新 token 的 attention 输出 (d,)，以及更新后的 K_cache, V_cache
    本步浮点运算量 ∝ t_old（只算 1 个 Q 去查全部 K/V），省掉了历史 K/V 的重算
    """
    q = new_token @ W_q                     # (d,)      ← 只算 1 个 Q
    k_new = new_token @ W_k                 # (d,)      ← 只算 1 个新 K
    v_new = new_token @ W_v                 # (d,)      ← 只算 1 个新 V
    K_cache = np.vstack([K_cache, k_new])   # (t_old+1, d)  追加
    V_cache = np.vstack([V_cache, v_new])
    scores = (K_cache @ q) * scale          # (t_old+1,)    Q·全部 K
    attn = softmax(scores)
    out = attn @ V_cache                    # (d,)
    return out, K_cache, V_cache

print("  无 cache：每步重算所有历史的 K、V → 计算量随序列线性增长")
print("  有 cache：历史 K、V 缓存，每步只算 1 个新 Q/K/V → 计算量大幅降低")
print("  数学结果完全相同（cache 不改变 attention 输出，只是避免重算）")


# ============================================================
# 【2】生成 demo：对比总计算量 + 墙钟时间
# ============================================================
print()
print("=" * 60)
print("【2】生成 demo：生成 N 个 token，对比两种方式")
print("=" * 60)

def generate_no_cache(N, d):
    """模拟无 cache 生成：每步都从头算整段 attention。"""
    tokens = rng.standard_normal((1, d))    # 起始 token
    flops_total = 0
    for t in range(1, N):
        all_tokens = np.vstack([tokens, rng.standard_normal((1, d))])
        out, Q, K, V = attention_no_cache(all_tokens, W_q, W_k, W_v, scale)
        tokens = all_tokens
        # 估算这步的浮点运算量（Q/K/V 投影 + score + softmax×V）
        t_len = all_tokens.shape[0]
        # 投影：3 次 (t,d)@(d,d) ≈ 3*t*d*d；score: t*t*d；输出: t*t*d
        flops_total += 3 * t_len * d * d + t_len * t_len * d + t_len * t_len * d
    return flops_total

def generate_with_cache(N, d):
    """模拟有 cache 生成：每步只算新 token。"""
    tokens = rng.standard_normal((1, d))
    # 第 0 步建立初始 cache（prefill：把第 0 个 token 的 K/V 存进去）
    K_cache = tokens @ W_k
    V_cache = tokens @ W_v
    flops_total = 0
    for t in range(1, N):
        new_token = rng.standard_normal((d,))
        out, K_cache, V_cache = attention_with_cache(
            new_token, K_cache, V_cache, W_q, W_k, W_v, scale)
        # 本步只算 1 个新 Q/K/V：3 次 (d,)@(d,d) ≈ 3*d*d；
        # score: t*d；输出: t*d
        t_old = K_cache.shape[0]
        flops_total += 3 * d * d + t_old * d + t_old * d
    return flops_total

N = 64
flops_no = generate_no_cache(N, d_model)
flops_yes = generate_with_cache(N, d_model)
print(f"  生成 {N} 个 token（d={d_model}）的总浮点运算量估算：")
print(f"    无 cache   : {flops_no:,.0f}")
print(f"    有 KV cache: {flops_yes:,.0f}")
print(f"    节省比例   : {1 - flops_yes/flops_no:.1%}  （cache 把每步 O(t) 降到接近 O(1) 的投影开销）")

# 墙钟时间对比（放大循环次数以让差异可测）
def time_generate(N, d, repeats=30, use_cache=True):
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        if use_cache:
            generate_with_cache(N, d)
        else:
            generate_no_cache(N, d)
        best = min(best, time.perf_counter() - t0)
    return best

t_no = time_generate(N, d_model, repeats=20, use_cache=False)
t_yes = time_generate(N, d_model, repeats=20, use_cache=True)
print(f"\n  墙钟时间（{20} 次取最小）：")
print(f"    无 cache   : {t_no*1000:7.2f} ms")
print(f"    有 KV cache: {t_yes*1000:7.2f} ms")
print(f"    加速比     : {t_no/t_yes:.1f}×")


# ============================================================
# 【3】正确性验证：两种方式输出应当完全相同
# ============================================================
print()
print("=" * 60)
print("【3】正确性：cache 不改变 attention 结果")
print("=" * 60)

np.random.seed(1)
test_tokens = np.random.randn(10, d_model) * 0.3
# 无 cache 方式算最后一步
out_no, _, _, _ = attention_no_cache(test_tokens, W_q, W_k, W_v, scale)
# 有 cache 方式逐步累积到第 10 步
K_c = np.empty((0, d_model)); V_c = np.empty((0, d_model))
out_last = None
for i in range(10):
    out_last, K_c, V_c = attention_with_cache(
        test_tokens[i], K_c, V_c, W_q, W_k, W_v, scale)
diff = np.max(np.abs(out_no - out_last))
print(f"  两种方式最后一个 token 输出的最大差异 = {diff:.2e}")
print("  （浮点噪声级 → 证明 KV cache 数学等价，只是避免重算）")


# ============================================================
# 【4】每步计算量随 token 数的变化
# ============================================================
print()
print("=" * 60)
print("【4】每步计算量 vs token 位置（cache 的核心收益）")
print("=" * 60)

steps = np.arange(1, N + 1)
# 无 cache：第 t 步重算整段 → 投影 3*t*d*d + score/output ~2*t*t*d
per_step_no = 3 * steps * d_model * d_model + 2 * steps * steps * d_model
# 有 cache：第 t 步只算 1 个新 Q/K/V → 3*d*d + score/output ~2*t*d
per_step_yes = 3 * d_model * d_model + 2 * steps * d_model
print(f"  第 1 步：无 cache {per_step_no[0]:>10,.0f} | 有 cache {per_step_yes[0]:>10,.0f}")
print(f"  第 {N} 步：无 cache {per_step_no[-1]:>10,.0f} | 有 cache {per_step_yes[-1]:>10,.0f}")
print(f"  无 cache 每步随 t【线性增长】；有 cache 每步【几乎恒定】")


# ============================================================
# 【5】可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

# 左：每步计算量
ax = axes[0]
ax.plot(steps, per_step_no, '-', color='#d62728', label='无 cache（每步重算）')
ax.plot(steps, per_step_yes, '-', color='#2ca02c', label='有 KV cache')
ax.set_xlabel('已生成 token 数 t'); ax.set_ylabel('每步浮点运算量')
ax.set_title('每步计算量：cache 后【几乎恒定】\n无 cache 随 t 线性增长')
ax.legend(); ax.grid(alpha=0.3)

# 右：累积计算量（生成 N 个 token 的总量）
ax = axes[1]
cum_no = np.cumsum(per_step_no)
cum_yes = np.cumsum(per_step_yes)
ax.plot(steps, cum_no, '-', color='#d62728', label='无 cache 累积')
ax.plot(steps, cum_yes, '-', color='#2ca02c', label='有 KV cache 累积')
ax.fill_between(steps, cum_yes, cum_no, alpha=0.15, color='#d62728')
ax.set_xlabel('生成长度'); ax.set_ylabel('累积浮点运算量')
ax.set_title(f'生成 {N} token 的总计算量\n(cache 省 {1-cum_yes[-1]/cum_no[-1]:.0%})')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, "04-KVcache.png")
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"\n【5】图已保存：{out}")


# ============================================================
# 要点小结（连回 PagedAttention / vLLM）
# ============================================================
print()
print("=" * 60)
print("✅ KV Cache / 推理优化 要点（连回面试）")
print("=" * 60)
print("  • 自回归生成是【访存受限 memory-bound】(arXiv:2309.06180)")
print("  • KV cache：缓存历史的 K/V，每步只算新 token 的 Q → 避免重算前缀")
print("  • 收益：每步从 O(t) 降到接近 O(1) 投影开销，生成越长收益越大")
print("  • 代价：KV cache 占显存巨大（13B 模型近 30% 显存给 KV）")
print("  • PagedAttention：借鉴 OS 分页，KV 按块非连续存放 → 零碎片")
print("    （传统连续分配实测仅 20.4%~38.2% 显存被有效利用）")
print("  • vLLM：PagedAttention + Continuous Batching → 吞吐 2~4×")
print("  • Continuous Batching：每步重组 batch，结束的移出、新的插入")
print("  • FlashAttention：tiling 在 SRAM 做 softmax，不物化 N×N 矩阵")
print("    精确（非近似），HBM 访问最多少 9×，显存随序列线性")
print("  • Speculative Decoding：小模型猜+大模型验，输出分布不变")
print("=" * 60)
