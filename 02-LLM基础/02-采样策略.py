"""
LLM 基础 第2课：采样策略（Temperature / Top-k / Top-p）
======================================================
纯 numpy 实现，不依赖任何外部 LLM——直接能跑通。
参考：知识库《02-LLM基础/系统知识.md》§5 采样策略
      Survey of LLMs §6.1（[S1]）—— temperature/top-k/top-p 公式来源

问题：自回归生成时，模型每一步在「整个词表」上输出一个概率分布。
      下一个 token 怎么选？
        • 贪心（greedy）：永远选概率最大的 → 单调、易重复、没创意
        • 纯随机采样：完全按分布抽 → 太发散、可能出乱码
        • 三大采样策略在这两者之间找平衡：
            ① Temperature  —— 调整分布「尖锐度」
            ② Top-k        —— 只在概率最高的前 k 个里采样
            ③ Top-p（nucleus）—— 在累计概率达到 p 的最小集合里采样

【本课重点】用 numpy 手写这三个策略，并画出它们如何改变分布。
            理解为什么「生产里常用 temperature≈0.7, top_p≈0.9」。

运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "02-数学基础/02-采样策略.py"
     （只依赖 numpy + matplotlib，已在 pyproject.toml 中，无需额外安装）
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设随机种子，保证每次运行图一样（教学可复现）
rng = np.random.default_rng(42)


# ============================================================
# 【1】softmax + 温度：把 logits 变成概率（带数值稳定）
# ============================================================
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """带温度的 softmax。

    公式（[S1] §6.1 原文式10）：
        P(x_j) = exp(l_j / t) / Σ exp(l_{j'} / t)

    行为（[S1] §6.1）：
      t→0   退化为贪心（argmax）
      t=1   默认随机采样
      t→∞   退化为均匀采样

    注意：是 logits 除以 t（不是乘）。低温 t<1 让分布更尖锐。
    """
    logits = np.asarray(logits, dtype=float)
    # 数值稳定：减最大值，避免 exp 溢出（softmax 标准技巧）
    scaled = logits / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


# ============================================================
# 【2】Temperature 采样：先调温度，再按分布抽
# ============================================================
def sample_with_temperature(logits, temperature=1.0, n=1):
    """调温度后按概率采样。temperature→0 时退化为贪心。"""
    if temperature <= 1e-8:
        # t→0 = 贪心：直接取 argmax（[S1] §6.1）
        idx = [int(np.argmax(logits)) for _ in range(n)]
        return np.array(idx)
    probs = softmax(logits, temperature)
    # 按概率分布抽 n 个样本（rng.choice 要求概率和为 1，浮点误差补一下）
    probs = probs / probs.sum()
    return rng.choice(len(probs), size=n, p=probs)


# ============================================================
# 【3】Top-k 采样：截断到前 k 高，重缩放后采样
# ============================================================
def top_k_sampling(logits, k=5, temperature=1.0, n=1):
    """Top-k：只在概率最高的前 k 个 token 里采样（[S1] §6.1）。

    步骤：
      ① 先过温度 softmax 得到原始概率；
      ② 除前 k 大的概率外，全部置 0（截断低概率词）；
      ③ 重新归一化（让剩下 k 个概率和为 1）；
      ④ 在这 k 个里采样。

    特点：k 是【固定数量】。分布尖锐时 k 太大会放进噪声词，
          分布平坦时 k 太小会丢掉合理候选。
    """
    probs = softmax(logits, temperature)
    # 找到概率第 k 大的阈值，低于它的全部置 0
    if k < len(probs):
        # argsort 升序，取倒数第 k 个就是第 k 大的值
        threshold_idx = np.argsort(probs)[-k]
        threshold = probs[threshold_idx]
        mask = probs >= threshold       # 保留前 k 大（含等值的）
        probs = np.where(mask, probs, 0.0)
    probs = probs / probs.sum()         # 重缩放
    return rng.choice(len(probs), size=n, p=probs)


# ============================================================
# 【4】Top-p 采样（nucleus）：累计概率达到 p 的最小集合
# ============================================================
def top_p_sampling(logits, p=0.9, temperature=1.0, n=1):
    """Top-p / nucleus：取累计概率 ≥ p 的【最小】集合（[S1] §6.1）。

    步骤：
      ① 过温度 softmax；
      ② 按概率降序排列；
      ③ 逐个累加，直到累计概率 ≥ p，截断后面的；
      ④ 重缩放后在该集合内采样。

    特点：候选数【自适应】。
      • 分布尖锐（一个词独大）→ 集合很小（保守，甚至就 1 个词）
      • 分布平坦（多词势均力敌）→ 集合很大（保留多样性）
      这就是为什么 top-p 通常比 top-k 更鲁棒（[S1] §6.1）。
    """
    probs = softmax(logits, temperature)
    # 按概率降序排序
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    # 计算累计概率
    cumulative = np.cumsum(sorted_probs)
    # 找到第一个使累计概率 >= p 的位置，保留到它（含）
    # 这里用 >=：确保集合是「累计概率 >= p 的最小集合」
    cutoff = np.searchsorted(cumulative, p)  # 第一个 cumulative >= p 的下标
    cutoff = min(cutoff, len(sorted_probs) - 1)
    # 保留 0..cutoff，其余置 0
    keep_mask_sorted = np.zeros_like(sorted_probs, dtype=bool)
    keep_mask_sorted[:cutoff + 1] = True
    # 映射回原始顺序
    keep_mask = np.zeros_like(probs, dtype=bool)
    keep_mask[sorted_idx] = keep_mask_sorted
    new_probs = np.where(keep_mask, probs, 0.0)
    new_probs = new_probs / new_probs.sum()
    return rng.choice(len(new_probs), size=n, p=new_probs)


# ============================================================
# 【5】构造一个「模拟词表概率分布」
# ============================================================
print("=" * 64)
print("【5】构造模拟场景：假设下一步候选词的概率分布")
print("=" * 64)

# 模拟一个 20 词的小词表。我们直接给 logits（模拟模型输出）。
vocab = ["今天", "天气", "很好", "不错", "下雨", "晴天", "外出", "在家",
         "开心", "难过", "abc", "xyz", "嗯", "的", "了", "是", "在",
         "我", "你", "他"]
assert len(vocab) == 20

# 模拟模型对「今天天气__」的预测：前几个合理候选概率较高，后面是噪声词
logits = np.array([
    3.5,   # 今天
    2.8,   # 天气
    2.2,   # 很好
    2.0,   # 不错
    1.5,   # 下雨
    1.3,   # 晴天
    1.0,   # 外出
    0.8,   # 在家
    0.6,   # 开心
    0.4,   # 难过
   -1.0, -1.2, -1.5, -1.8, -2.0, -2.2, -2.5, -3.0, -3.2, -3.5
])

base_probs = softmax(logits, temperature=1.0)
print(f"\n候选词表（{len(vocab)} 个）：{vocab}")
print(f"\n默认（t=1）概率分布（前 6 个）：")
for i in np.argsort(base_probs)[::-1][:6]:
    print(f"    {vocab[i]:>4}: {base_probs[i]:.4f}")


# ============================================================
# 【6】采样演示：同一分布下，不同策略抽到什么
# ============================================================
print("\n" + "=" * 64)
print("【6】采样演示：同一分布，不同策略/参数抽到的词（各抽 8 次）")
print("=" * 64)

def show_samples(name, fn, **kw):
    samples = fn(logits, n=8, **kw)
    words = [vocab[i] for i in samples]
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    summary = ", ".join(f"{w}×{c}" for w, c in sorted(counts.items(), key=lambda x: -x[1]))
    print(f"  {name:<32}: {summary}")

print()
show_samples("贪心 (t→0)",          sample_with_temperature, temperature=0.0)
show_samples("低温 t=0.3",          sample_with_temperature, temperature=0.3)
show_samples("默认 t=1.0",          sample_with_temperature, temperature=1.0)
show_samples("高温 t=2.0",          sample_with_temperature, temperature=2.0)
print()
show_samples("Top-k k=3",           top_k_sampling, k=3)
show_samples("Top-k k=5",           top_k_sampling, k=5)
print()
show_samples("Top-p p=0.5",         top_p_sampling, p=0.5)
show_samples("Top-p p=0.9",         top_p_sampling, p=0.9)

print("\n  → 贪心永远只抽到概率最大的「今天」；")
print("  → 低温抽到的高频词占比更大；高温把低频噪声词也拉进来；")
print("  → Top-k/Top-p 把低概率噪声词截掉，只在「合理候选」里采样。")


# ============================================================
# 【7】可视化 1：Temperature 如何改变分布形状
# ============================================================
print("\n" + "=" * 64)
print("【7】生成可视化图（保存到 img/）")
print("=" * 64)

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(img_dir, exist_ok=True)

temps = [0.3, 0.7, 1.0, 1.5]
fig, axes = plt.subplots(1, len(temps), figsize=(4 * len(temps), 4.2), sharey=True)
for ax, T in zip(axes, temps):
    p = softmax(logits, T)
    colors = ['#4C72B0'] * len(vocab)
    # 给 top-3 染色突出
    top3 = np.argsort(p)[-3:]
    for j in top3:
        colors[j] = '#C44E52'
    ax.bar(range(len(vocab)), p, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title(f'temperature = {T}', fontsize=11)
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab, rotation=90, fontsize=7)
    ax.set_ylim(0, max(base_probs) * 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    if T == temps[0]:
        ax.set_ylabel('概率')
plt.suptitle('Temperature 采样：低温→尖锐（贪心），高温→平滑（均匀）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out1 = os.path.join(img_dir, "02-temperature.png")
plt.savefig(out1, dpi=110, bbox_inches='tight')
plt.close()
print(f"  图1 已保存：{out1}")


# ============================================================
# 【8】可视化 2：Top-k vs Top-p 如何截断（对比柱状图）
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

# 辅助函数：算出「截断并重缩放后」的概率，用于可视化对比
def _viz_top_k(p, k):
    mask = p >= np.sort(p)[-k]
    q = np.where(mask, p, 0.0); return q / q.sum()
def _viz_top_p(p, thr):
    sidx = np.argsort(p)[::-1]
    cum = np.cumsum(p[sidx])
    cutoff = np.searchsorted(cum, thr) + 1
    keep = sidx[:cutoff]
    q = np.zeros_like(p); q[keep] = p[keep]; return q / q.sum()

viz = [
    ("原始分布 (t=1)", base_probs),
    ("Top-k = 5\n(固定保留 5 个)", _viz_top_k(base_probs, 5)),
    ("Top-p = 0.9\n(累计概率≥0.9)", _viz_top_p(base_probs, 0.9)),
    ("Top-p = 0.5\n(分布尖→集合小)", _viz_top_p(base_probs, 0.5)),
]
for ax, (title, p) in zip(axes, viz):
    nonzero = p > 0
    colors = ['#C44E52' if nonzero[i] else '#DDDDDD' for i in range(len(vocab))]
    ax.bar(range(len(vocab)), p, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab, rotation=90, fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    if title.startswith("原始"):
        ax.set_ylabel('概率')
plt.suptitle('Top-k（固定候选数）vs Top-p（累计概率自适应候选数）',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(img_dir, "02-topk-topp.png")
plt.savefig(out2, dpi=110, bbox_inches='tight')
plt.close()
print(f"  图2 已保存：{out2}")


# ============================================================
# 【9】可视化 3：采样稳定性——温度越低，结果越集中
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
scan_temps = np.linspace(0.05, 2.0, 60)
top1_share = []   # 在该温度下，抽 1000 次，最大概率词占比
for T in scan_temps:
    samples = sample_with_temperature(logits, temperature=T, n=1000)
    # 统计最常被抽到的词的占比
    vals, counts = np.unique(samples, return_counts=True)
    top1_share.append(counts.max() / 1000)

ax.plot(scan_temps, top1_share, color='#4C72B0', lw=2.5)
ax.fill_between(scan_temps, top1_share, alpha=0.15, color='#4C72B0')
ax.set_xlabel('temperature')
ax.set_ylabel('最常被抽到词的占比（1000 次采样）')
ax.set_title('温度越低 → 采样越集中（趋于贪心）；温度越高 → 越发散',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
ax.axhline(base_probs.max(), color='gray', ls='--', lw=1,
          label=f'最大词理论概率={base_probs.max():.3f}')
ax.legend()
plt.tight_layout()
out3 = os.path.join(img_dir, "02-采样稳定性.png")
plt.savefig(out3, dpi=110, bbox_inches='tight')
plt.close()
print(f"  图3 已保存：{out3}")


# ============================================================
# 【10】连回面试
# ============================================================
print("\n" + "=" * 64)
print("✅ 第2课要点（连回面试）")
print("=" * 64)
print("  • 采样 = 在词表概率分布上选下一个 token；策略决定多样性与质量。")
print("  • Temperature（[S1] §6.1 式10）：logits / t 再 softmax。")
print("      t→0 贪心；t=1 默认采样；t→∞ 均匀。")
print("      ⚠ 易错：是 logits 除以 t（不是乘）；t=0 是贪心不是「无采样」。")
print("  • Top-k：截断到概率最高的固定 k 个，重缩放后采样。k 固定。")
print("  • Top-p / nucleus：累计概率 ≥ p 的最小集合，重缩放后采样。")
print("      候选数自适应——比 top-k 鲁棒（[S1] §6.1）。")
print("  • ⚠ Top-p=1 仍采样，不是贪心。")
print("  • 生产经验：事实问答/RAG 用低温度（0~0.3）；创意写作用高温度（0.7~1.0）；")
print("      常配 top_p≈0.9 或 top_k≈40。三者可叠加（先 temperature 再 top-k/p）。")
print("=" * 64)
