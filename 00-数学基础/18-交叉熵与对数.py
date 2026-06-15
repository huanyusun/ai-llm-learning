"""
数学基础 第18课：交叉熵与对数（从零开始）
=========================================
参考：3Blue1Brown《信息论》——熵、交叉熵、KL 散度的直观讲解
      《动手学深度学习》https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html
        （交叉熵作为分类 / 语言模型的损失函数）

一句话串起来：
  • 信息熵 H：一个分布有多「不确定」（越乱熵越大）
  • 交叉熵 H(p,q)：用分布 q 去编码「真实分布 p」要花多少信息（成本）
  • KL 散度 KL(p‖q)：交叉熵比真实熵「多花」的那部分 = 两个分布的差异
  • 关键等式：H(p,q) = H(p) + KL(p‖q)
  • 分类 / 语言模型里，真实分布 p 就是「正确答案」(one-hot)，
    所以 H(p)=0，此时【交叉熵 = KL】，模型干脆直接最小化交叉熵。

为什么用 -log(预测概率) 当损失？
  • 模型对正确答案给的预测概率 p 越大，损失 -log(p) 越小（趋近 0）；
  • p 越小（预测越离谱），-log(p) 越大，且 →∞（狠狠惩罚「错得离谱」）。
  • 这正是 LLM 训练时「最小化下一个词的负对数似然」的来源。

【上一课 softmax + 本课交叉熵 = 分类模型的完整损失】
  softmax 负责把分数变概率，交叉熵负责量「这个概率离正确答案有多远」。

运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "00-数学基础/18-交叉熵与对数.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 【1】从「信息量」开始：越不可能发生的事，信息量越大
# ============================================================
# 3Blue1Brown 的故事：收到一条消息，告诉你「发生概率为 p 的事件发生了」，
# 它带来的「信息量」定义为  I = -log2(p)。
# 概率越小 → log 后越大 → 信息量越大（越让人惊讶）。
def information(p):
    """事件的信息量（以 2 为底，单位 bit）。"""
    return -np.log2(p)

print("【1】信息量 I = -log2(p)：越罕见越「惊讶」")
for p in [1.0, 0.5, 0.1, 0.01]:
    print(f"    p={p:<5} → 信息量 = {information(p):.3f} bit")
print("    p=1（必然发生）→ 0 bit（没有新信息）；p 越小，信息量越大\n")

# ============================================================
# 【2】信息熵 H(p)：一个分布里「平均」的信息量 = 不确定性
# ============================================================
def entropy(p):
    """信息熵 H(p) = -Σ p_i * log2(p_i)。
    p 是一组概率（和为 1）。返回平均不确定性，单位 bit。
    约定 0 * log(0) = 0（不可能的事件不贡献信息）。
    """
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                  # 去掉 0，避免 log(0)
    return -np.sum(p * np.log2(p))

print("【2】信息熵 H(p) = 不确定性（越平均越乱，熵越大）")
print(f"    抛硬币 [0.5, 0.5]   → H = {entropy([0.5, 0.5]):.3f} bit（最不确定）")
print(f"    偏心硬币 [0.9, 0.1] → H = {entropy([0.9, 0.1]):.3f} bit（比较确定）")
print(f"    必然事件 [1.0, 0.0] → H = {entropy([1.0, 0.0]):.3f} bit（完全确定）\n")

# ============================================================
# 【3】交叉熵 H(p, q)：用「以为的分布 q」去描述「真实分布 p」的成本
# ============================================================
def cross_entropy(p, q):
    """交叉熵 H(p,q) = -Σ p_i * log(q_i)。
    p 是真实分布，q 是预测分布。单位 bit（以 2 为底）。
    当 q 接近 p 时交叉熵小，当 q 在真实事件上给小概率时交叉熵大。
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0                  # 只在 p_i>0 的项上算（p_i=0 不贡献）
    return -np.sum(p[mask] * np.log2(q[mask]))

# 例子：真实是「猫」，预测分别给「猫」高/低概率
p_true = np.array([1.0, 0.0, 0.0])   # one-hot：正确答案是第 0 类（猫）
q_good = np.array([0.9, 0.05, 0.05]) # 预测得不错
q_bad  = np.array([0.1, 0.8, 0.1])   # 预测错了

print("【3】交叉熵 H(p,q)：真实=「猫」")
print(f"    预测 q_good={q_good} → H = {cross_entropy(p_true, q_good):.4f} bit（小，惩罚轻）")
print(f"    预测 q_bad ={q_bad}  → H = {cross_entropy(p_true, q_bad):.4f} bit（大，惩罚重）\n")

# ============================================================
# 【4】KL 散度：两个分布的差异 = 交叉熵 - 真实熵
# ============================================================
def kl_divergence(p, q):
    """KL(p‖q) = Σ p_i * log(p_i / q_i) = H(p,q) - H(p)。
    衡量「分布 q 偏离真实分布 p 的程度」。永远 ≥ 0，等于 0 当且仅当 p==q。
    （注意：KL 不对称，KL(p‖q) ≠ KL(q‖p)。）
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

print("【4】KL 散度 = H(p,q) - H(p) ≥ 0（衡量两分布差异）")
print(f"    KL(p‖q_good) = {kl_divergence(p_true, q_good):.4f}")
print(f"    KL(p‖q_bad)  = {kl_divergence(p_true, q_bad):.4f}")
print("    验证 H(p,q)=H(p)+KL：当 p 是 one-hot 时 H(p)=0，所以【交叉熵 == KL】")
print(f"    交叉熵 H(p_true,q_good)={cross_entropy(p_true, q_good):.4f}"
      f"  vs  KL={kl_divergence(p_true, q_good):.4f}  ✓ 相等\n")

# ============================================================
# 【5】关键：为什么用 -log(预测概率) 作为损失
# ============================================================
print("【5】分类损失 = -log(模型对正确类给的预测概率)")
# 当真实分布是 one-hot（只在正确类 c 上为 1）时：
#   交叉熵 = -Σ p_i log q_i = -1 * log(q_c) = -log(q_c)
# 即「只看模型给正确答案的那个概率」。
p_c = 0.9   # 模型给正确答案的概率
loss = -np.log(p_c)
print(f"    模型给正确答案的概率 p_c = {p_c} → 损失 -ln({p_c}) = {loss:.4f}")
print("    p_c 越大（越自信且正确）→ 损失越小；p_c → 0 → 损失 → ∞（错得离谱被重罚）\n")

# ============================================================
# 【6】迷你示例：用交叉熵训练一个「猜词」直觉
# ============================================================
print("【6】语言模型下一词预测的损失（交叉熵）")
vocab = ['今天', '天气', '真好', '糟糕']
# 真实下一词是「真好」（one-hot 在索引 2）
p_true4 = np.array([0., 0., 1., 0.])
preds = {
    '好模型':   np.array([0.05, 0.05, 0.85, 0.05]),
    '一般模型': np.array([0.25, 0.25, 0.30, 0.20]),
    '坏模型':   np.array([0.40, 0.30, 0.05, 0.25]),
}
for name, q in preds.items():
    ce = cross_entropy(p_true4, q)
    # 损失通常用自然对数 ln（不是 log2），换算：ln = log2 / log2(e)
    loss_ln = ce / np.log2(np.e)
    print(f"    {name}：预测「真好」={q[2]:.2f} → 交叉熵={ce:.3f} bit "
          f"(≈ 损失 {loss_ln:.3f})")
print("    → 训练目标：让交叉熵（损失）越来越小，即让正确词的概率越来越大\n")

# ============================================================
# 【7】可视化 1：-log(p) 曲线 —— 预测概率 vs 损失
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# 左图：-log(p) 曲线
p = np.linspace(0.001, 1.0, 400)
axes[0].plot(p, -np.log(p), color='#C44E52', lw=2.5)
axes[0].fill_between(p, -np.log(p), alpha=0.12, color='#C44E52')
axes[0].axhline(0, color='gray', lw=0.8)
axes[0].set_xlabel('模型给正确答案的预测概率 p')
axes[0].set_ylabel('损失 -ln(p)')
axes[0].set_title('损失曲线：预测越准(p→1)损失→0，越错(p→0)损失→∞',
                  fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)
# 标几个点
for pc, off in [(0.1, (0.18, -0.6)), (0.5, (-0.32, -0.5)), (0.9, (0.05, 0.8))]:
    axes[0].scatter([pc], [-np.log(pc)], color='black', zorder=5)
    axes[0].annotate(f'p={pc}\nloss={-np.log(pc):.2f}',
                     xy=(pc, -np.log(pc)), xytext=(pc + off[0], -np.log(pc) + off[1]),
                     fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='black'))

# 右图：两个预测分布对比 + 各自损失
labels = ['猫', '狗', '鸟']
x = np.arange(len(labels))
width = 0.35
q_good_show = np.array([0.9, 0.05, 0.05])
q_bad_show = np.array([0.1, 0.8, 0.1])
bars1 = axes[1].bar(x - width/2, q_good_show, width, color='#55A467',
                    label=f'好预测 (损失={-np.log(0.9):.2f})', edgecolor='black')
bars2 = axes[1].bar(x + width/2, q_bad_show, width, color='#C44E52',
                    label=f'坏预测 (损失={-np.log(0.1):.2f})', edgecolor='black')
# 标出正确答案
axes[1].bar([x[0] - width/2], [q_good_show[0]], width, color='#55A467',
            edgecolor='black', linewidth=2.2)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel('预测概率')
axes[1].set_title('真实答案=「猫」：好预测损失小，坏预测损失大',
                  fontsize=11, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "18-交叉熵与对数.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=110, bbox_inches='tight')
plt.close()
print(f"【7】图1 已保存：{out1}")

# ============================================================
# 【8】可视化 2：熵 vs 分布形状 —— 越平均熵越高
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
dists = {
    '确定(一家独大)': np.array([1.0, 0.0, 0.0, 0.0]),
    '偏一点':         np.array([0.7, 0.2, 0.1, 0.0]),
    '完全平均':       np.array([0.25, 0.25, 0.25, 0.25]),
}
for ax, (name, d) in zip(axes, dists.items()):
    H = entropy(d)
    ax.bar(['A', 'B', 'C', 'D'], d, color='#4C72B0', edgecolor='black', alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{name}\n熵 H = {H:.3f} bit', fontsize=11, fontweight='bold')
    ax.set_ylabel('概率')
    ax.grid(True, axis='y', alpha=0.3)
plt.suptitle('信息熵：分布越「平均/乱」，熵越大（越不确定）',
             fontsize=12, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "18-信息熵.png")
plt.savefig(out2, dpi=110, bbox_inches='tight')
plt.close()
print(f"【8】图2 已保存：{out2}\n")

# ============================================================
# 【9】连回 AI：交叉熵 = 语言模型的训练损失
# ============================================================
print("=" * 62)
print("✅ 第18课要点")
print("=" * 62)
print("  • 信息量 I = -log p：罕见事件带来的「惊讶」大")
print("  • 信息熵 H(p) = -Σ p·log p：分布的平均不确定性（越平均越高）")
print("  • 交叉熵 H(p,q) = -Σ p·log q：用 q 编码 p 的成本")
print("  • KL 散度 = H(p,q) - H(p) ≥ 0：两分布差异，等号当 p==q")
print("  • 分类里 p 是 one-hot → H(p)=0 → 【交叉熵 == KL】")
print("  • 损失 = -log(模型给正确答案的概率)：预测越准损失越小，越错越被重罚")
print()
print("🎯 AI 里的应用：")
print("  • 分类模型：softmax 出概率 → 交叉熵当损失（PyTorch 的 CrossEntropyLoss）")
print("  • 语言模型训练：最小化「下一个词」的负对数似然 = 交叉熵损失")
print("  • 困惑度(perplexity) = e^(交叉熵)：交叉熵的指数版，越低越好")
print("  • 上一课 softmax + 本课交叉熵 = 完整的分类 / 语言模型损失")
print("=" * 62)
