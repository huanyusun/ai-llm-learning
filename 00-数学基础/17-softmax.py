"""
数学基础 第17课：Softmax（从零开始）
=====================================
参考：《动手学深度学习》https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html
      3Blue1Brown《信息论》《神经网络》系列

问题：模型最后吐出一堆「分数」(logits)，比如 [3.2, 1.1, -0.5]，
      它们有正有负、和也不固定，没法当「概率」用。
Softmax 的任务：把任意一堆分数，变成一组【合法的概率】——
      ① 每个数都 > 0； ② 所有数加起来正好等于 1。

公式（一行话记牢）：
        softmax(x)_i = exp(x_i) / Σ exp(x_j)
先对每个分数取 e 的指数（变正数、放大差距），再除以总和（归一化到 1）。

温度参数 T（temperature）：把分数先除以 T 再 softmax
      • T 大（高温）→ 分数差距被「压平」→ 概率分布平滑、更随机
      • T 小（低温）→ 分数差距被「拉大」→ 概率尖锐、一家独大
LLM 采样时的 temperature 就是这个 T。

【本课重点】softmax 正是 attention 的第 4 步：
      把「当前词对每个词的关注分数」变成「注意力权重(概率)」。

运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "00-数学基础/17-softmax.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 【1】用 numpy 从零实现 softmax
# ============================================================
def softmax(x):
    """把一组分数 x 变成概率（和为 1）。

    实现细节：先减去最大值 max(x)，叫「数值稳定技巧」。
    原因：e^(很大的数) 会溢出变成 inf。减去同一个常数不改变 softmax 结果
    （因为分子分母同时乘了 e^(-max)），但让最大的指数变成 e^0=1，避免爆炸。
    这是《动手学深度学习》里反复强调的写法。
    """
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)          # 数值稳定：减最大值
    exp_x = np.exp(x)          # 取 e 的指数，把分数变正、放大差距
    return exp_x / np.sum(exp_x)  # 归一化：除以总和 → 和为 1


def softmax_with_temperature(x, T=1.0):
    """带温度 T 的 softmax：先把分数除以 T。"""
    return softmax(x / T)


# 演示一组分数 → 概率
logits = np.array([3.2, 1.1, -0.5, 0.4])
probs = softmax(logits)
print("【1】softmax 把分数变概率")
print(f"    原始分数 logits = {logits}")
print(f"    softmax 后概率  = {np.round(probs, 4)}")
print(f"    验证：概率之和 = {probs.sum():.6f}  （正好等于 1 ✅）")
print(f"    分数最大的 3.2 → 拿到最大概率 {probs.max():.4f}\n")

# ============================================================
# 【2】拆开看三步：取指数 → 求和 → 归一化
# ============================================================
print("【2】softmax 的三步拆解")
print(f"    第1步 取指数 exp({logits - logits.max()})")
print(f"          → {np.round(np.exp(logits - logits.max()), 4)}  （全部变正、差距被放大）")
print(f"    第2步 求和 = {np.exp(logits - logits.max()).sum():.4f}")
print(f"    第3步 归一化 = 每个指数 ÷ 总和 → 上面那组概率\n")

# ============================================================
# 【3】温度参数：高温平滑 / 低温尖锐
# ============================================================
print("【3】温度 T 怎么影响分布（同一组分数 [3.2, 1.1, -0.5, 0.4]）")
for T in [0.5, 1.0, 2.0]:
    p = softmax_with_temperature(logits, T)
    print(f"    T={T:>3}: 概率 = {np.round(p, 4)}  最大概率={p.max():.4f}")
print("    → T 小(低温)：最大的那个越来越突出（一家独大）")
print("    → T 大(高温)：差距被压平，更接近均匀分布\n")

# ============================================================
# 【4】大数 softmax 会「一家独大」—— 巨大的数几乎独占全部概率
# ============================================================
big = np.array([10.0, 0.0, 0.0])      # 第一个分数远超其他
p_big = softmax(big)
print("【4】当某个分数远大于其他：softmax 几乎独占")
print(f"    logits = {big}  → 概率 = {np.round(p_big, 6)}")
print("    这就是低温下 LLM 表现「确定/保守」、高温下「发散/有创造力」的数学根源\n")

# ============================================================
# 【5】可视化 1：不同温度下的概率分布（柱状图）
# ============================================================
temps = [0.5, 1.0, 2.0, 5.0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
labels = ['猫', '狗', '鸟', '鱼']
colors = ['#4C72B0', '#DD8452', '#55A467', '#C44E52']

for ax, T in zip(axes, temps):
    p = softmax_with_temperature(logits, T)
    bars = ax.bar(labels, p, color=colors, edgecolor='black', alpha=0.85)
    for b, v in zip(bars, p):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}',
                ha='center', fontsize=9)
    ax.set_title(f'T = {T}（{"低温·尖锐" if T <= 1 else "高温·平滑"}）', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('概率' if T == temps[0] else '')
    ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('Softmax 温度参数：低温 → 一家独大，高温 → 趋于平均',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "17-softmax温度.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=110, bbox_inches='tight')
plt.close()
print(f"【5】图1 已保存：{out1}")

# ============================================================
# 【6】可视化 2：分数不断拉大时，最大概率如何逼近 1（一家独大）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
gaps = np.linspace(0, 15, 200)        # 第一个分数比第二个高出多少
max_probs = []
for g in gaps:
    p = softmax([g, 0.0, 0.0])        # 三个分数：[g, 0, 0]
    max_probs.append(p[0])

ax.plot(gaps, max_probs, color='#C44E52', lw=2.5)
ax.fill_between(gaps, max_probs, alpha=0.15, color='#C44E52')
ax.axhline(1.0, color='gray', ls='--', lw=1)
ax.set_xlabel('最大分数与其它分数的差距')
ax.set_ylabel('最大那项分到的概率')
ax.set_title('分数差距越大 → softmax 越接近「一家独大」', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.annotate('差距≈10 时已几乎=1\n(这就是低温采样的样子)',
            xy=(10, max_probs[133]), xytext=(4, 0.55),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'))
plt.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "17-softmax一家独大.png")
plt.savefig(out2, dpi=110, bbox_inches='tight')
plt.close()
print(f"【6】图2 已保存：{out2}\n")

# ============================================================
# 【7】连回 AI：softmax 是 attention 的第 4 步
# ============================================================
print("=" * 60)
print("✅ 第17课要点")
print("=" * 60)
print("  • softmax(x)_i = exp(x_i) / Σ exp(x_j)：分数 → 合法概率")
print("  • 三个性质：全为正、和为 1、保留大小顺序（大的还是大）")
print("  • 减最大值再做：数值稳定，防止 e^大数 溢出")
print("  • 温度 T：除以 T。T 小→尖锐，T 大→平滑（即 LLM 的 temperature）")
print("  • 分数差距很大时 → 一家独大（最大项概率≈1）")
print()
print("🎯 AI 里的应用：")
print("  • 分类模型最后一层：softmax 把 logits 变成「各类概率」")
print("  • 语言模型：softmax 把隐状态变成「词表里每个词的概率」")
print("  • 【attention 第 4 步】：Q·K 得到的「关注分数」经 softmax")
print("    变成「注意力权重」，决定每个词该分配多少注意力（和为 1）")
print("  • 这就是为什么 softmax 是 Transformer 的核心部件之一")
print("=" * 60)
