"""
数学基础 第14课：指数 e 与对数 log（从零开始）
================================================
参考：3Blue1Brown《微积分的本质》《指数与对数》、概率论基础、《动手学深度学习》https://zh.d2l.ai

为什么 AI 课要讲 e 和 log？
- softmax 用 exp：把任意分数（可正可负）变成「概率」（全是正数、加起来=1）
- 交叉熵 / 负对数似然 用 log：衡量「预测概率」和「真实标签」差多远
- 对数有个神性质：把「乘法」变成「加法」，让数学和计算都更简单

本课用 matplotlib 把 eˣ 曲线和对数曲线画出来，让你「看见」它们。
运行：uv run --directory /Users/sunhuanyu/ai-llm-learning python "00-数学基础/14-指数e与对数.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体（开头模板，固定写法）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 【1】自然常数 e ≈ 2.71828... 它是「连续增长的天花板」
# ============================================================
# 直觉：你存 1 块钱，年利率 100%。
#   一年结算1次  → 1×(1+1)     = 2.0
#   一年结算2次  → 1×(1+0.5)^2 = 2.25
#   一年结算n次  → (1+1/n)^n   → 越切越细，最后收敛到 e ≈ 2.71828
# 也就是说：把钱无限频繁地复利，1 块钱最多长到 e 块（翻 2.718 倍）。
print("【1】自然常数 e")
print(f"    e ≈ {np.e:.6f}")
print("    直觉：1 块钱按 100% 年利无限频繁复利，最多长到 e 块")
print("    数学：(1 + 1/n)^n  当 n→∞ 时的极限\n")

n = np.array([1, 2, 10, 100, 1000, 10000])
seq = (1 + 1 / n) ** n
for ni, si in zip(n, seq):
    print(f"      n={ni:>5}  →  (1+1/n)^n = {si:.6f}")
print(f"      n→∞     →            e ≈ {np.e:.6f}\n")

# ============================================================
# 【2】指数函数 exp(x) = eˣ：永远为正、增长极快
# ============================================================
x = np.linspace(-3, 3, 200)
y_exp = np.exp(x)   # 即 e^x
print("【2】指数函数 exp(x) = eˣ")
print(f"    exp(0)  = {np.exp(0)}     （任何数的 0 次方都是 1）")
print(f"    exp(1)  = {np.exp(1):.4f}  （就是 e）")
print(f"    exp(2)  = {np.exp(2):.4f}  （e 的平方）")
print("    性质：永远 > 0（重要！softmax 就靠这点保证概率为正）")
print("    性质：exp(a+b) = exp(a)·exp(b)（指数把「加」变「乘」）\n")

# ============================================================
# 【3】对数 log = 指数的「反操作」（问：e 的几次方 = 这个数？）
# ============================================================
# 自然对数 ln(x) = log_e(x)：以 e 为底
# 在 numpy/数学里，np.log 默认就是 ln（自然对数，以 e 为底）
print("【3】对数 log（默认 ln，以 e 为底）")
print(f"    ln(e)   = {np.log(np.e):.4f}     （e 的 1 次方 = e，所以 ln(e)=1）")
print(f"    ln(1)   = {np.log(1):.4f}     （e 的 0 次方 = 1，所以 ln(1)=0）")
print(f"    ln(exp(2)) = {np.log(np.exp(2)):.4f}  （log 和 exp 互相抵消）")
print("    换底公式：log10(x) = ln(x) / ln(10)，其他底类似\n")

# ============================================================
# 【4】核心性质：对数把「乘法」变成「加法」
# ============================================================
a, b = 5.0, 7.0
print("【4】对数的神性质：log(乘法) = 加法")
print(f"    ln(a·b) = ln({a}×{b}) = ln({a*b}) ≈ {np.log(a * b):.4f}")
print(f"    ln(a)+ln(b) = {np.log(a):.4f} + {np.log(b):.4f} = {np.log(a) + np.log(b):.4f}")
print("    两者相等！  →  ln(a·b) = ln(a) + ln(b)")
print("    为什么有用？AI 里连乘很多个概率会下溢变成 0，")
print("              取 log 后变成连加，又稳定又好算（这就是对数似然）\n")

# ============================================================
# 【5】可视化：eˣ 曲线 和 对数曲线（互为反函数，关于 y=x 对称）
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# 图1：指数 eˣ
ax = axes[0]
ax.plot(x, y_exp, color='#d62728', lw=2.5, label=r'$e^x$（指数）')
ax.axhline(0, color='gray', lw=0.8)
ax.plot(0, 1, 'o', color='black')              # e^0 = 1
ax.plot(1, np.e, 'o', color='blue')            # e^1 = e
ax.annotate('  e^0=1', (0, 1), fontsize=11)
ax.annotate('  e^1≈2.718', (1, np.e), fontsize=11, color='blue')
ax.set_xlim(-3, 3); ax.set_ylim(-1, 12)
ax.set_title("① 指数 eˣ：永远>0，增长飞快", fontsize=12)
ax.grid(True, alpha=0.3); ax.legend(loc='upper left')

# 图2：对数 ln(x)（定义域 x>0）
ax = axes[1]
xpos = np.linspace(0.01, 8, 200)
ax.plot(xpos, np.log(xpos), color='#2ca02c', lw=2.5, label=r'$\ln(x)$（自然对数）')
ax.plot(1, 0, 'o', color='black')              # ln(1)=0
ax.plot(np.e, 1, 'o', color='blue')            # ln(e)=1
ax.annotate('  ln(1)=0', (1, 0), fontsize=11)
ax.annotate('  ln(e)=1', (np.e, 1), fontsize=11, color='blue')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlim(-1, 8); ax.set_ylim(-3, 3)
ax.set_title("② 对数 ln(x)：把乘变加，增长很慢", fontsize=12)
ax.grid(True, alpha=0.3); ax.legend(loc='lower right')

# 图3：eˣ 和 ln(x) 互为反函数，关于直线 y=x 对称
ax = axes[2]
xs = np.linspace(-2.2, 2.2, 200)
ax.plot(xs, np.exp(xs), color='#d62728', lw=2.5, label=r'$e^x$')
ax.plot(np.exp(xs), xs, color='#2ca02c', lw=2.5, label=r'$\ln(x)$')  # 反函数 = 对调 x,y
ax.plot([-2, 3], [-2, 3], '--', color='gray', lw=1.2, label='y=x（对称轴）')
ax.set_xlim(-2.5, 3); ax.set_ylim(-2.5, 3)
ax.set_title("③ eˣ 与 ln(x) 互为反函数（关于 y=x 对称）", fontsize=12)
ax.grid(True, alpha=0.3); ax.legend(loc='upper left')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "14-指数e与对数.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")

# ============================================================
# 【6】连回 AI：softmax 用 exp，交叉熵用 log
# ============================================================
print("【6】连回 AI")
logits = np.array([2.0, 1.0, 0.1])   # 模型输出的原始分数（可正可负，加起来不必=1）
probs = np.exp(logits) / np.sum(np.exp(logits))   # softmax
print(f"    原始分数 logits = {logits}")
print(f"    exp 后          = {np.exp(logits)}   （全变正数）")
print(f"    归一化(softmax) = {probs}   （全正、加起来=1 → 概率）")

# 交叉熵：假设正确答案是第 0 类，看模型给它的概率有多小
true_label = 0
p_true = probs[true_label]
cross_entropy = -np.log(p_true)   # 负对数似然
print(f"\n    正确类别概率 p = {p_true:.4f}")
print(f"    交叉熵 = -ln(p) = {cross_entropy:.4f}")
print("    p 越接近 1 → -ln(p) 越接近 0（预测对了，损失小）")
print("    p 越接近 0 → -ln(p) → +∞（预测错了，损失爆炸式变大）\n")

print("=" * 60)
print("✅ 第14课要点")
print("=" * 60)
print("  • e ≈ 2.71828：连续复利的极限，自然增长的「底」")
print("  • exp(x)=eˣ：永远为正、增长极快  → softmax 靠它造概率")
print("  • ln(x)=log_e(x)：指数的反操作；ln(1)=0, ln(e)=1")
print("  • 对数把乘变加：ln(a·b) = ln(a)+ln(b)  → 防止连乘下溢")
print()
print("🎯 AI 里的应用：")
print("  • softmax:  p_i = exp(z_i) / Σexp(z_j)   （exp 让全为正）")
print("  • 交叉熵:   L = -Σ y_i·ln(p_i)            （log 量度误差）")
print("  • 损失函数加 log 后更稳定、梯度更好算（对数似然）")
print("=" * 60)
