"""
数学基础 第8课：函数与极限 Functions & Limits（从零开始）
==========================================================
参考：3Blue1Brown《微积分的本质》、
      《动手学深度学习》https://zh.d2l.ai

函数和极限是微积分的「地基」，而微积分是神经网络训练（梯度下降）的数学语言。

本课讲三件事：
  1. 函数 = 「输入 → 输出」的映射机器（喂一个数，吐一个数）
  2. 极限 = 无限趋近某个值时，函数值「奔向」哪里（导数的灵魂）
  3. 连续 = 函数图像一笔画到底，没有「跳变」

用 matplotlib 把这些函数和直觉画出来，让你看见它。
运行：uv run python "00-数学基础/08-函数与极限.py"
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# macOS 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)


# ============================================================
# 【1】函数：一台「输入 → 输出」的机器
# ============================================================
# 一个函数 f 就是一条规则：给我一个 x，我就还你一个 y = f(x)。
# 例如 f(x) = x^2，输入 3，输出 9；输入 -2，输出 4。
def f_square(x):
    """二次函数 f(x) = x^2"""
    return x ** 2


print("【1】函数 = 输入 → 输出的映射机器")
print(f"    f(x) = x²，f(3) = {f_square(3)}，f(-2) = {f_square(-2)}")
print("    神经网络本身就是一个超大的函数：输入图片像素 → 输出「是猫的概率」\n")


# ============================================================
# 【2】常见函数大集合：认识它们的「长相」
# ============================================================
print("【2】AI 里最常见的几种函数：")
print("    • 线性函数  y = w·x + b     → 线性回归、单层神经元的核心")
print("    • 二次函数  y = x²          → 理解「弯曲」、损失函数常出现")
print("    • 指数函数  y = eˣ          → 增长/衰减，Softmax 分子里有它")
print("    • 对数函数  y = ln(x)       → 交叉熵损失的核心\n")

x = np.linspace(-3, 3, 400)


# ============================================================
# 【3】极限：无限「趋近」某个点时，函数值去往何方
# ============================================================
# 极限不是「等于」，而是「无限靠近」。
# 问 lim(x→0) sin(x)/x = ?  在 x=0 处分母为0，函数无定义，
# 但当 x 越来越接近 0 时，比值越来越接近 1。
def sinc_ratio(x):
    """sin(x)/x，在 x=0 处无定义，但极限存在"""
    return np.where(x == 0, np.nan, np.sin(x) / x)


# 数值验证：取 x = 0.1, 0.01, 0.001 ... 看 sin(x)/x 的趋势
print("【3】极限的直觉：lim(x→0) sin(x)/x")
for h in [0.1, 0.01, 0.001, 0.0001]:
    print(f"    x = {h:.4f} 时  sin(x)/x = {np.sin(h) / h:.8f}")
print("    → x 越接近 0，比值越接近 1。所以极限 = 1（哪怕 x=0 处函数本身没定义）")
print("    这正是导数的定义需要的东西：「差值的极限」\n")


# ============================================================
# 【4】连续：图像一笔画到底，没有「断」也没有「跳」
# ============================================================
# 连续的口语定义：笔不离开纸就能把图像画完。
# 数学的定义：lim(x→a) f(x) = f(a)，即「极限值 = 函数值」。
def step_func(x):
    """阶跃函数：x<0 时取 0，x≥0 时取 1。在 x=0 处「跳变」= 不连续"""
    return np.where(x >= 0, 1.0, 0.0)


print("【4】连续 vs 不连续")
print("    • f(x)=x² 处处连续（图像一笔画完）")
print("    • 阶跃函数在 x=0 处「跳」了一下 = 不连续")
print("    • AI 角度：激活函数 ReLU 在 0 处「折一下」但连续可画，sigmoid 处处光滑连续\n")


# ============================================================
# 【5】画出来：建立直觉
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ---- 图① 四种常见函数同框 ----
ax = axes[0, 0]
ax.plot(x, 2 * x + 1, label="线性 y=2x+1", lw=2)
ax.plot(x, x ** 2, label="二次 y=x²", lw=2)
ax.plot(x, np.exp(x) / np.exp(3), label="指数 y=eˣ/e³ (缩放)", lw=2)  # 缩放方便同框看
xp = np.linspace(0.01, 3, 400)
ax.plot(xp, np.log(xp), label="对数 y=ln(x)", lw=2)
ax.set_title("① 四种常见函数（AI 的基本积木）", fontsize=12)
ax.set_ylim(-3, 6); ax.set_xlim(-3, 3)
ax.grid(True, alpha=0.3); ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9, loc='upper left')

# ---- 图② 极限 sin(x)/x：逼近 0 时趋向 1 ----
ax = axes[0, 1]
xs = np.linspace(-3, 3, 400)
ax.plot(xs, sinc_ratio(xs), label="y=sin(x)/x", lw=2, color='purple')
ax.plot(0, 1, 'o', color='red', ms=10, label="极限点 (0, 1)")  # 极限值用空心/实心点标记
ax.set_title("② 极限：x→0 时 sin(x)/x → 1（哪怕 0 处无定义）", fontsize=12)
ax.set_ylim(-0.5, 1.5); ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9)

# ---- 图③ 连续 vs 不连续 ----
ax = axes[1, 0]
xc = np.linspace(-2, 2, 400)
ax.plot(xc, xc ** 2, label="连续 y=x²（一笔画完）", lw=2.5, color='green')
# 阶跃函数分两段画，制造「断开」视觉效果
ax.plot(xc[xc < 0], step_func(xc[xc < 0]), lw=2.5, color='red', label="阶跃（x=0 跳变）")
ax.plot(xc[xc >= 0], step_func(xc[xc >= 0]), lw=2.5, color='red')
ax.plot(0, 0, 'o', color='red', ms=8, fillstyle='none', mew=2)   # 左极限：开圆
ax.plot(0, 1, 'o', color='red', ms=8)                            # 右值：实心圆
ax.set_title("③ 连续（绿）vs 不连续（红：x=0 跳变）", fontsize=12)
ax.set_ylim(-0.5, 2.2); ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9, loc='upper left')

# ---- 图④ 复合函数：函数套函数（为下一课链式法则铺垫）----
ax = axes[1, 1]
g = x ** 2                 # 内层 g(x) = x²
fg = np.exp(g) / np.exp(4) # 外层 f(g) = e^g，缩放后画出来
ax.plot(x, g, label="内层 g(x)=x²", lw=2, color='blue')
ax.plot(x, fg, label="复合 f(g(x))=e^(x²)/e⁴", lw=2, color='orange')
ax.set_title("④ 复合函数：函数套函数（链式法则的主角）", fontsize=12)
ax.set_ylim(-1, 6); ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.8); ax.axvline(0, color='gray', lw=0.8)
ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(IMG, "08-函数与极限.png")
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"【5】图已保存：{out}\n")


# ============================================================
# 【6】连回 AI：这些东西在神经网络里干嘛？
# ============================================================
print("=" * 55)
print("✅ 第8课要点")
print("=" * 55)
print("  • 函数 = 输入→输出的映射；神经网络就是一个超大函数")
print("  • 常见函数：线性/二次/指数/对数，是构建模型的积木")
print("  • 极限 = 无限趋近时的归宿，是「导数」定义的地基")
print("  • 连续 = 一笔画完；激活函数多选连续光滑的（如 sigmoid）")
print()
print("🎯 AI 里的应用：")
print("  • 线性 y=w·x+b → 一个神经元就是它（w 权重、b 偏置）")
print("  • 指数+对数 → Softmax + 交叉熵，分类任务的标配")
print("  • 极限思想 → 下一课「导数」就靠它定义")
print("=" * 55)
