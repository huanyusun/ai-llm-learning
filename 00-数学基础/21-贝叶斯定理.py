"""
21 · 贝叶斯定理 — 从先验到后验
==============================================
面试考点：贝叶斯公式、先验/后验/似然的含义、朴素贝叶斯分类器
运行：uv run python "00-数学基础/21-贝叶斯定理.py"
"""

import numpy as np

# ── 1. 贝叶斯公式 ──────────────────────────────────────────
# P(A|B) = P(B|A) * P(A) / P(B)
# 后验 = 似然 × 先验 / 证据

print("=" * 60)
print("1. 贝叶斯公式：医学检测经典例题")
print("=" * 60)

# 场景：某疾病患病率 1%，检测灵敏度 99%，假阳性率 5%
# 问：检测阳性时，真正患病的概率？

p_disease = 0.01          # 先验：患病率
p_healthy = 1 - p_disease
p_positive_given_disease = 0.99   # 似然：患病时检测阳性
p_positive_given_healthy = 0.05   # 假阳性率

# 全概率公式求 P(阳性)
p_positive = p_positive_given_disease * p_disease + p_positive_given_healthy * p_healthy

# 贝叶斯公式求后验
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"患病率（先验）: {p_disease:.2%}")
print(f"检测灵敏度:     {p_positive_given_disease:.2%}")
print(f"假阳性率:       {p_positive_given_healthy:.2%}")
print(f"P(阳性):        {p_positive:.4f}")
print(f"P(患病|阳性):   {p_disease_given_positive:.2%}")
print()
print("→ 反直觉！检测阳性后，真正患病概率只有 ~16.7%")
print("→ 原因：先验太低（1%），假阳性的绝对数量远超真阳性")

# ── 2. 贝叶斯更新（连续观测） ──────────────────────────────
print("\n" + "=" * 60)
print("2. 贝叶斯更新：多次检测后概率如何变化")
print("=" * 60)

prior = p_disease
for i in range(1, 4):
    # 每次检测阳性后更新后验
    posterior = (p_positive_given_disease * prior) / \
               (p_positive_given_disease * prior + p_positive_given_healthy * (1 - prior))
    print(f"第 {i} 次阳性后: P(患病) = {posterior:.2%}")
    prior = posterior  # 后验变成下一次的先验

print("\n→ 多次阳性后，后验概率快速上升——这就是贝叶斯更新的力量")

# ── 3. 朴素贝叶斯分类器（手写） ──────────────────────────────
print("\n" + "=" * 60)
print("3. 朴素贝叶斯分类器：垃圾邮件检测")
print("=" * 60)

# 训练数据（简化）
# 特征：是否包含 "免费"、"中奖"、"会议"
# 标签：spam / ham
train_data = [
    ({"免费": 1, "中奖": 1, "会议": 0}, "spam"),
    ({"免费": 1, "中奖": 0, "会议": 0}, "spam"),
    ({"免费": 0, "中奖": 1, "会议": 0}, "spam"),
    ({"免费": 0, "中奖": 0, "会议": 1}, "ham"),
    ({"免费": 0, "中奖": 0, "会议": 1}, "ham"),
    ({"免费": 1, "中奖": 0, "会议": 1}, "ham"),
]

# 统计先验和条件概率（加 Laplace 平滑）
classes = ["spam", "ham"]
class_count = {c: sum(1 for _, l in train_data if l == c) for c in classes}
total = len(train_data)
words = ["免费", "中奖", "会议"]

print(f"训练样本: {total} 条 (spam={class_count['spam']}, ham={class_count['ham']})")

# P(word=1|class) with Laplace smoothing
cond_prob = {}
for c in classes:
    class_samples = [x for x, l in train_data if l == c]
    n = len(class_samples)
    for w in words:
        count = sum(1 for x in class_samples if x[w] == 1)
        cond_prob[(w, c)] = (count + 1) / (n + 2)  # Laplace

# 预测新邮件
test = {"免费": 1, "中奖": 1, "会议": 0}
print(f"\n测试邮件特征: {test}")

for c in classes:
    log_prob = np.log(class_count[c] / total)  # log 先验
    for w in words:
        if test[w] == 1:
            log_prob += np.log(cond_prob[(w, c)])
        else:
            log_prob += np.log(1 - cond_prob[(w, c)])
    print(f"  log P({c}|特征) ∝ {log_prob:.4f}")

print("\n→ spam 的 log 概率更高 → 分类为垃圾邮件 ✓")

# ── 4. 贝叶斯 vs 频率学派 ──────────────────────────────────
print("\n" + "=" * 60)
print("4. 贝叶斯 vs 频率学派（面试常问）")
print("=" * 60)
print("""
| 维度       | 频率学派              | 贝叶斯学派              |
|-----------|---------------------|----------------------|
| 参数观     | 固定未知常数           | 随机变量（有分布）        |
| 核心工具   | 最大似然估计 (MLE)     | 后验分布 (MAP/全贝叶斯)   |
| 先验信息   | 不使用               | 显式编码为先验分布        |
| 小样本     | 容易过拟合            | 先验起正则化作用          |
| AI 中应用  | 大部分深度学习         | 贝叶斯优化、不确定性估计   |

面试关键点：
- 贝叶斯公式本身不分学派，两派都用
- 贝叶斯学派的特点是"把参数也当随机变量"
- LLM 中的 RLHF 奖励模型训练用到了贝叶斯思想（偏好建模）
""")
