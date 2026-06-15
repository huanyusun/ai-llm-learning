"""
Prompt 工程 第3课：结构化输出（JSON / Pydantic 约束）
======================================================
理论来源：
- OpenAI Prompt Engineering Guide 策略一：Write clear instructions
  （用分隔符隔离指令与数据、指定输出格式、给 schema 示例）。
- 结构化输出三档手段（由弱到强）：
  1) Prompt 层约束：明确要求 JSON + 给字段 + few-shot 示例 + 分隔符；
  2) API/解码层约束：Function Calling / JSON Mode / response_format；
  3) 文法约束解码：outlines / llama.cpp GBNF，做到 100% 合规。
- 实战经验：模型较小时纯靠 prompt 难稳定，优先上 Schema/Function Calling；
  长输出建议「分片生成 + 校验 + 重试」，并对关键字段做后处理校验。

本文件包含三部分，全部【纯 Python 可直接跑】：
  A) Prompt 层：设计「信息抽取 → JSON」的 prompt 模板；
  B) 解析与校验：手写一个不依赖第三方库的「schema 校验器」，
     演示「解析失败 → 重试」的工程套路；
  C) Pydantic 版：当装了 pydantic 时，用 BaseModel 做更强约束；
     没装则自动跳过并给出 `uv add pydantic` 提示。

运行：uv run python "03-Prompt工程/03-结构化输出.py"
"""
from __future__ import annotations

import json
import re
from typing import Any


# ============================================================
# 【A】Prompt 层：信息抽取 → JSON 的 prompt 设计
# ============================================================
# 设计要点（对应 OpenAI 六策略之「Write clear instructions」）：
#   1. 明确角色与任务（你是信息抽取器）；
#   2. 明确字段 schema（name/age/role/skills）；
#   3. 用分隔符 <data> 隔离用户数据，防指令/数据混淆（也防注入）；
#   4. 给 1 个 few-shot 示例，固化格式；
#   5. 明令「只输出 JSON，不要解释」，降低格式破坏概率；
#   6. 要求「先思考再输出」可进一步降错（这里演示基础版）。

EXTRACT_SYSTEM_PROMPT = """你是简历信息抽取器。从 <data> 标签内的简历文本中，
抽取下面 4 个字段，并输出为一个 JSON 对象（只输出 JSON，不要任何解释）：
- name: 姓名（字符串）
- age: 年龄（整数）
- role: 应聘岗位（字符串）
- skills: 技能列表（字符串数组）

示例输入：
<data>张三，28 岁，应聘后端工程师，熟悉 Python、Go、MySQL。</data>
示例输出：
{"name": "张三", "age": 28, "role": "后端工程师", "skills": ["Python", "Go", "MySQL"]}

现在请抽取下面这份简历：
<data>{resume}</data>
"""


def build_prompt(resume: str) -> str:
    """拼出完整 prompt（用户数据被 <data> 包裹，防注入）。"""
    return EXTRACT_SYSTEM_PROMPT.replace("{resume}", resume)


# ============================================================
# 【B】解析与校验：手写 schema 校验 + 失败重试
# ============================================================
# 工程现实：模型输出常带「```json 代码块包裹」「前后多余文字」「缺字段」。
# 一个稳健的解析管线要：①抠出 JSON 片段 → ②解析 → ③按 schema 校验 →
# ④失败时重试（把错误信息回灌给模型）。
SCHEMA = {
    "name": str,     # 必须是字符串
    "age": int,      # 必须是整数
    "role": str,     # 必须是字符串
    "skills": list,  # 必须是列表
}


def extract_json(text: str) -> str:
    """从模型输出里抠出 JSON 文本（容忍 ```json 代码块和前后噪声）。"""
    text = text.strip()
    # 先试 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # 再退而求其次：第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text  # 让下面的 json.loads 去抛错


def validate(obj: dict[str, Any], schema: dict[str, type]) -> list[str]:
    """按 schema 校验对象，返回错误信息列表（空列表 = 通过）。"""
    errors: list[str] = []
    for field, typ in schema.items():
        if field not in obj:
            errors.append(f"缺字段：{field}")
            continue
        if not isinstance(obj[field], typ):
            errors.append(f"字段 {field} 应为 {typ.__name__}，实际 {type(obj[field]).__name__}")
    return errors


def parse_and_validate(raw: str) -> tuple[dict | None, list[str]]:
    """完整管线：抠 JSON → 解析 → 校验。返回 (对象或None, 错误列表)。"""
    try:
        obj = json.loads(extract_json(raw))
    except json.JSONDecodeError as e:
        return None, [f"JSON 解析失败：{e}"]
    if not isinstance(obj, dict):
        return None, ["解析结果不是 JSON 对象"]
    return obj, validate(obj, SCHEMA)


def retry_with_feedback(raw_outputs: list[str]) -> dict | None:
    """
    模拟「解析失败 → 把错误回灌 → 重试」的工程套路。
    这里把一组「模型多次输出」依次尝试，第一个通过校验的就采用；
    真实系统里是把上一次错误拼进 prompt 再调一次模型。
    """
    for i, raw in enumerate(raw_outputs, 1):
        obj, errs = parse_and_validate(raw)
        if obj is not None and not errs:
            print(f"  第 {i} 次输出通过校验 ✅")
            return obj
        print(f"  第 {i} 次输出校验失败：{errs}")
    print("  全部输出都未通过校验 ❌（需要更稳的约束，见 Pydantic/JSON Schema）")
    return None


# ============================================================
# 【C】Pydantic 版：装了 pydantic 就用更强的类型约束
# ============================================================
def demo_pydantic() -> None:
    print("=" * 72)
    print("【C】Pydantic 约束（装了才跑；没装给提示）")
    print("=" * 72)
    try:
        from pydantic import BaseModel, Field, ValidationError
    except ImportError:
        print("  本机未装 pydantic，跳过此演示。")
        print("  安装：uv add pydantic")
        print("  价值：用类型注解 + Field 约束声明 schema，解析失败自动抛")
        print("        ValidationError 并给出逐字段错误，比手写校验更省心、")
        print("        比 prompt 约束更硬（类型/必填/范围在 Python 侧强制）。")
        print()
        return

    class Resume(BaseModel):
        name: str = Field(..., min_length=1, description="姓名")
        age: int = Field(..., ge=0, le=150, description="年龄")
        role: str = Field(..., description="应聘岗位")
        skills: list[str] = Field(default_factory=list, description="技能列表")

    # 模拟模型输出（合法）
    raw_ok = '{"name": "李四", "age": 30, "role": "算法工程师", "skills": ["PyTorch", "CUDA"]}'
    obj = Resume.model_validate_json(raw_ok)
    print(f"  合法输出解析成功：{obj.model_dump()}")

    # 模拟模型输出（非法：age 是字符串、skills 缺失）
    raw_bad = '{"name": "王五", "age": "三十", "role": "前端工程师"}'
    try:
        Resume.model_validate_json(raw_bad)
    except ValidationError as e:
        print(f"  非法输出被 Pydantic 拦截，逐字段错误：")
        for line in str(e).splitlines():
            print(f"    {line}")
    print("  → 生产里配合 response_format=json_schema / Function Calling，")
    print("    再用 Pydantic 在 Python 侧兜底校验，格式可靠性最高。\n")


# ============================================================
# 【主演示】把 A/B 串起来：模拟模型的 3 次输出（越后越规范）
# ============================================================
def demo_prompt_and_parsing() -> None:
    print("=" * 72)
    print("【A+B】Prompt 设计 + 解析校验 + 失败重试（纯 Python 可跑）")
    print("=" * 72)

    resume = "赵六，26 岁，应聘数据分析师，擅长 SQL、Python、Tableau。"
    prompt = build_prompt(resume)
    print("【prompt 预览】（用户数据被 <data> 包裹，防注入）")
    print(prompt)
    print("-" * 72)

    # 模拟模型 3 次输出，质量递增（演示「格式不规整 → 规整」的真实分布）
    simulated_outputs = [
        # 第1次：带 markdown 代码块 + 前后闲话（常见噪声）
        '好的，这是抽取结果：\n```json\n{"name": "赵六", "age": 26, '
        '"role": "数据分析师", "skills": ["SQL", "Python", "Tableau"]}\n```\n希望有帮助！',
        # 第2次：缺字段（skills 漏了）
        '{"name": "赵六", "age": "26", "role": "数据分析师"}',
        # 第3次：干净合法
        '{"name": "赵六", "age": 26, "role": "数据分析师", '
        '"skills": ["SQL", "Python", "Tableau"]}',
    ]

    print("模拟模型 3 次输出，依次尝试解析+校验：")
    result = retry_with_feedback(simulated_outputs)
    print("-" * 72)
    if result is not None:
        print(f"最终结构化结果：{json.dumps(result, ensure_ascii=False)}")
    print()


# ============================================================
# 【附】各约束手段对照（连回面试）
# ============================================================
def print_constraint_ladder() -> None:
    print("=" * 72)
    print("【附】结构化输出约束手段（由弱到强，面试常考）")
    print("=" * 72)
    print("  1) Prompt 层  ：要求 JSON + 字段 + few-shot + 分隔符（本课 A）")
    print("     优点：零成本、跨厂商；缺点：模型小/输出长时不稳定。")
    print("  2) API 层    ：Function Calling / JSON Mode / response_format")
    print("     优点：解码时强制合法 JSON；缺点：各厂商 API 细节不同。")
    print("  3) 文法层    ：outlines / llama.cpp GBNF，GBNF 文法约束 token 采样")
    print("     优点：可做到 100% 合规；缺点：仅本地推理可用，配置较重。")
    print("  工程套路：解析失败 → 把错误回灌 prompt → 重试（本课 B 演示）；")
    print("           关键字段务必在 Python 侧二次校验，别盲信模型。")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    demo_prompt_and_parsing()
    demo_pydantic()
    print_constraint_ladder()
    print("=" * 72)
    print("面试连接：")
    print("  • 如何让模型稳定输出 JSON？→ prompt 明确字段+示例+分隔符，")
    print("    不够就上 Function Calling / response_format=json_schema。")
    print("  • 解析失败怎么办？→ 抠 JSON 片段 → 解析 → schema 校验 → 失败重试。")
    print("  • 为什么用分隔符？→ 隔离指令与数据，同时是防 Prompt Injection 手段。")
    print("  • Pydantic 的价值？→ 用类型注解声明 schema，解析失败自动报错，")
    print("    比 prompt 硬、比手写校验省心。")
    print("=" * 72)
