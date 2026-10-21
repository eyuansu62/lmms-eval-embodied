"""MomaGraph-Bench 的评测逻辑。

数据来源
--------
``cheryyunl/MomaGraph-Bench``（HF dataset，未 gated）。
3 个 parquet 共 1315 题，**图片以 PIL 对象内嵌在 parquet 里**，
不需要任何网络请求或额外媒体下载 —— 这是它和 CausalPhys 最大的不同。

字段
----
```
image            PIL 图片对象
question         题干（**不含**选项）
options          list[str]，选项自带 "A) ..." 前缀
correct_answer   "(C)" 这种带括号的字母
idx              题号
scene            场景标识，如 "lamp_collection/421267/lamp"
task_instruction 任务描述，如 "Power on the lamp."
type             题型，见下
```

题型（``type``，实测 14 个取值）
------------------------------
```
Spatial Reasoning     306     Object Selection      273
Action Reasoning      265     Affordance Reasoning  218
Effect Prediction     168     Sequence Planning      75
Dynamic Verification    3
```
另有 7 个只出现 1 次的变体值：
``Action Reasoning (overall plan)``、``Affordance Reasoning (power)``、
``Object Selection (cold items)``、``Action Reasoning (Scene 1 → Scene 2)``、
``Action Reasoning (Scene 1 → Scene 3)``、``Graph Inference``、
``Object Selection (Goal state planning)``。

**按 type 拆分成多个 task**（对应 能力体系（CAPABILITIES.md） 的不同 P0 条目）：

| task | 筛选 | 题数 | 对应 P0 |
|---|---|---|---|
| ``momagraph_effect_prediction`` | ``type == "Effect Prediction"`` | 168 | #18 #31 |
| ``momagraph_spatial_reasoning`` | ``type == "Spatial Reasoning"`` | 306 | #30 |
| ``momagraph_action_sequence`` | ``type`` 以 ``Action Reasoning`` 开头，或 ``== "Sequence Planning"`` | 343 | #29 |
| ``momagraph_bench`` | 全量 | 1315 | 参考基线 |

关于 #30 的 reachability 子筛：文档要求"再筛 question 涉及 reachability 的题"，
但**实测数据里没有这类题** —— 306 题中 301 条是
``"What is the spatial relationship between X and Y?"`` 模板；
全量检索 ``reach``/``reachab``/``arm``/``grasp``/``able to`` 等关键词，
题干命中 0 条、选项命中 2 条。所以只按 type 筛，不做子筛。

关于 #29 的映射：文档说"发布 type 与 Action Sequence Reasoning 的对应关系待核验；
不把 Action Reasoning / Object Selection 整体等同"。这里按已核验的口径取
``Action Reasoning``（含 3 个 ``(xxx)`` 变体）+ ``Sequence Planning``；
**不含** Object Selection。

**答案没有按序排列**（1315 题里答案切换 891 次），所以不需要像
CausalPhys 那样打散。但分布偏斜：全量 ``B`` 占 642/1315（49%），
各 type 内部也不同。评估时留意这一点。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

# 题型 -> 中文说明（供 yaml 注释与文档引用）
TYPE_LABELS: Final = {
    "Spatial Reasoning": "空间推理（P0 #5 相对位置 / #6 相对距离）",
    "Object Selection": "物体选择（P0 #10 目标理解）",
    "Action Reasoning": "动作推理（P0 #20 指令顺序理解）",
    "Affordance Reasoning": "可供性推理（P0 #2 客体可供性感知）",
    "Effect Prediction": "效果预测（P0 #22 约束满足检查 / #21 物理可行性）",
    "Sequence Planning": "序列规划（P0 #19 子任务分解）",
    "Dynamic Verification": "动态验证（P0 #24 失败检测）",
}

# Action Sequence（P0 #29）纳入的 type 前缀。用前缀匹配是为了带上
# "Action Reasoning (overall plan)" 这类变体；**不含** Object Selection。
_ACTION_SEQUENCE_PREFIX: Final = "Action Reasoning"
_ACTION_SEQUENCE_EXACT: Final = ("Sequence Planning",)


def _type_of(example: Mapping[str, Any]) -> str:
    return str(example.get("type") or "").strip()


def process_docs_effect_prediction(dataset):
    """筛出 ``type == "Effect Prediction"``（168 题）。对应 P0 #18 / #31。"""
    return dataset.filter(lambda ex: _type_of(ex) == "Effect Prediction")


def process_docs_spatial_reasoning(dataset):
    """筛出 ``type == "Spatial Reasoning"``（306 题）。对应 P0 #30。

    文档 #30 还要求"再筛 question 涉及 reachability 的题"，但**实测数据里
    没有这类题**：306 题中 301 条是 spatial relationship 模板，全量检索
    ``reach``/``arm``/``grasp``/``able to`` 等关键词，题干命中 0 条。
    故只按 type 筛。
    """
    return dataset.filter(lambda ex: _type_of(ex) == "Spatial Reasoning")


def process_docs_action_sequence(dataset):
    """筛出 Action Sequence 相关题（343 题）。对应 P0 #29。

    纳入 ``Action Reasoning`` 全部变体（含 ``(overall plan)`` 等）
    以及 ``Sequence Planning``；不含 ``Object Selection``。
    """
    def _keep(ex):
        t = _type_of(ex)
        return t.startswith(_ACTION_SEQUENCE_PREFIX) or t in _ACTION_SEQUENCE_EXACT
    return dataset.filter(_keep)


def doc_to_visual(doc: Mapping[str, Any]) -> list[Any]:
    """返回图片。

    图片是 parquet 里的 PIL 对象，直接返回即可（lmms-eval 的视觉输入
    既接受本地路径也接受 PIL Image）。
    """
    image = doc.get("image")
    return [image] if image is not None else []


def doc_to_text(doc: Mapping[str, Any], lmms_eval_specific_kwargs: Mapping[str, str] | None = None) -> str:
    """题干 + 选项 + 作答指令。

    与 CausalPhys 不同，这里的 ``question`` **不含**选项，
    选项在独立的 ``options`` 字段里，且自带 ``A)`` 前缀。
    """
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get(
        "post_prompt", "\nAnswer with the option's letter from the given choices directly."
    )

    question = str(doc.get("question") or "").strip()
    options = doc.get("options") or []
    options_text = "\n".join(str(o) for o in options)

    parts = [question]
    if options_text:
        parts.append(options_text)
    text = "\n".join(parts)
    return f"{pre_prompt}{text}{post_prompt}"


def doc_to_target(doc: Mapping[str, Any]) -> str:
    """正确答案的字母（去掉括号）。

    ``correct_answer`` 是 ``"(C)"`` 形式，这里取 ``"C"``。
    """
    return _normalize_answer(doc.get("correct_answer"))


def _normalize_answer(raw: Any) -> str:
    """把 ``"(C)"`` / ``"C"`` / ``"c)"`` 之类统一成 ``"C"``。"""
    s = str(raw or "").strip()
    m = re.search(r"([A-Za-z])", s)
    return m.group(1).upper() if m else ""


def _valid_choices(doc: Mapping[str, Any]) -> set[str]:
    """从 ``options`` 的 ``A)`` 前缀推断合法选项集合。"""
    found = set()
    for o in doc.get("options") or []:
        m = re.match(r"^\s*([A-Za-z])\s*[\)\.\:]", str(o))
        if m:
            found.add(m.group(1).upper())
    return found or {"A", "B", "C", "D"}


def _extract_choice(text: str, valid: set[str]) -> str | None:
    """从模型输出里抽出选项字母。

    优先级顺序同 CausalPhys：严格匹配优先，避免把模型回显的整个选项
    （``"C) Insert the plug"``）误判成 A。
    """
    if not text:
        return None
    s = text.strip()

    # 1) 严格：整段就是单个字母（可带括号/点/冒号）
    m = re.fullmatch(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?[\.\:,]?\s*", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 2) "answer is X" / "答案是 X"
    m = re.search(
        r"(?:answer|答案|choice|选项)\s*(?:is|为|是|:)?\s*[\(\[]?\s*([A-Za-z])\b",
        s,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 3) 开头就是 "X)" / "X:" / "X." / "(X)"
    m = re.match(r"^\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*[\.\:\)]\s", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 4) 兜底：取最后一个孤立的大写字母
    for ch in reversed(re.findall(r"\b([A-Z])\b", s)):
        if ch in valid:
            return ch
    return None


def process_results(doc: Mapping[str, Any], results: Sequence[str]) -> dict[str, float]:
    prediction = _extract_choice(results[0] if results else "", _valid_choices(doc))
    target = _normalize_answer(doc.get("correct_answer"))
    return {"momagraph_accuracy": 1.0 if prediction == target else 0.0}


def aggregate_results(results: Sequence[float]) -> float:
    return sum(results) / len(results) if results else 0.0
