"""NegVQA 筛选切片的评测逻辑。

数据来源
--------
``yuhuizhang/NegVQA``（ACL 2025 Findings，未 gated）。
**只接经过审核的切片，不是全量** —— 见 ``screen_negvqa.py``。

为什么不用全量
--------------
NegVQA 的否定是 LLM 自动改写生成的，答案机械翻转。实测 dev 全量 1000 例的答案
是**完美一一翻转**（488A/512B -> 488B/512A），说明缺陷是系统性的，不是个别样本：

* ``index=326``（MMMU）：``Do not predict the overtime wages...`` ——
  "不要预测"是禁止*行为*，不等于"预测一个错误数值"，答案翻转不成立。
* ``How many bananas are there? 4 / 2`` —— "有几根香蕉不在图里"，
  答案标为 4，补集不是另一个数。

筛选后的 91/1000 全部是**互斥二选一**（朝向、位置、类别、月相这类
"选了 A 就不可能是 B"的属性题），只有在这些题上"不是 A"才等价于"是 B"。

三个指标
--------
每个样本同时出两道题（原题 + 否定题），分别计分，并额外报"成对全对"：

    negvqa_original_accuracy   原题准确率
    negvqa_negated_accuracy    否定题准确率
    negvqa_paired_accuracy     成对全对率（同一视觉内容下两问都答对）

成对全对率是这套评测的关键指标：它排除"靠猜"的干扰。
若模型在两问上独立乱猜，成对全对率会明显低于单问准确率的平方关系所暗示的值。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

_IMG_TAG: Final = re.compile(r"<image \d+>")


def _clean(text: Any) -> str:
    """去掉题干里的 ``<image N>`` 占位符。"""
    return _IMG_TAG.sub("", str(text or "")).strip()


def _options_text(doc: Mapping[str, Any]) -> str:
    """两选项，自带 ``A)`` / ``B)`` 前缀。"""
    return f"A) {doc.get('A', '')}\nB) {doc.get('B', '')}"


def _build(question: str, doc: Mapping[str, Any]) -> str:
    return f"{_clean(question)}\n{_options_text(doc)}"


# --- 原题 ---


def doc_to_visual_original(doc: Mapping[str, Any]) -> list[Any]:
    """原题与否定题共用同一张图（这是成对对照的前提）。"""
    image = doc.get("image")
    return [image] if image is not None else []


def doc_to_text_original(doc: Mapping[str, Any], lmms_eval_specific_kwargs: Mapping[str, str] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    post = kwargs.get("post_prompt", "\nAnswer with the option's letter from the given choices directly.")
    return f"{_build(doc.get('original_question'), doc)}{post}"


def doc_to_target_original(doc: Mapping[str, Any]) -> str:
    return str(doc.get("original_answer") or "").strip().upper()


# --- 否定题 ---


def doc_to_text_negated(doc: Mapping[str, Any], lmms_eval_specific_kwargs: Mapping[str, str] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    post = kwargs.get("post_prompt", "\nAnswer with the option's letter from the given choices directly.")
    return f"{_build(doc.get('negated_question'), doc)}{post}"


def doc_to_target_negated(doc: Mapping[str, Any]) -> str:
    return str(doc.get("negated_answer") or "").strip().upper()


# --- 判分 ---


def _extract_choice(text: str, valid: set[str]) -> str | None:
    """从模型输出里抽出选项字母。优先级同 CausalPhys 的实现。"""
    if not text:
        return None
    s = text.strip()

    m = re.fullmatch(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?[\.\:,]?\s*", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    m = re.search(
        r"(?:answer|答案|choice|选项)\s*(?:is|为|是|:)?\s*[\(\[]?\s*([A-Za-z])\b",
        s,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    m = re.match(r"^\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*[\.\:\)]\s", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    for ch in reversed(re.findall(r"\b([A-Z])\b", s)):
        if ch in valid:
            return ch
    return None


def _valid_choices(doc: Mapping[str, Any]) -> set[str]:
    """NegVQA 恒为两选项，但保留从选项文本推断的余地。"""
    found = set()
    for key in ("A", "B"):
        if str(doc.get(key) or "").strip():
            found.add(key)
    return found or {"A", "B"}


def process_results_original(doc: Mapping[str, Any], results: Sequence[str]) -> dict[str, float]:
    pred = _extract_choice(results[0] if results else "", _valid_choices(doc))
    target = doc_to_target_original(doc)
    return {"negvqa_original_accuracy": 1.0 if pred == target else 0.0}


def process_results_negated(doc: Mapping[str, Any], results: Sequence[str]) -> dict[str, float]:
    pred = _extract_choice(results[0] if results else "", _valid_choices(doc))
    target = doc_to_target_negated(doc)
    return {"negvqa_negated_accuracy": 1.0 if pred == target else 0.0}


def aggregate_results(results: Sequence[float]) -> float:
    return sum(results) / len(results) if results else 0.0
