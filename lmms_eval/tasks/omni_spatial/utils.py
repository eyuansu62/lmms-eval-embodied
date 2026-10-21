"""OmniSpatial task for lmms-eval.

A 1,533-item single-image MCQ on comprehensive spatial reasoning, with
four task types (Complex_Logic, Dynamic_Reasoning, Perspective_Taking,
Spatial_Interaction) and ten sub-task types.

Each item carries 2+ options with a 0-based ``answer`` index. The
default scoring mode (``eval_type: direct``) does deterministic
letter-match — the question prompt asks the model to reply with a
single letter — so no LLM judge is required.

Dataset: nv-njb/OmniSpatial-Test — a bundled re-host of the test split
of qizekun/OmniSpatial (the canonical release ships a 1.66 GB zip
that doesn't load directly via load_dataset).

Reference: https://huggingface.co/datasets/qizekun/OmniSpatial
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from PIL import Image

# ---------------------------------------------------------------------------
# Prompting (matches the default direct-MCQ mode of the upstream OmniSpatial
# reference implementation)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Multi-View Image** - a RGB frame depicting a scene from 6 novel views.
3. **Question** - a natural-language query about spatial relationships between objects in the image.
4. **Options** - >=2 answer candidates, each tagged by a capital letter (A, B, C, D...).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option---never refuse or reply "insufficient information."
"""

DIRECT_FORMAT = "\nNote: You only need to respond with A, B, C, or D without providing any additional information.\n"


def _strip_think(text: str) -> str:
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thought>", "", text, flags=re.DOTALL)
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def omni_spatial_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    return [doc["image"].convert("RGB")]


def omni_spatial_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    prompt = DEFAULT_SYSTEM_PROMPT + DIRECT_FORMAT + "\n" + doc["question"]
    for i, opt in enumerate(doc["options"]):
        prompt += f"\n{chr(65 + i)}. {opt}"
    return prompt


def omni_spatial_doc_to_answer(doc: Dict[str, Any]) -> str:
    return chr(65 + int(doc["answer"]))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _extract_answer_letter(text: str) -> str:
    """从回复里取出选项字母；取不到返回 ""。

    原来是 ``response.strip().upper()[:1]`` —— 直接截首字符，任何带包装的回复都会取错：

        '(B)'              -> '('   （应为 B）
        'Answer: B'        -> 'A'   （应为 B）
        'The answer is B.' -> 'T'   （应为 B）
        '**B**'            -> '*'   （应为 B）

    同 embspatial 的同类 bug（那边已按同一套改法修过）。prompt 里写了
    "only need to respond with A, B, C, or D"，所以实测 1533 条全部合规、当前
    零影响 —— 但换一个爱先讲一段再给答案的模型就会系统性错判。
    """
    text = text.strip()
    if not text:
        return ""

    # 首选「整段开头就是一个独立选项字母」：字母后必须跟分隔符或空白/结尾。
    # `(?:\s|$)` 是拦 'Based'/'The' 这类单词首字母的关键（B 后面是 'a'，不满足）。
    m = re.match(r"^[\(\s]*([A-Z])[\)\.\:\,]*(?:\s|$)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 回退：正文里显式给出的答案，取最后一个（答案通常在末尾）。
    for pat in (
        r"\(([A-D])\)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\b",
        r"\b([A-D])\)",
    ):
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            return found[-1].upper()

    # 再退一步：整段里恰好只出现一个选项字母
    letters = set(re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE))
    if len(letters) == 1:
        return letters.pop().upper()
    return ""


def omni_spatial_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    raw = results[0] if results else ""
    response = _strip_think(raw)
    pred_letter = _extract_answer_letter(response)
    gt_letter = chr(65 + int(doc["answer"]))
    score = int(pred_letter == gt_letter)
    item = {
        "score": score,
        "task_type": str(doc.get("task_type", "")),
        "sub_task_type": str(doc.get("sub_task_type", "")),
    }
    return {
        "omni_spatial_score": item,
        "complex_logic": item,
        "dynamic_reasoning": item,
        "perspective_taking": item,
        "spatial_interaction": item,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def omni_spatial_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    return _mean([r["score"] for r in results])


def _by_task(results, target):
    return _mean([r["score"] for r in results if r["task_type"] == target])


def omni_spatial_aggregate_complex_logic(results):
    return _by_task(results, "Complex_Logic")


def omni_spatial_aggregate_dynamic_reasoning(results):
    return _by_task(results, "Dynamic_Reasoning")


def omni_spatial_aggregate_perspective_taking(results):
    return _by_task(results, "Perspective_Taking")


def omni_spatial_aggregate_spatial_interaction(results):
    return _by_task(results, "Spatial_Interaction")
