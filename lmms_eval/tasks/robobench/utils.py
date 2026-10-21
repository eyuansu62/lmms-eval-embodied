"""RoboBench task utilities for lmms-eval.

Images are downloaded on-demand from HF and cached at
~/.cache/lmms_eval/robobench/<unique_id>/<frame>.png.

Supports both MCQ subtasks (A/B/C/D) and open-ended planning subtasks
(implicit_demand_goal, explicit_object_goal, robobench_plan_*).  Open-ended Q3
(yes/no) is scored here; Q1/Q2 store raw output and are scored post-hoc by
judge_robobench_openended.py (LLM judge, official protocol).

Prompts follow the official harness (yulin-luo/RoboBench,
src/robobench/data/dataset.py): a subtask-specific preamble from
data/system_prompt.json is prepended to every question. Before 2026-10-14 these
tasks sent the bare question only — e.g. affordance questions never said the
point coordinates are normalised, and the error-analysis questions never
defined the four error classes. Results produced before that are not
comparable with the official protocol.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import re
from typing import List

from loguru import logger as eval_logger

CACHE_DIR = pathlib.Path(__file__).resolve().parent / "media"
HF_REPO = "LeoFan01/RoboBench"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _strip_question_suffix(unique_id: str) -> str:
    """Remove _Q1/_Q2/_Q3 suffix added by RoboBench to unique_id for multi-question episodes.

    unique_id = .../images/rtx_frames__episode_1036_Q1
    actual dir =  .../images/rtx_frames__episode_1036
    """
    return re.sub(r"_Q\d+$", "", unique_id)


def _candidate_paths(unique_id: str, filename: str) -> list[str]:
    """仓库里两种布局，按 unique_id 与文件名的关系判断先试哪个。

    单图题：图片与 unique_id 同名，直接躺在 images/ 下
        unique_id = .../images/28_packtable_2
        file      = 28_packtable_2.png
        -> .../images/28_packtable_2.png
    多帧题：每个 episode 一个子目录（unique_id 可能带 _Q1/_Q2 后缀，需先剥掉）
        unique_id = .../images/rtx_frames__episode_1036_Q2
        file      = frame_01.png
        -> .../images/rtx_frames__episode_1036/frame_01.png
    """
    uid_clean = _strip_question_suffix(unique_id)
    parent = str(pathlib.PurePosixPath(uid_clean).parent)
    sibling = f"{parent}/{filename}"
    nested = f"{uid_clean}/{filename}"
    stem = pathlib.PurePosixPath(filename).stem
    if stem == pathlib.PurePosixPath(uid_clean).name:
        return [sibling, nested]
    return [nested, sibling]


def _get_image(unique_id: str, filename: str):
    """Download image from HF if not cached; return PIL Image."""
    from huggingface_hub import hf_hub_download
    from PIL import Image

    last_exc: Exception | None = None
    for hf_path in _candidate_paths(unique_id, filename):
        cache_file = CACHE_DIR / hf_path
        if cache_file.exists():
            return Image.open(cache_file).convert("RGB")
        try:
            local = hf_hub_download(
                HF_REPO,
                repo_type="dataset",
                filename=hf_path,
                local_dir=str(CACHE_DIR),
            )
            return Image.open(local).convert("RGB")
        except Exception as exc:  # 404 -> 试另一种布局
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def doc_to_visual(doc):
    images = []
    for fname in doc["image_paths"]:
        try:
            images.append(_get_image(doc["id"], fname))
        except Exception as exc:
            eval_logger.warning(f"RoboBench: failed to load {doc['id']}/{fname}: {exc}")
    return images


@functools.lru_cache(maxsize=1)
def _system_prompts() -> dict:
    return json.loads((pathlib.Path(__file__).resolve().parent / "data" / "system_prompt.json").read_text())


def _subtask_of(unique_id: str) -> str:
    """'2_perception_reasoning/.../static_attribute/images/...' -> 'static_attribute'"""
    return unique_id.split("/images/", 1)[0].rsplit("/", 1)[-1]


# 与官方 dataset._build_multiple_choice_prompt 一致（键名的拼写错误 afforadance 是上游原样）
_MCQ_SUBTASK_PROMPT = {
    "robot_type": "robot_type",
    "robot_view": "robot_view",
    "spatial_relation": "spatial_relation",
    "spatial_temporal_causality": "spatial_temporal_causality",
    "static_affordance": "afforadance_point",
    "dynamic_affordance": "afforadance_point",
    "navigation_visual_prompt": "afforadance_point",
    "high_level_planning_error": "high_level_planning_error",
    "low_level_execution_error": "low_level_execution_error",
}
_ATTRIBUTE_PROMPT = {
    "is_sealed": "Sealed",
    "material": "Material",
    "contents": "Content",
    "can_contain_liquid": "Liquid",
    "transparency": "Transparency",
    "deformability": "Deformability",
    "fragility": "Fragility",
    "mass": "Mass",
}


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Official MCQ prompt: <subtask preamble><question>\n<A. ...>\n...<MCQ_post_prompt>."""
    sp = _system_prompts()
    subtask = _subtask_of(doc["id"])
    preamble = ""
    if subtask == "static_attribute":
        attribute = doc["id"].split("/images/", 1)[1].split("/", 1)[0]
        preamble = sp[_ATTRIBUTE_PROMPT[attribute]]
    elif subtask in _MCQ_SUBTASK_PROMPT:
        preamble = sp[_MCQ_SUBTASK_PROMPT[subtask]]

    opts = doc["options"]  # dict: {"A": "...", "B": "...", ...}
    opt_str = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    return f"{preamble}{doc['question']}\n{opt_str}{sp['MCQ_post_prompt']}"


def doc_to_target(doc):
    return doc["answer"]


# ── scoring ──────────────────────────────────────────────────────────────────

def _extract_letter(text: str, valid: str = "ABCDE") -> str:
    """Extract the option letter a model response settles on.

    Reasoning models answer *after* thinking, so the choice usually sits at
    the end of a paragraph. Prefer an explicit answer marker and a trailing
    letter; only then fall back to the first letter in the text (which is
    what a short, direct reply looks like).
    """
    text = text.strip()
    if not text:
        return ""

    # Direct single letter: "B" / "B." / "(B)"
    m = re.fullmatch(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?[\.\:\,\*]*\s*", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Explicit marker: "Answer: B" / "The answer is C" / "答案：D"
    m = re.search(
        r"(?:answer|答案|choice|选项)\s*(?:is|为|是|:)?\s*[\(\[]?\s*\**\s*([A-Za-z])\b",
        text,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Trailing letter after sentence punctuation or as markdown bold: "...left. C" / "**A**"
    for pat in (r"[\.\!\?]\s+[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*\.?\s*$",
                r"\*\*\s*([A-Za-z])\s*\*\*\s*$"):
        m = re.search(pat, text)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    # Last resort: last bare letter that is actually an option
    letters = [c for c in re.findall(r"\b([" + valid + r"])\b", text.upper()) if c in valid]
    if letters:
        return letters[-1]
    return ""


def process_results(doc, results):
    response = results[0] if results else ""
    pred = _extract_letter(response)
    correct = int(pred == doc["answer"].upper()) if pred else 0
    return {"robobench_accuracy": correct}


def aggregate_results(results: List[float]) -> float:
    # lmms-eval 传入的是该 metric 的取值序列，不是 process_results 的字典
    return sum(results) / len(results) if results else 0.0


# ── open-ended subtasks ───────────────────────────────────────────────────────

def doc_to_text_openended(doc, lmms_eval_specific_kwargs=None):
    """Official planning prompt (dataset._build_planning_prompt).

    <skill_list_Q1|Q2|Q3>[<color|number|shape|size 说明>]You are a <type> robot.[<多视角说明>]<question>

    Q1 的 skill list 定义了动作语法 ``1-move_to(none, x), 2-grasp(x), ...``；不给它，
    模型答出来的是自然语言，judge 的逐节点匹配无从谈起。
    """
    sp = _system_prompts()
    subtask = doc["subtask"]
    if subtask == "navigation":
        prompt = sp["navigation_skill_list"]
    else:
        prompt = sp[f"skill_list_{doc['qtype']}"]
    if subtask in ("color", "number", "shape", "size"):
        prompt += sp[subtask]

    robotic_type = "single-arm" if doc["robotic_type"] == "human" else doc["robotic_type"]
    prompt += f"You are a {robotic_type} robot."
    if subtask == "multi_view":
        view_names = [pathlib.PurePosixPath(f).stem for f in doc["image_paths"]]
        prompt += (
            "The multiple images you have received now represent camera images from "
            "different views at the same time. The names of these views are "
            f"{view_names} in order."
        )
    return prompt + doc["question"]


def extract_yes_no(text: str) -> str:
    """取回复里**最后一个** yes/no（推理模型先想后答，答案在末尾）。"""
    found = re.findall(r"\b(yes|no)\b", text.lower())
    return found[-1] if found else ""


def process_results_openended(doc, results):
    """Q3（yes/no）在这里直接判；Q1/Q2 记 -1（待判），由 judge_robobench_openended.py 覆写。

    -1 不是分数：build_report 与 aggregate_openended 都会把负值当「未判」剔掉。
    """
    response = results[0] if results else ""
    if doc["qtype"] == "Q3":
        pred = extract_yes_no(response)
        score = float(pred == doc["answer"].strip().lower()) if pred else 0.0
    else:
        score = -1.0
    return {"rb_openended_score": score}


def aggregate_openended(results: List[float]) -> float:
    """只平均已判分的题。全部待判时返回 -1，提醒去跑 judge。"""
    judged = [r for r in results if r >= 0]
    return sum(judged) / len(judged) if judged else -1.0
