import os
import re
from collections import defaultdict
from pathlib import Path

import yaml
from loguru import logger as eval_logger

from lmms_eval import utils as lmms_utils

# PhysBench category breakdowns
TASK_TYPES = ["property", "relationships", "scene", "dynamics"]
ABILITY_TYPES = ["identify", "comparison", "static", "dynamic", "perception", "prediction", "judgment", "reasoning"]


def _load_task_config():
    with open(Path(__file__).parent / "physbench.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = [line for line in raw_data if "!function" not in line]
    return yaml.safe_load("".join(safe_data))


def _get_cache_dir():
    # 优先用项目内 media 目录（离线部署），回退到 HF_HOME
    local = Path(__file__).resolve().parent / "media"
    if local.exists() and any(local.iterdir()):
        return str(local)
    config = _load_task_config()
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return lmms_utils.resolve_cache_dir(config["dataset_kwargs"]["cache_dir"], base_dir=hf_home)


def physbench_process_docs(dataset):
    """Drop the source rows whose public answer is empty."""
    return dataset.filter(lambda example: bool(example.get("answer", "").strip()))


def _resolve_media_path(cache_dir, fname):
    """Resolve media file path, checking subdirectory and flat layouts.

    The PhysBench README instructs users to ``unzip image.zip -d image`` which
    produces ``{cache_dir}/image/foo.png``.  The lmms-eval framework's zip
    extraction, however, extracts flat into ``cache_dir`` giving
    ``{cache_dir}/foo.png``.  We check both locations.
    """
    ext = os.path.splitext(fname)[1].lower()
    subdir = "video" if ext == ".mp4" else "image"

    # Prefer the subdirectory layout (manual unzip / README instructions)
    subdir_path = os.path.join(cache_dir, subdir, fname)
    if os.path.exists(subdir_path):
        return subdir_path

    # Fall back to flat layout (framework zip extraction)
    flat_path = os.path.join(cache_dir, fname)
    if os.path.exists(flat_path):
        return flat_path

    # Neither found; return the subdirectory path for the warning message
    eval_logger.warning(f"PhysBench media file not found: {subdir_path} (also checked {flat_path})")
    return subdir_path


def physbench_doc_to_visual(doc):
    """Return list of media paths (images and videos) for a document."""
    cache_dir = _get_cache_dir()
    media = doc.get("file_name") or doc.get("media_path", "")
    if isinstance(media, str):
        media = [media] if media else []
    return [_resolve_media_path(cache_dir, fname) for fname in media]


def physbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format the question text.

    The question field already contains inline options (A-D) with <video>/<image>
    placeholders. The lmms-eval framework handles visual injection, so we keep
    the placeholders as-is and append the post_prompt.
    """
    question = doc["question"].strip()

    post_prompt = ""
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return question + post_prompt


def _extract_choice(text: str, valid: set[str]) -> str | None:
    """从模型输出里抽出选项字母。优先级同 CausalPhys / NegVQA 的实现。

    原先只看首字符，遇到会输出推理过程的模型（如 GLM-5.3-Flash）会把
    ``"Based on the video at 00:00..."`` 的首字母 B 当成答案，判分被污染。
    这里改为按优先级提取，既保留官方「答案在前」的语义，又能容忍冗长输出。
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

    # 3) 开头就是 "X:" / "X." / "(X)"
    m = re.match(r"^\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*[\.\:]\s", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 4) 兜底：取最后一个孤立的大写字母
    letters = re.findall(r"\b([A-Z])\b", s)
    for ch in reversed(letters):
        if ch in valid:
            return ch
    return None


def _valid_choices(doc) -> set[str]:
    """从题干推断合法选项集合（PhysBench 的选项内联在 question 里）。

    选项可能是换行分隔，也可能挤在同一行（``A. Less than twice B. Exactly twice``），
    所以不能锚定行首。按出现顺序找连续递增的字母序列（A→B→C→D），
    避免把题干正文里的 ``I.`` 之类误当成选项。
    """
    q = str(doc.get("question") or "")
    best: list[str] = []
    run: list[str] = []
    for m in re.finditer(r"([A-Z])\s*[\.\)]\s", q):
        ch = m.group(1)
        if ch == "A":
            run = ["A"]
        elif run and ch == chr(ord(run[-1]) + 1):
            run.append(ch)
        if len(run) > len(best):
            best = list(run)
    return set(best) or {"A", "B", "C", "D"}


def physbench_process_results(doc, results):
    """Answer extraction then comparison against the ground truth letter.

    Protocol note: PhysBench's official scorer takes the response's *first*
    character as the answer. That breaks on models that reason in the response
    body, so extraction is now priority-based (see ``_extract_choice``).
    """
    pred = results[0].strip() if results else ""
    pred_ans = _extract_choice(pred, _valid_choices(doc)) or ""
    gt_ans = doc.get("answer", "")

    score = 1.0 if pred_ans.upper() == gt_ans.upper() and gt_ans else 0.0

    data_dict = {
        "idx": doc.get("idx", doc.get("index", "")),
        "task_type": doc.get("task_type", ""),
        "sub_type": doc.get("sub_type", ""),
        "ability_type": doc.get("ability_type", ""),
        "pred_answer": pred_ans,
        "answer": gt_ans,
        "score": score,
    }

    return {"physbench_accuracy": data_dict}


def physbench_aggregate_results(results):
    """Compute overall and per-category accuracy."""
    # Per task_type breakdown
    task_type_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    # Per ability_type breakdown
    ability_type_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total_answered = 0

    for result in results:
        total_answered += 1
        correct = result["score"]
        total_correct += correct

        task_type = result.get("task_type", "")
        if task_type:
            task_type_scores[task_type]["total"] += 1
            task_type_scores[task_type]["correct"] += correct

        ability_type = result.get("ability_type", "")
        if ability_type:
            ability_type_scores[ability_type]["total"] += 1
            ability_type_scores[ability_type]["correct"] += correct

    # Log per-category results
    for task_type in TASK_TYPES:
        stats = task_type_scores.get(task_type, {"correct": 0, "total": 0})
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info(f"PhysBench task_type={task_type}: {acc:.1f}% ({int(stats['correct'])}/{stats['total']})")

    for ability_type in ABILITY_TYPES:
        stats = ability_type_scores.get(ability_type, {"correct": 0, "total": 0})
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info(f"PhysBench ability_type={ability_type}: {acc:.1f}% ({int(stats['correct'])}/{stats['total']})")

    overall_acc = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"PhysBench Overall: {overall_acc:.1f}% ({int(total_correct)}/{total_answered})")
    return overall_acc
