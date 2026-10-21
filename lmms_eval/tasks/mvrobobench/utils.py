"""MV-RoboBench task utilities for lmms-eval.

Images are downloaded on-demand from HF (AaronFengZY24/MV_Robobench)
and cached at ~/.cache/lmms_eval/mvrobobench/<domain>/<task>/<QAxxx>/<view>.png.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import List

from loguru import logger as eval_logger

CACHE_DIR = pathlib.Path(__file__).resolve().parent / "media"
HF_REPO = "AaronFengZY24/MV_Robobench"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _get_image(hf_path: str):
    """Download a single view image from HF if not cached; return PIL Image."""
    from huggingface_hub import hf_hub_download
    from PIL import Image

    cache_file = CACHE_DIR / hf_path
    if not cache_file.exists():
        local = hf_hub_download(
            HF_REPO,
            repo_type="dataset",
            filename=hf_path,
            local_dir=str(CACHE_DIR),
        )
        cache_file = pathlib.Path(local)
    return Image.open(cache_file).convert("RGB")


def doc_to_visual(doc):
    """Load all view images for a question."""
    # id = "<domain>/<task>/<QAxxx>"
    # views stored as view1.png, view2.png, ... beside QA.json
    domain_task_qa = doc["id"]  # e.g. agiworld/robotic_affordance_recognition/QA079
    n_views = doc.get("n_views", 3)
    images = []
    for i in range(1, n_views + 1):
        hf_path = f"{domain_task_qa}/view{i}.png"
        try:
            images.append(_get_image(hf_path))
        except Exception as exc:
            eval_logger.warning(f"MV-RoboBench: failed to load {hf_path}: {exc}")
    return images


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = ""
    post = "\nAnswer with the option's letter (A, B, C, D, or E) only."
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", pre)
        post = lmms_eval_specific_kwargs.get("post_prompt", post)

    opts = doc["options"]  # dict {"A": "...", ...}
    opt_str = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    return f"{pre}{doc['question']}\n{opt_str}{post}"


def doc_to_target(doc):
    return doc["answer"]


# ── scoring ──────────────────────────────────────────────────────────────────

def _extract_letter(text: str, valid: str = "ABCDE") -> str:
    """Extract the option letter a model response settles on.

    Matches the RoboBench implementation: reasoning models answer *after*
    thinking, so the trailing letter is preferred over the leading one.
    """
    text = text.strip()
    if not text:
        return ""

    m = re.fullmatch(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?[\.\:\,\*]*\s*", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    m = re.search(
        r"(?:answer|答案|choice|选项)\s*(?:is|为|是|:)?\s*[\(\[]?\s*\**\s*([A-Za-z])\b",
        text,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    for pat in (r"[\.\!\?]\s+[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*\.?\s*$",
                r"\*\*\s*([A-Za-z])\s*\*\*\s*$"):
        m = re.search(pat, text)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    letters = [c for c in re.findall(r"\b([" + valid + r"])\b", text.upper()) if c in valid]
    if letters:
        return letters[-1]
    return ""


def process_results(doc, results):
    response = results[0] if results else ""
    pred = _extract_letter(response)
    correct = int(pred == doc["answer"].upper()) if pred else 0
    return {"mvrb_accuracy": correct}


def aggregate_results(results: List[float]) -> float:
    # lmms-eval 传入的是该 metric 的取值序列，不是 process_results 的字典
    return sum(results) / len(results) if results else 0.0
