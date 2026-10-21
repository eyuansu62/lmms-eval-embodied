"""EPIC-Bench task utilities for lmms-eval.

EPIC-Bench is a visual grounding benchmark covering three categories:
  - TargetLocalization  (24 subtypes, e.g. SpatialRelation, Color, Geometry)
  - Manipulation        (AffordanceRegion, PlacementRegion, ContactRelationship)
  - Navigation          (GroundDetection, FeasiblePath_Ego/Exo, VisualMatching)

Each sample supplies:
  - image                 : scene image (PIL)
  - reference_object_image: cropped reference object (PIL), always present
  - ground_truth_mask     : binary mask (L-mode PIL) same size as image
  - text_label            : natural-language grounding instruction
  - category / task_type  : category and fine-grained task

Scoring: model outputs (x, y) pixel coordinates; hit = point falls inside
the GT mask (mask pixel > 0).  Same approach as PIO/RefSpatial.

HF dataset: rxc205/EPIC-Bench  (train split, 6661 rows)
"""

from __future__ import annotations

import os
import re
from typing import List

from loguru import logger as eval_logger


# ── prompt templates ──────────────────────────────────────────────────────────

PROMPT_DEFAULT = (
    "Given the scene image and the reference object image, locate the target described below:\n"
    "{text_label}\n"
    "Provide the coordinates of the target as (x, y) pixel values in the original scene image. "
    "Output only the coordinates."
)

PROMPT_BY_CATEGORY = {
    "TargetLocalization": (
        "Locate the following target in the scene image:\n"
        "{text_label}\n"
        "Use the reference object image as a hint. "
        "Output the (x, y) pixel coordinates of the target. Output only the coordinates."
    ),
    "Manipulation": (
        "For the manipulation task, identify the region described below in the scene image:\n"
        "{text_label}\n"
        "The reference image shows the relevant object. "
        "Output the (x, y) center pixel coordinates of that region. Output only the coordinates."
    ),
    "Navigation": (
        "For navigation, locate the following feature in the scene image:\n"
        "{text_label}\n"
        "Output the (x, y) pixel coordinates. Output only the coordinates."
    ),
}


def epic_doc_to_visual(doc):
    """Return [scene_image, reference_object_image] for this sample."""
    visuals = []
    if doc.get("image") is not None:
        visuals.append(doc["image"])
    if doc.get("reference_object_image") is not None:
        visuals.append(doc["reference_object_image"])
    return visuals


def epic_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", pre)
        post = lmms_eval_specific_kwargs.get("post_prompt", post)

    cat = doc.get("category", "")
    template = PROMPT_BY_CATEGORY.get(cat, PROMPT_DEFAULT)
    text = template.format(text_label=doc.get("text_label", ""))
    return f"{pre}{text}{post}"


def epic_doc_to_target(doc):
    # Ground truth is the mask — we return a sentinel; scoring uses the mask directly
    return "mask"


# ── scoring ───────────────────────────────────────────────────────────────────

def _extract_coords(text: str, width: int, height: int):
    """Extract (x, y) pixel coordinates from model output.

    Handles:  (320, 240)  /  320, 240  /  0.5, 0.5 (normalised)

    正则必须含 ``\\d``：宽松的 ``[\\d.]+`` 能匹配**纯点号**，模型输出
    ``(., .)`` 这类占位坐标时会走到 ``float('.')`` 抛 ``ValueError``，
    把整个任务打断、一条结果都不落盘（pio 就这么崩过，见 reports/B_tier_four_models.md）。
    ``float`` 再加一层保护，解析不出返回 ``(None, None)`` 而非崩溃。
    """
    text = text.strip()

    # 只扫开头一段。坐标总在输出最前面，但退化输出能长到 max_new_tokens：
    # 实测 mimo-v2.6-pro 在 epic 上吐了一条 65536 个 '0' 的回复（见下），
    # 让无锚正则跑到灾难性回溯。截断把最坏规模钉死。
    # 2026-10-11：mimo-v2.6-pro 的 epic 因此单线程空转 19 小时（py-spy 确认卡在
    # 下面 patterns 的第二条 re.search）。与 pio 2026-10-06 那次是同一个 bug。
    head = text[:1024]

    # 廉价前置过滤：三种格式都至少含一个逗号或空白。纯数字退化输出在这里就被
    # O(n) 挡掉，根本不进正则 —— 单靠截断不够，那条正则在 1024 个 '0' 上仍要 ~6 s。
    if "," not in head and " " not in head:
        return None, None

    # 占有量词（3.11+）让回溯彻底不可能，1024 字符降到 ~3 ms。老解释器（本项目声明
    # >=3.10）回落到等价的可回溯写法，由上面的截断兜住最坏情况。
    try:
        re.compile(r"\d++")
        num = r"(\d++\.?\d*+|\.\d++)"
    except re.error:
        num = r"(\d+\.?\d*|\.\d+)"

    # bbox 必须先行匹配：4 元组 [x1,y1,x2,y2] 会被下面「两两数字」的规则截成左上角。
    # 3B 在 epic 上 30/30 都输出 bbox（``{"bbox_2d": [...]}``），把左上角当成点等于在测
    # 「方框角点是否落在物体里」——角落几乎永远不在，box 输出的模型被系统性判 0。
    # 取中心，与「点定位」的语义对齐。
    # 注：这条本身以 ``\[`` 锚定，回溯风险远低于下面两条，但一并走 head/num。
    mb = re.search(rf"\[\s*{num}\s*,\s*{num}\s*,\s*{num}\s*,\s*{num}\s*\]", head)
    if mb:
        try:
            x1, y1, x2, y2 = (float(mb.group(i)) for i in range(1, 5))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if 0 <= cx <= 1 and 0 <= cy <= 1:  # 归一化 bbox
                cx, cy = cx * width, cy * height
            return cx, cy
        except ValueError:  # 保守：解析不了就走普通路径
            pass

    patterns = [
        rf"\(\s*{num}\s*,\s*{num}\s*\)",
        rf"{num}\s*,\s*{num}",
        rf"{num}\s+{num}",
    ]
    for pat in patterns:
        m = re.search(pat, head)
        if m:
            try:
                x, y = float(m.group(1)), float(m.group(2))
            except ValueError:  # 理论到不了，保守兜底
                continue
            # If values are in [0, 1] treat as normalised
            if 0 <= x <= 1 and 0 <= y <= 1:
                x, y = x * width, y * height
            return x, y
    return None, None


def _point_in_mask(px: float, py: float, mask) -> bool:
    """Return True if the pixel at (px, py) is non-zero in the mask."""
    try:
        import numpy as np
        arr = np.array(mask)
        ix, iy = int(round(px)), int(round(py))
        h, w = arr.shape[:2]
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False
        return bool(arr[iy, ix] > 0)
    except Exception:
        return False


def epic_process_results(doc, results):
    response = results[0] if results else ""
    img = doc.get("image")
    mask = doc.get("ground_truth_mask")
    w, h = img.size if img is not None else (1, 1)

    px, py = _extract_coords(response, w, h)
    if px is None:
        return {
            "epic_accuracy": 0,
            "epic_parse_ok": 0,
            "epic_accuracy_by_category": {"category": doc.get("category", ""), "value": 0},
            "epic_accuracy_by_task_type": {"task_type": doc.get("task_type", ""), "value": 0},
        }

    # 模型看不到原图尺寸（API 传图会缩放），所以「原图像素坐标」这个要求它没法满足：
    # 实测 27b 输出的坐标全在 0~876，而图像宽高跨 333~1280，19/30 直接越界。
    # 默认两种解释都试取命中；GT mask 中位仅占 0.32%，虚高可忽略（<0.7 个百分点）。
    scheme = os.environ.get("EPIC_COORD_FORMAT", "auto").lower()
    if scheme == "auto":
        # _extract_coords 已把 [0,1] 的小数放大成像素，这里的候选只剩像素与 0-1000 两种
        cands = [(px, py), (px / 1000.0 * w, py / 1000.0 * h)]
    elif scheme == "relative_1000":
        cands = [(px / 1000.0 * w, py / 1000.0 * h)]
    elif scheme == "absolute":
        cands = [(px, py)]
    else:
        # 原来的写法是 `if os.environ.get(...) or scheme == "auto":`，任何非空取值都
        # 落进 auto 分支，relative_1000 / absolute 永远到不了 —— 想强制单一口径复核
        # 虚高时这个开关给不出结果，是个哑开关。现在显式取值生效；拼错的取值不再
        # 静默当 auto（那会让「以为设了单一口径」变成「两种都试」而虚高无人察觉），
        # 改为告警后回落 auto，避免直接抛异常打断整个 task。
        eval_logger.warning(
            f"EPIC_COORD_FORMAT 取值非法: {scheme!r}，已回落 auto（可选 auto/absolute/relative_1000）"
        )
        cands = [(px, py), (px / 1000.0 * w, py / 1000.0 * h)]

    hit = 0
    if mask is not None:
        hit = max(int(_point_in_mask(x, y, mask)) for x, y in cands)

    cat = doc.get("category", "")
    task_type = doc.get("task_type", "")
    return {
        "epic_accuracy": hit,
        "epic_parse_ok": 1,
        "epic_accuracy_by_category": {"category": cat, "value": hit},
        "epic_accuracy_by_task_type": {"task_type": task_type, "value": hit},
    }


# ── aggregation ───────────────────────────────────────────────────────────────

def epic_aggregate_accuracy(results: List[float]) -> float:
    return sum(results) / len(results) if results else 0.0


def epic_aggregate_parse_rate(results: List[float]) -> float:
    return sum(results) / len(results) if results else 0.0


def epic_aggregate_by_category(results: list) -> dict:
    from collections import defaultdict
    counts: dict = defaultdict(list)
    for r in results:
        counts[r["category"]].append(r["value"])
    return {k: sum(v) / len(v) for k, v in counts.items()}


def epic_aggregate_by_task_type(results: list) -> dict:
    from collections import defaultdict
    counts: dict = defaultdict(list)
    for r in results:
        counts[r["task_type"]].append(r["value"])
    return {k: sum(v) / len(v) for k, v in counts.items()}
