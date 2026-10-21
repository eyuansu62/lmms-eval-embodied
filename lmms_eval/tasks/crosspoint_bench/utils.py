"""CrossPoint-Bench task for lmms-eval.

Cross-view point correspondence benchmark covering four sub-tasks:

- Fine-grained Grounding (coordinate output, in-mask hit)
- Visibility Reasoning (binary MCQ)
- Correspondence-Judgement (MCQ)
- Correspondence-Pointing (coordinate output, in-mask hit)

The HF dataset stores only the JSONL; image files live alongside under
``image/`` and are fetched once with ``snapshot_download`` and cached on
disk for subsequent ``doc_to_visual`` lookups.

Reference: https://arxiv.org/abs/2512.04686
"""

from __future__ import annotations

import base64
import io
import os
import os.path as osp
import re
import threading
from typing import Any, Dict, List

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

REPO_ID = "WangYipu2002/CrossPoint-Bench"

COORDINATE_TASK_TYPES = {"Fine-grained Grounding", "Correspondence-Pointing"}
MCQ_TASK_TYPES = {"Visibility Reasoning", "Correspondence-Judgement"}

COORDINATE_PROMPT_SUFFIX = " Output the point coordinates in JSON format."


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------


_IMAGE_ROOT_LOCK = threading.Lock()
_IMAGE_ROOT: str | None = None


def _image_root() -> str:
    """Download (once) and return the local path to the ``image/`` tree.

    每个文档判分都会走到这里（``_score`` 要读图尺寸），所以这里必须便宜且
    不重复。原先用 ``@lru_cache(maxsize=1)``，但 **lru_cache 在执行期间会释放
    锁**，判分的 4 个线程会同时击穿缓存各下一次 ``snapshot_download``：实测
    4 线程并发耗时 130s（单线程 2s），网络一抖就放大成几十分钟的假死 ——
    2026-10-07 跑 crosspoint 时卡在 715/717 达 20 分钟。改成自己加锁双检。

    另外先试 ``local_files_only=True``：快照已缓存时 0.016s，直接联网要 2s
    （即使文件全在本地，也不会跳过 ETag 往返）。只有本地没有快照才回退下载。
    """
    global _IMAGE_ROOT
    if _IMAGE_ROOT is not None:
        return _IMAGE_ROOT
    with _IMAGE_ROOT_LOCK:
        if _IMAGE_ROOT is not None:  # 双检：等锁期间可能已被别的线程填好
            return _IMAGE_ROOT
        kwargs = dict(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=["image/**", "CrossPoint-Bench.jsonl"],
        )
        try:
            local_dir = snapshot_download(local_files_only=True, **kwargs)
        except Exception:
            # 本地没有完整快照（首次运行 / 缓存被清）才联网
            local_dir = snapshot_download(**kwargs)
        _IMAGE_ROOT = osp.join(local_dir, "image")
        return _IMAGE_ROOT


def crosspoint_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    root = _image_root()
    out = []
    for rel in doc["images"]:
        path = osp.join(root, rel)
        with Image.open(path) as im:
            out.append(im.convert("RGB"))
    return out


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def crosspoint_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] | None = None,
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    question = str(doc["question"])
    if doc.get("type") in COORDINATE_TASK_TYPES:
        question = question + COORDINATE_PROMPT_SUFFIX
    return f"{pre}{question}{post}"


def crosspoint_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc.get("answer", ""))


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


# 坐标必须是**至少含一位数字**的数。原来的 `[0-9.]+` 能匹配纯点号，
# 模型输出 `{"point_2d": [., .]}` 这类占位坐标时会走到 `float('.')` 抛
# ValueError。判分在 evaluator.py:312 调用且外层没有 try，一处异常会打断
# 整个 task、一条结果都不落盘（pio 就是这么崩的，见 reports/B_tier_four_models.md）。
# float 再加一层保护：解析不出就换下一种口径，而不是让任务崩掉。
_NUM = r"(\d+\.?\d*|\.\d+)"


def _extract_coordinates_from_json(text: str):
    patterns = [
        rf'\{{[^{{}}]*"(?:point_2d|point|coordinates?)"\s*:\s*\[\s*{_NUM}\s*,\s*{_NUM}\s*\]',
        rf'\{{[^{{}}]*"x"\s*:\s*{_NUM}[^{{}}]*"y"\s*:\s*{_NUM}',
        rf'\{{[^{{}}]*"y"\s*:\s*{_NUM}[^{{}}]*"x"\s*:\s*{_NUM}',
    ]
    for i, pat in enumerate(patterns):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                x, y = float(m.group(1)), float(m.group(2))
            except ValueError:  # 正则已收紧，理论上到不了；保守兜底
                continue
            return (y, x) if i == 2 else (x, y)
    return None


def _extract_coordinates_from_text(text: str):
    patterns = [
        (rf'<point\s+x="{_NUM}"\s+y="{_NUM}"[^>]*>', False),
        (rf"(?:coordinates?|position|location).*?(?:are|is)\s+{_NUM},\s*{_NUM}", False),
        (rf"[xX]\s*:\s*{_NUM}.*?[yY]\s*:\s*{_NUM}", False),
        (rf"[yY]\s*:\s*{_NUM}.*?[xX]\s*:\s*{_NUM}", True),
        (rf'["\']?x["\']?\s*:\s*{_NUM}.*?["\']?y["\']?\s*:\s*{_NUM}', False),
        (rf"\[{_NUM},\s*{_NUM}\]", False),
        (rf"\({_NUM},\s*{_NUM}\)", False),
        (rf"{_NUM},\s*{_NUM}", False),
    ]
    for pat, swapped in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                x, y = float(m.group(1)), float(m.group(2))
            except ValueError:  # 正则已收紧，理论上到不了；保守兜底
                continue
            if swapped:
                x, y = y, x
            return x, y
    return None


def _extract_coordinates(text: str):
    return _extract_coordinates_from_json(text) or _extract_coordinates_from_text(text)


def _extract_mcq_letter(text: str):
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()

    patterns = [
        r"\\boxed\{(?:\\text\{)?([ABCD])(?:\..*?)?\}",
        r"\((?:Choice\s+)?([ABCD])\)",
        r"\*\*Answer:\s*([ABCD])\*\*",
        r"\*\*([ABCD])\.\s*(?:Yes|No)",
        r"(?:answer is|correct answer is|choose)\s*[:\s]*\*?\*?\(?([ABCD])\)?",
        r"\*\*([ABCD])\*\*",
        r"(?:^|\s|Answer:\s*)\(?([ABCD])\)?\s*[\.:,]",
        r"^\s*\(?([ABCD])\)?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    final_patterns = [
        r"(?:Final Answer|Therefore|So|Hence|Thus|The correct answer is|The answer is)" r".*?(?:\\boxed\{([ABCD])\}|\*\*([ABCD])\*\*|\(?([ABCD])\)?)",
        r"Answer:\s*\(?([ABCD])\)?",
    ]
    for pat in final_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            for g in m.groups():
                if g:
                    return g.upper()

    letters = re.findall(r"\b([ABCD])\b", s)
    if len(letters) == 1:
        return letters[0].upper()
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _decode_base64_mask(b64: str):
    try:
        return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L"))
    except Exception:
        return None


def _point_in_mask(x: int, y: int, mask: np.ndarray, threshold: int = 128) -> bool:
    h, w = mask.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(mask[y, x] > threshold)


def _image_size(path: str):
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


def _point_hit(x_raw: float, y_raw: float, coord_format: str, dims, mask) -> int:
    """在给定坐标口径下判断落点是否命中。"""
    if dims is not None:
        w, h = dims
        if coord_format == "relative_1":
            x_abs, y_abs = int(x_raw * w), int(y_raw * h)
        elif coord_format == "relative_1000":
            x_abs, y_abs = int(x_raw / 1000.0 * w), int(y_raw / 1000.0 * h)
        else:
            x_abs, y_abs = int(x_raw), int(y_raw)
    else:
        x_abs, y_abs = int(x_raw), int(y_raw)
    return int(_point_in_mask(x_abs, y_abs, mask))


def _score(pred: str, doc: Dict[str, Any]) -> int:
    task_type = str(doc.get("type", ""))
    answer = str(doc.get("answer", ""))

    if task_type in COORDINATE_TASK_TYPES:
        coords = _extract_coordinates(pred)
        if coords is None:
            return 0
        x_raw, y_raw = coords
        img_paths = list(doc.get("images") or [])
        dims = None
        if img_paths:
            dims = _image_size(osp.join(_image_root(), img_paths[0]))
        mask = _decode_base64_mask(answer)
        if mask is None:
            return 0

        # 提示词只说 "Output the point coordinates"，没说绝对像素还是归一化，模型各按各的猜：
        # 3B 吐绝对像素，qwen3.8-27b / flash 吐 0-1000 归一化。默认两种都试取命中，
        # 否则分数只是在量「模型有没有猜中我们的口径」，不是量空间定位能力。
        # GT mask 中位仅占图像 0.26%，两种口径同时误命中的虚高可忽略（<0.5 个百分点）。
        coord_format = os.environ.get("CROSSPOINT_COORD_FORMAT", "auto").lower()
        formats = ("absolute", "relative_1", "relative_1000") if coord_format == "auto" else (coord_format,)
        return max(_point_hit(x_raw, y_raw, f, dims, mask) for f in formats)

    if task_type in MCQ_TASK_TYPES:
        letter = _extract_mcq_letter(pred)
        if letter is None:
            return 0
        return int(letter == answer.strip().upper())

    return 0


# All per-doc metric values share the same {hit, type, level} structure so any
# aggregator can filter on type or level.
_METRIC_KEYS = (
    "crosspoint_accuracy",
    "fine_grained_grounding",
    "visibility_reasoning",
    "correspondence_judgement",
    "correspondence_pointing",
    "level_object",
    "level_part",
)


def crosspoint_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    pred = results[0] if results else ""
    item = {
        "hit": _score(pred, doc),
        "type": str(doc.get("type", "")),
        "level": str(doc.get("level", "")),
    }
    return {k: item for k in _METRIC_KEYS}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def crosspoint_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    return _mean([r["hit"] for r in results])


def _aggregate_by_type(results, target_type):
    return _mean([r["hit"] for r in results if r["type"] == target_type])


def _aggregate_by_level(results, target_level):
    return _mean([r["hit"] for r in results if r["level"] == target_level])


def crosspoint_aggregate_fine_grained_grounding(results):
    return _aggregate_by_type(results, "Fine-grained Grounding")


def crosspoint_aggregate_visibility_reasoning(results):
    return _aggregate_by_type(results, "Visibility Reasoning")


def crosspoint_aggregate_correspondence_judgement(results):
    return _aggregate_by_type(results, "Correspondence-Judgement")


def crosspoint_aggregate_correspondence_pointing(results):
    return _aggregate_by_type(results, "Correspondence-Pointing")


def crosspoint_aggregate_level_object(results):
    return _aggregate_by_level(results, "object")


def crosspoint_aggregate_level_part(results):
    return _aggregate_by_level(results, "part")
