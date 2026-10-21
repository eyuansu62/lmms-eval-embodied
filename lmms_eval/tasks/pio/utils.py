"""Point-It-Out (PIO) task utilities for lmms-eval.

Output type: generate_until (free text coordinate)
Scoring: point-in-polygon test — model must output (x, y) that falls
inside the ground-truth polygon.

Images: downloaded from HF repo xavihart/PIO (unofficial mirror) or
from local cache at ~/.cache/lmms_eval/pio/images_s1s2/.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import List

from loguru import logger as eval_logger

CACHE_DIR = pathlib.Path(__file__).resolve().parent / "media" / "images_s1s2"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# HF repo that mirrors PIO images (user uploaded; fallback to GitHub raw)
_HF_REPO = "eyuansu71/PIO"  # set to None to force GitHub-only
_GITHUB_BASE = "https://raw.githubusercontent.com/xavihart/PIO/main/data/images_s1s2"


def _get_image(image_path: str):
    """Load PIO image from local cache or download."""
    from PIL import Image

    cache_file = CACHE_DIR / image_path
    if cache_file.exists():
        return Image.open(cache_file).convert("RGB")

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Try HF mirror first
    if _HF_REPO:
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                _HF_REPO, repo_type="dataset",
                filename=f"images_s1s2/{image_path}",
                local_dir=str(CACHE_DIR.parent),
            )
            return Image.open(local).convert("RGB")
        except Exception:
            pass

    # Fallback: GitHub raw
    import urllib.request
    url = f"{_GITHUB_BASE}/{image_path}"
    urllib.request.urlretrieve(url, str(cache_file))
    return Image.open(cache_file).convert("RGB")


def doc_to_visual(doc):
    try:
        return [_get_image(doc["image_path"])]
    except Exception as exc:
        eval_logger.warning(f"PIO: failed to load {doc['image_path']}: {exc}")
        return []


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre = ""
    # 默认必须与 yaml 的 post_prompt 一致：都要求 0-1 归一化坐标。
    # 两处不一致时，yaml 缺 post_prompt 的模型会拿到「pixel values」提示词，
    # 而判分器按归一化解释 —— 又会退回「量单位猜得准不准」。
    post = (
        "\nProvide the coordinates of the target point as (x, y), where x and y are "
        "NORMALIZED to the range 0 to 1 (0,0 is the top-left corner, 1,1 is the "
        "bottom-right corner), not pixel values. Output only the coordinates."
    )
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", pre)
        post = lmms_eval_specific_kwargs.get("post_prompt", post)
    return f"{pre}Question: {doc['lang']}{post}"


def doc_to_target(doc):
    # Return polygon as JSON string (used only for offline analysis)
    return json.dumps(doc["polygon"])


# ── polygon utils ─────────────────────────────────────────────────────────────

def _flat_to_pairs(flat: list) -> list[tuple[float, float]]:
    """Convert [x1,y1,x2,y2,...] to [(x1,y1),(x2,y2),...]."""
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]


def _point_in_polygon(px: float, py: float, polygon: list[tuple]) -> bool:
    """Ray-casting algorithm."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def _extract_coords_raw(text: str):
    """从模型输出里取出 (x, y)，**不做任何单位解释**，原样返回。

    正则里必须带 ``\\d``：宽松的 ``[\\d.]+`` 能匹配**纯点号**，模型输出
    ``(., .)`` 这类占位坐标时会走到 ``float('.')`` 抛 ``ValueError``，
    把整个任务打断、一条结果都不落盘（2026-10-03 doubao 撞上过）。
    这里再做一层 ``float`` 保护——解析不出来就返回 ``(None, None)``，
    走既有的 ``pio_parse_ok=0`` 兜底路径，而不是让任务崩掉。

    单位解释交给调用方。理由见 ``process_results``：口径必须唯一，
    否则量的是「模型猜没猜中我们的单位」而不是定位能力。
    """
    text = text.strip()
    if not text:
        return None, None

    # 只扫开头一段。坐标总在输出最前面（"[(x, y)]" / "The point is (x, y)"），
    # 但退化输出能长到 max_new_tokens：实测 mimo-v2.6-pro 在 pio_affordance
    # 的 doc 25 上吐了 65538 个 '0'，让下面那条正则跑到灾难性回溯 ——
    # `(\d+\.?\d*|\.\d+)` 两条分支在纯数字串上互相让位，而 re.search 还要对
    # 每个起点各扫一遍，整体 O(n²)，实测 2000 字符就要 49 s、65536 字符永不返回
    # （2026-10-06 那次 pio 单线程空转 9 小时就是卡在这里）。截断把最坏规模钉死。
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

    # Match "(x, y)" or "x, y" or "x y"；数字必须含至少一位
    patterns = [
        rf"\(\s*{num}\s*,\s*{num}\s*\)",
        rf"{num}\s*,\s*{num}",
        rf"{num}\s+{num}",
    ]
    for pat in patterns:
        m = re.search(pat, head)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:  # 理论上不会再有，保守兜底
                continue
    return None, None


def _extract_coords(text: str, width: int, height: int):
    """旧口径的像素坐标提取（[0,1] 小数会放大成像素）。

    仅为重算**改 prompt 之前**的旧 samples 保留。
    新跑的 run 走 ``PIO_COORD_FORMAT=normalized``，不用这个函数。
    """
    x, y = _extract_coords_raw(text)
    if x is None:
        return None, None
    if 0 <= x <= 1 and 0 <= y <= 1:
        x, y = x * width, y * height
    return x, y


def process_results(doc, results):
    response = results[0] if results else ""
    w, h = doc["width"], doc["height"]

    poly_flat = doc["polygon"][0] if doc["polygon"] else []
    poly = _flat_to_pairs(poly_flat)

    # 口径必须唯一。2026-10-06 起提示词改成要求 0-1 归一化坐标
    # （"normalized to [0,1]"），与 RefSpatial 官方 suffix 的做法一致，
    # 判分器只认这一种解释。
    #
    # 为什么不再用旧的 auto（像素与 0-1000 两种都试）：
    # 旧提示词说 "pixel values"，但 API 传图会缩放、模型看不到原图尺寸，
    # 于是各猜各的（qwen3.8-max 吐像素、mimo 吐 0-1000、GLM 两种混用）。
    # auto 给每个模型用它真实的口径，看着公平，实则埋了歧义：
    # 多边形只占图 1.08%，除以 1000 是单调变换、不是随机扰动，
    # 「错误口径侥幸落进多边形」的概率远高于乱猜的 1%（实测 mimo 达 37.5%）。
    # 结果是排名随口径翻转——绝对像素下 glm 41.9% 第一，auto 下 doubao 73.2% 第一。
    # 那种数字说不清是空间能力还是数值巧合，所以改成单一、无歧义的口径。
    scheme = os.environ.get("PIO_COORD_FORMAT", "normalized").lower()

    if scheme == "normalized":
        nx, ny = _extract_coords_raw(response)
        if nx is None:
            return {"pio_accuracy": 0, "pio_parse_ok": 0}
        # 新提示词要求 [0,1]；容忍模型仍回 0-1000（>1 且 <=1000）或像素，按值域归一。
        if max(nx, ny) > 1.0:
            scale = 1000.0 if max(nx, ny) <= 1000.0 else float(max(w, h))
            nx, ny = nx / scale, ny / scale
        cands = [(nx * w, ny * h)]
    else:
        # 复算旧 samples 用：保留 auto / absolute / relative_1000 三种旧口径。
        px, py = _extract_coords(response, w, h)
        if px is None:
            return {"pio_accuracy": 0, "pio_parse_ok": 0}
        if scheme == "absolute":
            cands = [(px, py)]
        elif scheme == "relative_1000":
            cands = [(px / 1000.0 * w, py / 1000.0 * h)]
        else:
            if scheme != "auto":
                eval_logger.warning(
                    f"PIO_COORD_FORMAT 取值非法: {scheme!r}，已回落 auto（可选 normalized/auto/absolute/relative_1000）"
                )
            cands = [(px, py), (px / 1000.0 * w, py / 1000.0 * h)]

    if len(poly) < 3:
        return {"pio_accuracy": 0, "pio_parse_ok": 1}

    hit = max(int(_point_in_polygon(x, y, poly)) for x, y in cands)
    return {"pio_accuracy": hit, "pio_parse_ok": 1}


def aggregate_accuracy(results: List[float]) -> float:
    # lmms-eval 传入的是该 metric 的取值序列，不是 process_results 的字典
    return sum(results) / len(results) if results else 0.0


def aggregate_parse_rate(results: List[float]) -> float:
    return sum(results) / len(results) if results else 0.0
