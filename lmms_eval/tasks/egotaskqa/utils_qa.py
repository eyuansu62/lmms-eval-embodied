"""EgoTaskQA 自由文本问答的评测逻辑。

为什么不用内置 utils
--------------------
本目录原有的 ``utils.py`` 是给 ``nv-njb/EgoTaskQA-MCQ`` 写的 MCQ 判分
（要求 ``q`` / ``option`` / ``a`` 三个字段）。但实测：

* **官方数据**（``eyuansu71/EgoTaskQA`` 的 ``data/qa/*/test_qas.json``）字段为
  ``question / type / answer / interval / ...``，答案是**自由文本短语**，**没有选项**。
* ``nv-njb`` 的 MCQ 版**答案键与官方不一致**：题干文本能对上 92.6%，
  但其中 53.7% 的官方答案**不在 MCQ 的任何选项里**；能对上的部分，
  字母一致率也只有 172/174。

  例：官方答案 ``'no'`` vs MCQ 选项 C ``'yes'``；
      官方答案 ``'sink'`` vs MCQ 选项 D ``'knife'``。

所以两者**不能 join**，MCQ 版是自选干扰项的再加工，答案已被改动。

本模块改为对官方格式做**自由文本判分**。

判分方式
--------
归一化后精确匹配：

1. 转小写、去标点、压空白
2. 去掉冠词（a/an/the）
3. 与答案集精确比较

抽样核对：切片里 293 种不同答案**全部命中官方 answer_set**（800/800），
说明答案形态规范（``'yes'`` / ``'in microwave'`` / ``'on top of table'``
这类短短语），机械判分可行。

**局限**：语义等价但措辞不同会判错（如 ``'on the table'`` vs
``'on top of table'``）。这是严格判分，与官方脚本口径一致。
若要放宽，需引入 LLM judge —— 那是另一套设计，此处不做。

证据缺口（如实说明）
--------------------
本模块的判分正确性**无法在本轮独立验证** —— 需要看视频才能确认答案对错。
已验证的只是：答案形态规范、全部在官方答案集内、归一化逻辑对构造样例正确。
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger as eval_logger

_ARTICLES = {"a", "an", "the"}


def video_dir() -> str:
    """视频目录。

    优先 ``EGOTASKQA_VIDEO_DIR``，其次项目内 ``media/videos/``（离线部署），
    最后回退 ``~/.cache/lmms_eval/egotaskqa/videos/``。
    视频文件名与 ``interval`` 字段一致（``<interval>.mp4``）。
    """
    override = os.environ.get("EGOTASKQA_VIDEO_DIR")
    if override:
        return override
    # 与同目录 utils.py 保持一致：优先项目内 media 目录，回退 ~/.cache
    local = Path(__file__).resolve().parent / "media" / "videos"
    if local.exists():
        return str(local)
    return str(Path.home() / ".cache" / "lmms_eval" / "egotaskqa" / "videos")


def doc_to_visual(doc: Mapping[str, Any]) -> list[str]:
    """视频路径。切片里用 ``interval`` 字段（等价于官方的视频标识）。"""
    ident = str(doc.get("interval") or doc.get("video_path") or "").strip()
    if ident.endswith(".mp4"):
        ident = ident[:-4]
    path = os.path.join(video_dir(), f"{ident}.mp4")
    if not os.path.exists(path):
        eval_logger.warning(f"视频缺失: {path}（可设 EGOTASKQA_VIDEO_DIR 覆盖）")
    return [path]


def doc_to_text(doc: Mapping[str, Any], lmms_eval_specific_kwargs: Mapping[str, str] | None = None) -> str:
    """官方是自由文本答案，故不提供选项，要求答短语。"""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get(
        "post_prompt",
        "\nAnswer with a short phrase only, without extra explanation.",
    )
    q = str(doc.get("question") or "").strip()
    return f"{pre}Based on the video, answer the following question.\n{q}{post}"


def doc_to_target(doc: Mapping[str, Any]) -> str:
    return str(doc.get("answer") or "").strip()


def normalize(text: Any) -> str:
    """归一化：小写、去标点、去冠词、压空白。"""
    s = str(text or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)          # 标点 -> 空格
    s = re.sub(r"\s+", " ", s)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _strip_answer_prefix(text: str) -> str:
    """去掉模型可能的套话前缀（"The answer is X" / "Answer: X"）。"""
    s = str(text or "").strip()
    s = re.sub(r"^(?:the\s+)?answer\s*(?:is|:)\s*", "", s, flags=re.I)
    s = re.sub(r"^(?:it\s+is|there\s+is|this\s+is)\s+", "", s, flags=re.I)
    return s.strip()


def process_results(doc: Mapping[str, Any], results: Sequence[str]) -> dict[str, float]:
    raw = results[0] if results else ""
    pred = normalize(_strip_answer_prefix(raw))
    target = normalize(doc.get("answer"))
    return {"egotaskqa_accuracy": 1.0 if pred == target else 0.0}


def aggregate_results(results: Sequence[float]) -> float:
    return sum(results) / len(results) if results else 0.0
