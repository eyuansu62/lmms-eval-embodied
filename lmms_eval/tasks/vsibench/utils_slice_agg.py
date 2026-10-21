"""VSI-Bench 切片专用的聚合函数。

为什么需要单独一份
------------------
本目录原有的 `utils.py` 里，`_compute_all_subscores` 结尾是这样：

    output["object_rel_direction_accuracy"] = sum([
        output.pop("object_rel_direction_easy_accuracy"),
        output.pop("object_rel_direction_medium_accuracy"),
        output.pop("object_rel_direction_hard_accuracy"),
    ]) / 3.0

它**无条件** pop 那三个键 —— 对**全量** vsibench 成立（所有题型都在），
但本目录的切片每个只含部分题型（如 `rel_distance` 只有 distance 类），
一跑到这里就 `KeyError: 'object_rel_direction_easy_accuracy'`。

实测：3798 题的 `rel_distance` 切片跑完全部推理后，在聚合阶段崩溃，
不产出任何结果。

所以这里另写一份**切片安全**的聚合：
* 只聚合实际出现在结果里的题型
* 对 direction 三类，**存在几个就平均几个**（切片可能只含其中一类或两类）
* 不假设任何题型必须存在

不改动上游 `utils.py`（它是给全量 task 用的），本模块只服务于切片 yaml。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from loguru import logger as eval_logger

# 与上游 utils 保持一致的题型划分
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
    "object_counting",  # 上游把它放在 MCA 里，保持一致
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {"accuracy": "exact_match"}
METRICS_FOR_NA = {"MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)"}

_DIRECTION_TYPES = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
)


def compute_slice_subscores(results: Sequence[dict]) -> dict[str, float]:
    """只聚合结果里实际存在的题型。"""
    df = pd.DataFrame(results)
    if df.empty:
        return {"overall": 0.0}

    output: dict[str, float] = {}
    for question_type, idx in df.groupby("question_type").groups.items():
        sub = df.iloc[idx]
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA:
                if metric in sub:
                    output[f"{question_type}_{metric}"] = float(sub[metric].mean())
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA:
                if metric in sub:
                    output[f"{question_type}_{metric}"] = float(sub[metric].mean())
        else:
            eval_logger.warning(f"切片里出现未预期的 question_type: {question_type}，跳过")

    # direction 汇总：**有几个算几个**，不假设三类都存在
    present = [
        output.pop(f"{d}_accuracy")
        for d in _DIRECTION_TYPES
        if f"{d}_accuracy" in output
    ]
    if present:
        output["object_rel_direction_accuracy"] = sum(present) / len(present)

    output["overall"] = sum(output.values()) / len(output) if output else 0.0
    return output


def vsibench_slice_aggregate_overall(results: Sequence[dict]) -> float:
    output = compute_slice_subscores(results)
    eval_logger.info(f"切片评测结果: {output}")
    return round(output["overall"], 6)


# --- 各题型单独的聚合（可选，供 yaml 里分别报分） ---


def _make_aggregate(metric_key: str):
    def _agg(results: Sequence[dict]) -> float:
        out = compute_slice_subscores(results)
        return round(out.get(metric_key, 0.0), 6)

    return _agg


vsibench_slice_aggregate_object_abs_distance_mra = _make_aggregate("object_abs_distance_MRA:.5:.95:.05")
vsibench_slice_aggregate_object_rel_distance_accuracy = _make_aggregate("object_rel_distance_accuracy")
vsibench_slice_aggregate_object_size_estimation_mra = _make_aggregate("object_size_estimation_MRA:.5:.95:.05")
vsibench_slice_aggregate_room_size_estimation_mra = _make_aggregate("room_size_estimation_MRA:.5:.95:.05")
vsibench_slice_aggregate_object_rel_direction_accuracy = _make_aggregate("object_rel_direction_accuracy")
