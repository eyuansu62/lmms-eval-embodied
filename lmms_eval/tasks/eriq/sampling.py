"""ERIQ · 采样工厂（A 档「切法 A」口径的运行时落实）。

背景
----
A 档的预算是 `_a_tier_sampling/build_sample.py` 定的：**每个 benchmark 800 题**，
按切片规模等比分配（最大余数法、下限 50、种子 20260926）。

但那个脚本只处理**仓库内 `data_files` 指向的 .jsonl**（它靠解析 yaml 里的
`data_files:` 找数据文件）。A 档里走 HF 的那几个都没被它采样 —— 实测题数：
momagraph 817 / negvqa 182 / refspatial 7，都是各自的自有规模，不是 800。

ERIQ 也是 HF 型，所以 800 的分配在这里、由 `process_docs` 在**运行时**落实。

分配表从哪来
------------
按 `build_sample.py` 的同一套规则，对 ERIQ 的 8 个子任务（= 8 个切片）跑一遍
「切法 A」得到，逐位一致：

    #14 指令具化        eriq_human_intention     225 ->  56
    #25 失败定位与原因分析 eriq_mistake_classify     116 ->  50
    #27 单体/多体状态理解  eriq_action_understanding 600 -> 148
    #28 子任务分解       eriq_subtask_planning    600 -> 148
    #32 进度估计        eriq_task_progress       281 ->  69
    #33 失败检测        eriq_mistake_existence   334 ->  82
    #33 失败检测        eriq_success_detection   800 -> 197
    #34 恢复策略        eriq_mistake_recovery    105 ->  50
                                         合计  3061 -> 800

⚠️ 采样后单任务 n 落在 50~197，单任务准确率的 95% 置信半宽约 ±7~14 个百分点。
报数时必须带上这个前提，别把小任务的波动当结论。
"""

from __future__ import annotations

import random

import datasets

# 子任务 -> 采样后题数。改动前先重跑一遍「切法 A」，别手改。
SAMPLE_SIZES: dict[str, int] = {
    "eriq_human_intention": 56,
    "eriq_mistake_classify": 50,
    "eriq_action_understanding": 148,
    "eriq_subtask_planning": 148,
    "eriq_task_progress": 69,
    "eriq_mistake_existence": 82,
    "eriq_success_detection": 197,
    "eriq_mistake_recovery": 50,
}

# 与 build_sample.py 同一个种子，便于复现与对照
SEED = 20260926


def sample_to(dataset: datasets.Dataset, n: int, seed: int = SEED) -> datasets.Dataset:
    """从 dataset 里固定种子随机抽 n 条；n >= 总长时原样返回。

    用 `datasets.Dataset.shuffle(seed=...).select(...)` 而不是 `random.sample`：
    前者对 Arrow 表是惰性/零拷贝的，后者会把整表拉进内存（ERIQ 单文件最大
    800 行不算大，但这个 task 会与其它 A 档任务同进程加载，保持轻量）。
    """
    if n is None or n <= 0 or n >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(n))


# ── yaml 里 !function 引用的具名工厂 ───────────────────────────────────────
# 每个都必须存在，否则 process_docs 解析失败、task 直接加载不起来。


def eriq_human_intention_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_human_intention"])


def eriq_mistake_classify_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_mistake_classify"])


def eriq_action_understanding_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_action_understanding"])


def eriq_subtask_planning_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_subtask_planning"])


def eriq_task_progress_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_task_progress"])


def eriq_mistake_existence_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_mistake_existence"])


def eriq_success_detection_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_success_detection"])


def eriq_mistake_recovery_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_to(dataset, SAMPLE_SIZES["eriq_mistake_recovery"])
