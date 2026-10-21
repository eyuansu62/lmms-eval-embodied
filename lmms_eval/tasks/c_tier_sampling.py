"""C 档采样：按类别均衡取题。

背景
----
A/B 档每个 benchmark 控制在 800 题（见 _a_tier_sampling/build_sample.py）。C 档最初
8 个 benchmark 全是**整份数据集直接跑**，合计 27,895 题——是 A/B 单档的 4.7 倍，
一次全量按实测约 30 小时，且四个模型各跑一遍不现实。

为什么必须「按类别」而不能直接取前 N
------------------------------------
C 档数据集是**按类别排序**的，直接取前 N 会切出一堆同质题：

* sparbench 前 800 条只覆盖 3/20 类（正常应 20 类）
* epic_bench 前 800 条只覆盖 5 个 task_type
* 探路时取的 30 题更是全部落在同一个 task 上

所以这里按类别分组、每类取固定条数。「均衡」而非「按自然比例」是有意的：epic 的
TargetLocalization 占 66%，按比例采的话 Manipulation(15%)/Navigation(19%) 会被淹没，
而这两类恰恰是 3B 最弱的（8%~13%），需要足够样本量才有区分度。

⚠️ 「均衡」不等于严格等量
------------------------
某类不足 ``per_category`` 时会把该类**全部**取走，此时各类权重不再相等。
实测（2026-10-05）：

* 其余 7 个任务各类都够取，是严格等量（epic 801 / omni 800 / embspatial 801 /
  multihopspatial 801 / phyx 798 / sparbench 800；robospatial 全量仅 350，
  未挂采样器）
* **crosspoint 例外**：四类为 161 / 220 / 156 / 463，per_category=200 下只能取
  161+200+156+200 = **717**，实际权重 22.5 / 27.9 / 21.8 / 27.9，**不是 25×4**。
  已决定接受现状，不降到 156 去凑严格等量。

**因此 C 档分数的含义是「类别均衡加权下的准确率」，不是「该 benchmark 真实分布下的
水平」。** 两者的差距可达 8 个百分点量级：epic 按自然比例是 31.35%、均衡后 21.18%
（TargetLocalization 从 66% 权重降到 33%）。跨模型横比一律用均衡口径（所有人一致），
自然比例口径目前只有 3B 有全量样本可算。

类内**必须先打乱再取前 N**，不能直接取前 N —— 数据集在类别内部还按与难度相关的轴
成块排序（详见 ``sample_by_category`` 的说明）。doc_id 保留原值，便于和全量结果对照。

如何恢复全量
------------
设 ``LMMS_C_TIER_SAMPLE_FULL=1`` 即关闭采样，跑完整数据集（27,895 题）。
"""

from __future__ import annotations

import os
import random
from typing import Any

import datasets
from loguru import logger as eval_logger


def sampled_indices(vals: list, category_key: str, per_category: int,
                    shuffle_seed: int = 42) -> list[int]:
    """返回采样保留的**原始下标**（升序）。

    单一事实来源：yaml 的 ``process_docs`` 和离线重算脚本都调这里，
    避免两边各写一份采样逻辑而在参数上悄悄分叉。
    """
    keep: list[int] = []
    for cat in dict.fromkeys(vals):  # 保序去重，逐类处理
        idx = [j for j, v in enumerate(vals) if v == cat]
        if len(idx) > per_category:
            # 类内固定种子打乱后取前 N，消除块状排序带来的难度偏斜
            rng = random.Random(f"{shuffle_seed}:{category_key}:{cat}")
            keep.extend(rng.sample(idx, per_category))
        else:
            keep.extend(idx)
    return sorted(keep)


def sample_by_category(dataset: datasets.Dataset, category_key: str, per_category: int,
                       shuffle_seed: int = 42) -> datasets.Dataset:
    """每个 ``category_key`` 类别取 ``per_category`` 条。

    ``per_category`` 传 0 或负数、或设了 ``LMMS_C_TIER_SAMPLE_FULL=1`` 时返回原数据集。

    ⚠️ 类内**必须先打乱再取前 N**，不能直接取前 N。
    C 档的数据集在类别内部还按与难度相关的轴成块排序，实测直接取前 N 会严重偏斜：

    * multihopspatial 是 hop×view 双轴，每个 hop 块内前半是 ego、后半是 exo。
      只按 hop 分层后取前 N，每类都落在 ego 段上 —— 采样集 14.23%，
      而全量是 29.71%，偏差 15.5 个百分点。
    * epic_bench 的 task_type 有 23 类（成 27 段）、omni_spatial 的 sub_task_type
      10 类（成 29 段），同样成块。

    打乱用固定种子，保证可复现；doc_id 仍保留原值，便于和全量结果对照。
    """
    if os.getenv("LMMS_C_TIER_SAMPLE_FULL"):
        eval_logger.info("LMMS_C_TIER_SAMPLE_FULL 已设：跳过采样，跑全量")
        return dataset

    if per_category <= 0:
        eval_logger.warning(f"per_category={per_category} <= 0，未采样")
        return dataset

    if category_key not in dataset.column_names:
        raise KeyError(f"采样键 {category_key!r} 不在数据集列 {dataset.column_names} 中")

    keep = sampled_indices(dataset[category_key], category_key, per_category, shuffle_seed)
    out = dataset.select(keep)
    n_cat = len(set(dataset[category_key]))
    eval_logger.info(
        f"C 档采样：按 {category_key!r} 每类取 {per_category} 条（类内打乱 seed={shuffle_seed}；"
        f"不足则全取），{len(dataset)} -> {len(out)} 题（{n_cat} 类）"
    )
    return out


def make_sampler(category_key: str, per_category: int):
    """生成一个可给 yaml ``process_docs`` 用的偏函数。"""

    def _process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
        return sample_by_category(dataset, category_key, per_category)

    return _process_docs


# 供 yaml 直接引用的具名函数（!function 需要可导入的点路径）
def crosspoint_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_by_category(dataset, "type", 200)


def multihopspatial_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_by_category(dataset, "hop", 267)


def omni_spatial_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_by_category(dataset, "task_type", 200)


def embspatial_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_by_category(dataset, "data_source", 267)


def phyx_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_by_category(dataset, "category", 133)


def epic_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return sample_by_category(dataset, "category", 267)


def sparbench_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """sparbench 原有的 process_docs 只做可选 shuffle，这里串联它再采样。

    sparbench 有 20 个 task、分布最均匀（每类约 5.5%），所以 40/类 = 800 题
    正好覆盖全部 20 类；若直接取前 800 只会落到 3 类上。
    """
    from lmms_eval.tasks.sparbench.utils import process_docs as _shuffle_only

    return sample_by_category(_shuffle_only(dataset), "task", 40)
