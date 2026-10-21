"""EgoTaskQA 切片的 ``process_docs``：只保留"是/否"题。

为什么加这层过滤
----------------
源数据（``tagged_qas.json``）有两个**正交**的字段：

* ``type``       —— 认知类别（descriptive / explanatory / counterfactual / prediction）。
                    三个切片就是按这根轴切的，对应 P0 #13/#22/#24 三项能力。
* ``structural`` —— 答案格式（``verify`` = 是/否，``query`` = 自由文本）。

这两个字段与"答案是否是 yes/no"**完全等价**（2026-10-02 复核）：
verify 题 100% 答案是 yes/no，query 题 0%。

开放题为什么不能要
------------------
``query`` 题的 gold 是一个 **1015 条的封闭动作用语词表**（``Get tomato from
cutting-board``、``on top of juicer base`` 这类），而模型输出的是自然语言
（``Moved it from the cutting board to the bread plate.``）。``utils_qa.py`` 做
归一化精确匹配，必然判错：

* 452 道开放题里，只有 **8.2%** 的回答落在该词表内；
* 四个模型的开放题得分全部塌到 **2–6%**，彼此无区分度；
* 放宽匹配器也救不回来——实测"有词重叠"口径下极差反而反了（3B 最高）。

是/否题不受此影响（答案空间只有 2，题干里的动作位占位符 ``something`` 不影响
判 yes/no 的对错），且实测有真实信号——348 题上：

    flash 61.2% / 27b 58.0% / 0902 57.2% / 3B 54.3%
    相对随机基线 50% 的 z 值：+4.29 / +3.04 / +2.71 / +1.61

过滤后共 **348 题**（counterfactual 81 + event_order 225 + explanatory 42）。

如何恢复全量
------------
设 ``EGOTASKQA_INCLUDE_OPEN=1`` 即关闭本过滤，回到 800 题（开放题会照旧全判 0）。
原始 jsonl 未被改动，全量池仍在 ``data/_full/``。

⚠️ explanatory 子任务过滤后只剩 42 题，且四个模型在该子集上**测不出差异**
（0902 = 52.4%，与常数基线 52.4% 完全重合）——#24 归因理解算"已覆盖"，
但不应解读该项分数。详见 README / 报告注记。
"""

from __future__ import annotations

import os

from loguru import logger as eval_logger

_VERIFY = "verify"


def process_docs_verify_only(dataset):
    """只保留 ``structural == "verify"``（即答案为 yes/no）的题。

    设 ``EGOTASKQA_INCLUDE_OPEN=1`` 可跳过过滤、恢复全量 800 题。
    """
    if os.environ.get("EGOTASKQA_INCLUDE_OPEN"):
        eval_logger.info("EGOTASKQA_INCLUDE_OPEN 已设置：跳过滤，保留全部题（含开放题）")
        return dataset

    before = len(dataset)
    kept = dataset.filter(lambda ex: str(ex.get("structural", "")).strip().lower() == _VERIFY)
    eval_logger.info(
        f"EgoTaskQA：仅保留是/否题 {len(kept)}/{before} 题"
        "（开放题判分不可用，见 utils_filter.py；设 EGOTASKQA_INCLUDE_OPEN=1 可恢复全量）"
    )
    return kept
