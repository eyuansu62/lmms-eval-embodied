"""C 档的「数据集 + 判分器 + 采样参数」规格表 —— 单一事实来源。

为什么必须重算，不能读落盘分数
------------------------------
C 档有 4 个评分 bug 是**静默**的（分数变 0、exit 0、无失败题，汇总表上看不出来）：
crosspoint/epic 的坐标口径、phyx 的裸字母、sparbench 缺 `import partial`。
它们在 `process_results` 阶段，模型回复早已存进 samples，所以**重算即可、不必重跑**。

但磁盘上的逐题字段仍是修复前的旧值 —— 实测（3B 全量样本）：

    phyx_mc     0.00%  -> 29.95%
    epic_bench  0.00%  -> 22.72%
    sparbench   字段取不到 -> 28.21%
    crosspoint  24.20% -> 32.08%

**任何报表都必须用当前判分器重算**，直接读落盘的 `*_accuracy` / `*_score` 会得到
一批自信的错数。

采样参数也放这里
----------------
C 档分数是「类别均衡加权下的准确率」（见 `c_tier_sampling.py` 的说明）。3B 的归档目录
其实是**全量 run**，要拿到与其它模型可比的均衡口径，必须按 `sampled_indices` 筛出采样
子集再算。类别键与每类条数因此是报数的必需参数。

注意 crosspoint 四类为 161/220/156/463，`per_category=200` 只能取 717 题（权重
22.5/27.9/21.8/27.9，不是严格等量）—— 已决定接受现状，见 memory/c-tier-sampling-weighting。
"""

from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger as eval_logger

REPO_ROOT = Path(__file__).resolve().parents[2]

# C 档结果目录 2026-10-05 从 /tmp/fullrun_before_sampling 整体搬到这里
# （被 .gitignore 的 /eval_*/ 覆盖，不入库）。A/B 档不受影响，仍在仓库根。
ARCHIVE_NAME = "eval_C_pre_sampling_fullrun"

# 每个 C 档任务在样本行里取分数的位置：(字段名, 键)。
# 键为 None 表示该字段本身是标量（epic），或键名随 task 变（sparbench，见下）。
VALUE_KEYS: dict[str, tuple[str, str | None]] = {
    "crosspoint_bench": ("crosspoint_accuracy", "hit"),
    "multihopspatial": ("mcq_acc", "mcq_correct"),
    "omni_spatial": ("omni_spatial_score", "score"),
    "robo_spatial_all": ("robo_spatial_score", "score"),
    "sparbench": ("sparbench_score", None),
    "embspatial": ("embspatial_acc", "is_correct"),
    "phyx_mc": ("eval_results", "true_false"),
    "epic_bench": ("epic_accuracy", None),
}

# sparbench 的答案格式随 task 变：选择题在 accuracy、数值题在 MRA、视角题在 vci_metric。
SPARBENCH_VALUE_KEYS = ("accuracy", "MRA:.5:.95:.05", "vci_metric", "score")

# task -> 规格。grader=None 表示该任务不重算（沿用落盘值）。
#   sample_key/per_category：供 c_tier_sampling.sampled_indices 决定采样子集；
#   sample_key=None 表示该任务没挂采样器（robospatial 全量仅 350 题，采不采一样）。
C_TIER_SPECS: dict[str, dict] = {
    "crosspoint_bench": dict(
        dir="eval_C_crosspoint_sampled", dataset="WangYipu2002/CrossPoint-Bench",
        config=None, split="train", sample_key="type", per_category=200,
        module="lmms_eval.tasks.crosspoint_bench.utils", grader="crosspoint_process_results"),
    "multihopspatial": dict(
        dir="eval_C_multihopspatial_sampled", dataset="etri-vilab/MultihopSpatial",
        config="benchmark", split="test", sample_key="hop", per_category=267,
        module="lmms_eval.tasks.multihopspatial.utils", grader="multihopspatial_process_results"),
    "omni_spatial": dict(
        dir="eval_C_omnispatial_sampled", dataset="nv-njb/OmniSpatial-Test",
        config=None, split="test", sample_key="task_type", per_category=200,
        module="lmms_eval.tasks.omni_spatial.utils", grader="omni_spatial_process_results"),
    "sparbench": dict(
        dir="eval_C_sparbench_sampled", dataset="jasonzhango/SPAR-Bench",
        config=None, split="test", sample_key="task", per_category=40,
        module="lmms_eval.tasks.sparbench.utils", grader="sparbench_process_results"),
    "embspatial": dict(
        dir="eval_C_embspatial_sampled", dataset="FlagEval/EmbSpatial-Bench",
        config=None, split="test", sample_key="data_source", per_category=267,
        module="lmms_eval.tasks.embspatial.utils", grader="embspatial_process_results"),
    "phyx_mc": dict(
        dir="eval_C_phyx_sampled", dataset="Cloudriver/PhyX",
        config="data_for_llms_eval", split="phyx_mc", sample_key="category", per_category=133,
        module="lmms_eval.tasks.phyx.utils", grader="phyx_process_results_mc"),
    "epic_bench": dict(
        dir="eval_C_epic_sampled", dataset="rxc205/EPIC-Bench",
        config=None, split="train", sample_key="category", per_category=267,
        module="lmms_eval.tasks.epic_bench.utils", grader="epic_process_results"),
    # robospatial 全量仅 350 题，未挂采样器 —— 沿用落盘值即可
    # （mask_path 来自 process_docs 的临时文件，离线重算拿不到，所以不重算）。
    "robo_spatial_all": dict(
        dir="eval_C_robospatial_sampled", dataset="chanhee-luke/RoboSpatial-Home",
        config=None, split="combined", sample_key=None, per_category=0,
        module=None, grader=None),
}

# build_report.py 的 C 档目录词干（连写）-> 任务名
BENCH_TO_TASK: dict[str, str] = {
    "crosspoint": "crosspoint_bench",
    "multihopspatial": "multihopspatial",
    "omnispatial": "omni_spatial",
    "robospatial": "robo_spatial_all",
    "sparbench": "sparbench",
    "embspatial": "embspatial",
    "phyx": "phyx_mc",
    "epic": "epic_bench",
}

_ROBO_DATASET_KWARGS = dict(
    data_files={"combined": ["data/compatibility-*.parquet",
                             "data/configuration-*.parquet",
                             "data/context-*.parquet"]},
    verification_mode="no_checks",
)


def is_c_task(task: str) -> bool:
    return task in C_TIER_SPECS


def resolve_dir(dir_name: str) -> Path | None:
    """按「仓库根 -> C 档归档根」的顺序找结果目录，找不到返回 None。"""
    for root in (REPO_ROOT, REPO_ROOT / ARCHIVE_NAME):
        p = root / dir_name
        if p.is_dir():
            return p
    return None


_DATASET_CACHE: dict[str, tuple] = {}
_SAMPLE_IDX_CACHE: dict[str, list[int]] = {}


def load_task_data(task: str):
    """返回 ``(dataset, grader)``；不重算的任务返回 ``(None, None)``。结果按 task 缓存。"""
    if task in _DATASET_CACHE:
        return _DATASET_CACHE[task]

    spec = C_TIER_SPECS[task]
    if spec["grader"] is None:
        _DATASET_CACHE[task] = (None, None)
        return _DATASET_CACHE[task]

    from datasets import load_dataset

    if spec["split"] == "combined":
        ds = load_dataset(spec["dataset"], **_ROBO_DATASET_KWARGS, split="combined")
    elif spec["config"]:
        ds = load_dataset(spec["dataset"], spec["config"], split=spec["split"])
    else:
        ds = load_dataset(spec["dataset"], split=spec["split"])

    grader = getattr(importlib.import_module(spec["module"]), spec["grader"])
    _DATASET_CACHE[task] = (ds, grader)
    return _DATASET_CACHE[task]


def sampled_original_indices(task: str, ds) -> set[int] | None:
    """采样集对应的**原始下标**集合；返回 None 表示该任务不采样（全取）。

    doc_id 语义：全量 run 的 doc_id = 原始位置；采样 run 的 doc_id = 采样后位置。
    采样规则只有一处实现，这里直接复用 ``c_tier_sampling.sampled_indices``，
    避免各脚本另写一份而悄悄分叉。
    """
    spec = C_TIER_SPECS[task]
    if spec["sample_key"] is None:
        return None
    return set(_sorted_sampled_indices(task, ds))


def _sorted_sampled_indices(task: str, ds) -> list[int]:
    """采样保留的原始下标，**升序**。

    顺序很关键：``c_tier_sampling.sample_by_category`` 先 ``keep.sort()`` 再
    ``dataset.select(keep)``，所以采样后的数据集是按原始顺序排的，紧凑下标 i 恰好
    对应 ``sorted_indices[i]``。下面是这个映射的唯一依据。
    """
    if task in _SAMPLE_IDX_CACHE:
        return _SAMPLE_IDX_CACHE[task]
    spec = C_TIER_SPECS[task]
    from lmms_eval.tasks.c_tier_sampling import sampled_indices

    out = sorted(sampled_indices(ds[spec["sample_key"]], spec["sample_key"], spec["per_category"]))
    _SAMPLE_IDX_CACHE[task] = out
    return out


# 各任务用来做「口径指纹」的文本列。取不到就没法指纹判定，退回行数启发式。
# epic 没有 question 列（题面在 text_label），所以必须按任务指定，否则空串会恒真匹配。
_FINGERPRINT_FIELD = {
    "crosspoint_bench": "question",
    "multihopspatial": "question",
    "omni_spatial": "question",
    "sparbench": "question",
    "embspatial": "question",
    "phyx_mc": "question",
    "epic_bench": "text_label",
    "robo_spatial_all": "question",
}


def fingerprint_convention(task: str, ds, rows: list, sample: int = 25) -> str | None:
    """用样本行里的 ``input`` 文本对拍数据集，判出 doc_id 口径；判不了返回 None。

    这是**主判据**，比行数启发式可靠得多：行数无法区分「20 行恰好是紧凑下标的 0..19」
    还是「原始下标的 0..19」（探路 run 就是这种歧义，行数判定只能给 unknown）。

    ⚠️ 只有 ``idx[did] != did`` 的行才携带信息
    ------------------------------------------
    采样集在**前缀上常常与原始下标重合**：crosspoint 第一类 161 题全保留（161 < 200），
    所以 ``idx[0..160] == 0..160``。这些行上两套假设指向**同一道题**，对拍必然两边都命中，
    是「无信息」行，不能拿来判等价。

    2026-10-06 踩过：只在这样的前缀上判等价，得出「口径等价 -> original」，而实际是
    采样 run（``idx[716] == 999`` 处分叉）。按 original 解释会拿 ``ds[did]`` 而不是
    ``ds[idx[did]]`` 去索引 —— **取到错误的题，分数全错且不报错**。
    这个错误结论还会**抢在**行数启发式之前返回（而后者本可判对）。

    规则：
    * 存在有信息的行 -> 只在这些行上对拍，谁命中多用谁
    * 无有信息的行（映射处处恒等）-> 两套口径真的等价，判 ``original``
    """
    field = _FINGERPRINT_FIELD.get(task)
    if field is None:
        return None

    spec = C_TIER_SPECS[task]
    idx = _sorted_sampled_indices(task, ds) if spec["sample_key"] else list(range(len(ds)))

    def probe(i):
        try:
            v = ds[i].get(field)
        except Exception:
            return ""
        return str(v).strip()[:60] if v else ""

    n_orig = n_comp = n_info = 0
    for r in rows:
        if n_info >= sample:
            break
        txt = r.get("input")
        did = r.get("doc_id")
        if not isinstance(txt, str) or not txt or not isinstance(did, int):
            continue
        if not (0 <= did < len(idx)):
            continue
        real = idx[did]
        if real == did:
            continue  # 无信息：两套假设指向同一题
        n_info += 1
        po = probe(did)
        pc = probe(real)
        if po and po in txt:
            n_orig += 1
        if pc and pc in txt:
            n_comp += 1

    if n_info == 0:
        # 映射处处恒等，两套口径等价
        return "original"
    if n_info < 3 or (n_orig == 0 and n_comp == 0):
        return None
    if n_orig > n_comp:
        return "original"
    if n_comp > n_orig:
        return "compact"
    return None


def doc_id_convention(task: str, ds, n_rows: int, rows: list | None = None) -> str:
    """判定某份 samples 文件的 ``doc_id`` 是哪套口径。

    ⚠️ 归档里**并存两种口径**（2026-10-05 补 27b 数据时发现）：

    * ``"original"`` —— ``doc_id`` 就是数据集里的原始下标。
      跑在采样开关挂上 yaml **之前**的 run 都属于这种：归档里 3B 全部、27b 的
      crosspoint/multihopspatial、以及 4 个模型的 20~30 行探路 run。
    * ``"compact"`` —— ``doc_id`` 是采样后的**紧凑下标** 0..n-1。
      依据：``api/task.py`` 里是 ``enumerate(self.eval_docs_no_media)``，而
      ``dataset_no_image`` 派生自 ``process_docs`` **之后**的数据集，所以过滤后的
      数据集重新编号。直接拿紧凑下标去索引数据集会取到**错误的题**。

    先试**指纹判定**（``rows`` 传入时，看 ``input`` 与哪套题面对拍得上）；指纹判不了
    才退回行数启发式。行数启发式在探路 run 上必然给 ``unknown``，所以尽量传 ``rows``。

    返回 ``"unknown"`` 表示无法判定，调用方**必须跳过**该任务，而不是冒险取错题、
    也不是回落到落盘值（那是修复前判分器写的，见模块 docstring）。
    """
    spec = C_TIER_SPECS[task]
    if spec["sample_key"] is None:
        return "original"  # 未挂采样器，无歧义

    n_full = len(ds)
    n_sample = len(_sorted_sampled_indices(task, ds))
    if n_sample >= n_full:
        return "original"  # 采样没真正生效，两套口径等价

    if rows:
        fp = fingerprint_convention(task, ds, rows)
        if fp is not None:
            return fp

    # 指纹不可用（老样本没有 input 字段等）才走行数
    if n_rows == n_full:
        return "original"
    if n_rows == n_sample:
        return "compact"
    if n_rows > n_sample:
        return "original"
    eval_logger.warning(f"{task}: samples 行数 {n_rows}（全量 {n_full} / 采样 {n_sample}）与指纹都判不出口径，跳过重算")
    return "unknown"


def original_index_map(task: str, ds, n_rows: int) -> list[int] | None:
    """``doc_id -> 数据集原始下标`` 的映射表；``None`` 表示 doc_id 本身就是原始下标。

    调用方用它把样本还原到数据集里的题：``ds[mapping[doc_id]]``。
    """
    if doc_id_convention(task, ds, n_rows) != "compact":
        return None
    return _sorted_sampled_indices(task, ds)


def dig_value(out, field: str, key: str | None) -> float | None:
    """从 ``process_results`` 的返回里取出**该题的标量分**（bool -> 1.0/0.0）。

    这是唯一一处取分数的实现，报表脚本共用，防止各处口径不一致。
    """
    if not isinstance(out, dict):
        return None
    v = out.get(field)
    if v is None:
        return None
    if isinstance(v, dict):
        if key is None:
            for k in SPARBENCH_VALUE_KEYS:
                if k in v:
                    key = k
                    break
            else:
                return None
        v = v.get(key)
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) else None
