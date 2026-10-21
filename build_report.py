#!/usr/bin/env python3
"""扫描磁盘上的评测结果，汇总成 模型 × benchmark 矩阵（含/剔除失败题两个口径）。

用法:
    python build_report.py            # 全部模型
    python build_report.py --json     # 输出机器可读 JSON

口径说明:
  - 每个 (目录, 子任务) 只取时间戳最新的一次运行；_merged 目录优先于裸目录。
  - 逐题分数取样本记录里的数值型指标字段 / dict 里的 score|accuracy|acc。
  - "含失败" 把 LMMS_EVAL_REQUEST_FAILED 当 0 分计入；"剔失败" 只在成功题上平均。
  - benchmark 总分 = 各子任务分的算术平均（与 lmms-eval 的 overall 一致）。
  - vsibench 额外给出官方口径（按题型分组）供对照。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

from lmms_eval.tasks import c_tier_specs as ct_specs

FAIL_MARK = "[LMMS_EVAL_REQUEST_FAILED"
# C 档的结果目录 2026-10-05 从 /tmp/fullrun_before_sampling 整体搬进了
# eval_C_pre_sampling_fullrun/（归档名只在 c_tier_specs.ARCHIVE_NAME 存一份，
# 走 ct_specs.resolve_dir 解析）。仓库根已无 eval_C_*；A/B 档不受影响，仍在仓库根。
# C 档「疑似只有探路样本」的阈值。27b / doubao 的 C 档有 6 个任务只跑过 30 题探路
# （doc_id 0~29），而采样集是散布的，交集只剩 7~18 条 —— 这时算出来的百分比
# （常见 100.00%）是几道题的巧合，不是结论。低于此阈值就在表里标出来。
C_MIN_N = 100
META = {
    "id", "dataset", "scene_name", "question_type", "question", "ground_truth",
    "options", "prediction", "answer", "idx", "pred_answer", "sub_type",
    "ability_type", "query", "pred", "parsed_points",
}
NON_METRIC = {
    "doc_id", "target", "filtered_resps", "token_counts", "doc_hash", "input",
    "input_media", "resps", "arguments", "doc",
}
CACHE: dict[pathlib.Path, list[tuple[str, dict]]] = {}

# C 档每条样本只认一个指标，且分数藏在 dict 的不同字段里（甚至有的是布尔）。
# (metric 字段名, 取分数的键) —— 键为 None 表示该字段本身就是标量。
C_PRIMARY: dict[str, tuple[str, str | None]] = {
    "crosspoint_bench": ("crosspoint_accuracy", "hit"),
    "multihopspatial": ("mcq_acc", "mcq_correct"),
    "omni_spatial": ("omni_spatial_score", "score"),
    "robo_spatial_all": ("robo_spatial_score", "score"),
    # sparbench 的答案格式随 task 变：选择题在 accuracy、数值题在 MRA、视角题在 vci_metric
    "sparbench": ("sparbench_score", None),
    "embspatial": ("embspatial_acc", "is_correct"),
    "phyx_mc": ("eval_results", "true_false"),
    "epic_bench": ("epic_accuracy", None),
}

# A/B 档里同样「一题多指标」的任务：只取分数那一个。
#
# pio 的每条样本带两个数值字段：pio_accuracy（分数）与 pio_parse_ok（解析成功率，
# 诊断用）。value() 的兜底逻辑会把它们一起平均，而 parse_ok 恒为 1.0，
# 于是报出来的分数变成 (accuracy + 1.0) / 2 —— 恒定虚高，且越低分虚高越多：
#   真实 0.161 → 报 0.581        真实 0.573 → 报 0.786
# 这就是 2026-10-06 之前 PIO 那列数字偏高的原因，与坐标口径无关。
AB_PRIMARY: dict[str, tuple[str, str | None]] = {
    "pio_affordance": ("pio_accuracy", None),
    "pio_attribute_reference": ("pio_accuracy", None),
    "pio_safety": ("pio_accuracy", None),
    # #20 序数指代的 PIO 子集：不单独跑，分数从 pio_attribute_reference 的样本里
    # 按 doc_id 取（见 dump_reports._pio_ordinal_scores）。这里登记是为了让
    # 万一将来单独跑时也走同一个取分口径。
    "pio_ordinal_reference": ("pio_accuracy", None),
    # ERIQ：每条样本同时落 eriq_accuracy（官方严格口径）与
    # eriq_accuracy_lenient（宽松口径）两个字段。不在这里指定主指标，
    # value() 就会走「非指标字段求平均」的兜底，把两个口径平均成一个
    # 无意义的中间值。主指标取严格口径（与 README 的 baseline 可比）。
    #
    # ⚠️ key 必须是 "is_correct"，**不能写 None**：None 的含义是「该字段本身
    # 就是标量」，而 eriq_accuracy 是个 dict
    #   {"id":…, "ground_truth":…, "pred_strict":…, "is_correct": True/False, …}
    # 分数载体是其中的**布尔** is_correct。写 None 会让 _scalar 拿到 dict 后
    # 找不到 score/accuracy/acc，再兜底时又因「排除布尔」而放弃 → 每题都返回
    # None → 整列空白、n=0（2026-10-12 就是这么把 eriq 那行写空的）。
    # RoboBench 规划题：judge 写回的 *_judged_samples_* 里每题带 rb_openended_score（标量）
    # 和 rb_judge（明细 dict，内含 node_correctness 等数值）。不指定主指标的话兜底逻辑会把
    # 明细里的数也平均进去。负值是「待判」，在 value() 里剔掉。
    **{t: ("rb_openended_score", None) for t in (
        ["robobench_implicit_goal", "robobench_implicit_goal_q1", "robobench_explicit_goal", "robobench_number"]
        + [f"robobench_{v}_{s}" for v in ("plan", "plan20") for s in (
            "single_arm", "dual_arm", "human", "mobile_manipulation",
            "articulated", "deformable", "multi_object", "rigid", "special",
            "color", "number", "shape", "size", "world_knowledge",
            "multi_view", "single_view",
        )]
    )},
    **{t: ("eriq_accuracy", "is_correct") for t in (
        "eriq_human_intention", "eriq_mistake_classify", "eriq_action_understanding",
        "eriq_subtask_planning", "eriq_task_progress", "eriq_mistake_existence",
        "eriq_success_detection", "eriq_mistake_recovery",
    )},
    # ASIMOV：每题同一个 entry dict 挂在多个指标名下（accuracy / balanced / 两边召回 /
    # 解析率），不指定主指标就会被兜底逻辑平均成无意义的数。分数载体同 ERIQ，是布尔 is_correct。
    **{t: ("asimov_accuracy", "is_correct") for t in (
        "asimov_constraints", "asimov_vla_feasibility", "asimov_uncertainty",
        "asimov_gauge_readable", "asimov_gauge_unreadable",
    )},
    # ASIMOV 安全监控（#36）：一题挂 4 个指标，主分取**四类等权**的 balanced
    # （105/21/21/21 失衡，按题数平均会被 high_stop 淹没）
    "asimov_safety": ("asimov_safety_balanced", "is_correct"),
    "asimov_hsm": ("asimov_hsm_balanced", "is_correct"),  # 停/继续各 200，逐题平均 = 平衡准确率
    # EAI：entry 挂在 8 个指标名下，且同时带 tp/fp/fn 计数（整数，会被兜底逻辑误当分数）。
    # 逐题分数取该题 F1；⚠️ 官方榜单是 micro F1，逐题平均与之不等（见 tasks/eai/utils.py）。
    "eai_behavior_goal": ("eai_f1", "f1"),
    "eai_virtualhome_goal": ("eai_f1", "f1"),
    # 动作序列（#31）：entry 是 utils_as.score 的明细 dict，主分取布尔 executable
    "eai_action_seq": ("eai_as_executable", "executable"),
    # AQuA：judge 判出的策略等级是否等于真值；judge 失败的题 score=None，不计
    **{f"aqua_level{i}": ("aqua_strategy_accuracy", "score") for i in range(4)},
    # OpenEQA：judge 判「与标准答案语义是否等价」；judge 失败的题 score=None，不计
    "openeqa_world_knowledge": ("openeqa_accuracy", "score"),
    "openeqa_functional_reasoning": ("openeqa_accuracy", "score"),
    # IntPhys 2：entry 挂在 5 个指标名下；主分 = 官方 Yes/No 子串口径的 is_correct（无效回复计错）
    "intphys2_permanence": ("intphys2_accuracy", "is_correct"),
    "intphys2_continuity": ("intphys2_accuracy", "is_correct"),
    "corecognition_conservation": ("corecognition_accuracy", "is_correct"),
    "corecognition_permanence": ("corecognition_accuracy", "is_correct"),
    "tvbench_cap13_action_sequence": ("tvbench_cap13_accuracy", "is_correct"),
    "tvbench_cap13_egocentric_sequence": ("tvbench_cap13_accuracy", "is_correct"),
    "vlm4d_cap37_ego": ("vlm4d_cap37_accuracy", "is_correct"),
    "vlm4d_cap37_exo": ("vlm4d_cap37_accuracy", "is_correct"),
}

# ERIQ 的第三个口径（Yes/No → 字母归一），仅在 summary_subtasks.csv 里出列，
# **不替换 AB_PRIMARY**：主表仍用官方严格口径，与 README 的 6 个 baseline 可比。
# 用法说明见 utils.eriq_aggregate_accuracy_yn 的 docstring。
ERIQ_YN_TASKS = (
    "eriq_human_intention", "eriq_mistake_classify", "eriq_action_understanding",
    "eriq_subtask_planning", "eriq_task_progress", "eriq_mistake_existence",
    "eriq_success_detection", "eriq_mistake_recovery",
)
AB_PRIMARY_YN: dict[str, tuple[str, str | None]] = {
    t: ("eriq_accuracy_yn", "yn_correct") for t in ERIQ_YN_TASKS
}


def _scalar(v) -> float | None:
    """从一个指标值里取出数。dict 型取 score/accuracy/acc，否则取唯一的数值字段。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for key in ("score", "accuracy", "acc"):
            if isinstance(v.get(key), (int, float)) and not isinstance(v.get(key), bool):
                return float(v[key])
        rest = [x for kk, x in v.items()
                if kk not in META and isinstance(x, (int, float)) and not isinstance(x, bool)]
        if rest:
            return float(rest[0])
    return None


def _c_value(task: str, r: dict) -> float | None:
    """C 档逐题分数。

    每个任务的 lmms-eval metric 只取一个字段，且藏在 dict 里的键各不相同；有的还是
    布尔（embspatial 的 is_correct、multihopspatial 的 mcq_correct）。统一归一化到
    0-1，让外围的 *100 显示口径对得上官方数字。

    不要退回 _scalar 的兜底逻辑：它会挑到第一个数值字段，于是
    multihopspatial 取到 iou（6.46%，官方 mcq_acc 是 29.71%），
    sparbench 取到该题的 MRA 而漏掉 accuracy。
    """
    field, key = C_PRIMARY[task]
    d = r.get(field)
    if not isinstance(d, dict):
        return _scalar(r.get(field))
    if key is None:  # 每题字段名不固定：按优先级挑第一个存在的
        for k in ("accuracy", "MRA:.5:.95:.05", "vci_metric", "score"):
            if k in d:
                key = k
                break
        else:
            return None
    v = d.get(key)
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) else None


def _yn_value(task: str, r: dict) -> float | None:
    """ERIQ 的 Yes/No 归一口径，取一条样本的分数。

    优先读落盘字段 `eriq_accuracy_yn.yn_correct`（2026-10-14 之后跑的都有）。
    但**已有的 10 个模型结果都在这之前跑完**，样本里没有这个键，所以要能回退：
    用 `utils` 的同一套逻辑现场重算。回退走的是与评测时**完全相同**的函数，
    不是另写一份近似实现 —— 否则两个口径会慢慢漂移。
    """
    from lmms_eval.tasks.eriq import utils as _eriq

    e = r.get("eriq_accuracy")
    if not isinstance(e, dict):
        return None

    yn = r.get("eriq_accuracy_yn")
    if isinstance(yn, dict) and "yn_correct" in yn:
        x = yn.get("yn_correct")
        if isinstance(x, bool):
            return float(x)

    # 回退：旧样本没有第三口径，现场重算。
    gt = str(e.get("ground_truth", "")).strip()
    is_yes_no = gt.lower() in ("yes", "no")
    resp = str(r.get("filtered_resps", ""))
    # 失败占位符不该被当成回复 —— 判 0（与「含失败」口径一致）。
    if resp.startswith(FAIL_MARK):
        return 0.0
    pred_strict = _eriq.extract_official(resp, is_yes_no)
    pred_yn, converted = _eriq.extract_yn_normalized(resp, gt, str(r.get("input", "")))
    gt_cmp = gt.lower() if is_yes_no else gt.upper()

    def _cmp(p: str | None) -> bool:
        if p is None:
            return False
        return (p.lower() if is_yes_no else p.upper()) == gt_cmp

    return float(_cmp(pred_yn) if converted else _cmp(pred_strict))


def value(task: str, r: dict) -> float | None:
    """取一条样本的分数。

    vsibench 的每条记录同时带 9 个指标，只用 vsibench_overall；C 档每条只该看一个
    指标、且分数藏在 dict 的不同字段里（见 C_PRIMARY）；A/B 档里 pio 同时带分数与
    解析率两个数值字段（见 AB_PRIMARY）。其余 benchmark 每条只有 1 个指标，
    取全部候选的平均（通常就等于那一个）。
    """
    if task in C_PRIMARY:
        return _c_value(task, r)
    # ⚠️ **不要**在这里加 AB_PRIMARY_YN 的分支。主表（summary_A_B.csv /
    # summary_subtasks.csv）必须保持官方严格口径，才能与 README 的 6 个 baseline
    # 对读；yn 口径只出现在 summary_eriq_calibers.csv 里，由 dump_reports 直接
    # 调 _yn_value 取数。2026-10-14 我把 yn 分支插在这里过，结果主表被整列换成了
    # yn 分数（qwen3.8-max 88.69→88.75、deepseek 53.36→62.56），且因为两个 dict
    # 的键完全相同、分支在前者抢先命中，表面上完全看不出来。
    if task in AB_PRIMARY:
        field, key = AB_PRIMARY[task]
        v = r.get(field)
        if key is None:
            x = _scalar(v)
            # RoboBench 规划题的 -1 = 未经 judge 判分（Q1/Q2 评测时只占位），不是 0 分
            return None if field == "rb_openended_score" and x is not None and x < 0 else x
        # key 非 None：指定了字段内部的取分键。**布尔要当 0/1 收**——
        # ERIQ 的每篇条目是 {"is_correct": True/False, ...}，分数载体就是布尔，
        # 而 _scalar 的兜底逻辑按设计排除布尔（防它把 is_yes_no 之类当分数）。
        # 这里是指名道姓要 is_correct，没有误取风险，所以显式放开。
        # 与 _c_value（C 档 embspatial 的 is_correct）同一处理。
        if not isinstance(v, dict):
            return None
        x = v.get(key)
        if isinstance(x, bool):
            return float(x)
        return float(x) if isinstance(x, (int, float)) else None
    if "vsibench_overall" in r:
        return _scalar(r["vsibench_overall"])
    vals = [_scalar(v) for k, v in r.items() if k not in NON_METRIC]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load(d: pathlib.Path) -> list[tuple[str, dict]]:
    if d in CACHE:
        return CACHE[d]
    best: dict[str, pathlib.Path] = {}
    best_ts: dict[str, str] = {}
    for f in glob.glob(str(d / "**" / "*_samples_*.jsonl"), recursive=True):
        p = pathlib.Path(f)
        task = p.name.split("_samples_")[-1][: -len(".jsonl")]
        ts = p.name.split("_samples_")[0]
        # 时间戳要和时间戳比。之前拿 ts 去比 best[task].name（完整文件名），时间戳不同时
        # 碰巧正确；同一时间戳的 "<ts>_judged_samples_x" 与 "<ts>_samples_x" 则取决于 glob 顺序。
        if task not in best or ts > best_ts[task]:
            best[task], best_ts[task] = p, ts
    out = []
    for task, p in best.items():
        with open(p, errors="replace") as fh:
            for line in fh:
                if line.strip():
                    out.append((task, json.loads(line)))
    CACHE[d] = out
    return out


MCA_Q = {
    "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard", "object_rel_distance",
    "route_planning", "obj_appearance_order",
}
NA_Q = {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}
NA_METRIC = "MRA:.5:.95:.05"


def vsibench_official(rows: list[tuple[str, dict]]) -> dict:
    """复刻 lmms_eval/tasks/vsibench/utils.py 的 _compute_all_subscores + overall。

    按 question_type 分组各取该题型的指标均值（MCA 用 accuracy、NA 用 MRA），
    再把 object_rel_direction 的 easy/medium/hard 三类合并，最后对题型求平均。
    未跑到的题型（object_counting / obj_appearance_order / route_planning）不计入，
    所以这里的 overall 与官方全量 vsibench 分数不可直接比较。
    """
    seen: set[tuple[str, int]] = set()
    per_q: dict[str, list[float]] = collections.defaultdict(list)
    for task, r in rows:
        key = (task, r["doc_id"])
        if key in seen:
            continue
        seen.add(key)
        d = r.get("vsibench_overall")
        if not isinstance(d, dict):
            continue
        qt = d.get("question_type")
        metric = "accuracy" if qt in MCA_Q else NA_METRIC if qt in NA_Q else None
        if metric is None:
            continue
        v = d.get(metric)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            per_q[qt].append(float(v))

    out = {f"{qt}_{'accuracy' if qt in MCA_Q else NA_METRIC}": sum(v) / len(v)
           for qt, v in per_q.items() if v}
    # 逐题口径（能力表用）：rel_direction 三档并成一个池子，答对题数 / 总题数
    pooled: dict[str, tuple[float, int]] = {}
    for qt, v in per_q.items():
        base = "object_rel_direction" if qt.startswith("object_rel_direction_") else qt
        s, n = pooled.get(base, (0.0, 0))
        pooled[base] = (s + sum(v), n + len(v))
    trio = [out.pop(f"object_rel_direction_{s}_accuracy", None) for s in ("easy", "medium", "hard")]
    trio = [x for x in trio if x is not None]
    if trio:
        out["object_rel_direction_accuracy"] = sum(trio) / len(trio)
    return {
        "per_question": {k: (s / n, n) for k, (s, n) in pooled.items() if n},
        "overall": sum(out.values()) / len(out) if out else None,
        "question_types": {qt: len(v) for qt, v in sorted(per_q.items())},
        "subscores": out,
        "missing_types": sorted((MCA_Q | NA_Q) - set(per_q)),
    }


# egotaskqa：数据集 406/159/235 行里只有 225/81/42 道是 verify（是/否）题，其余是开放题
# —— 在我们这套生成式判分下几乎全错。评测本身已用 utils_filter.process_docs_verify_only
# 过滤，但那是 2026-10-02 12:30 才加的：更早的三个 run（qwen3max / qwen3.8-flash /
# qwen3.8-27b）落在磁盘上的是**全量口径**（406 行），之后的（doubao / glm）是过滤后的
# （225 行）。两种口径混在同一张表里不可比 —— 全量那份等于给模型白扣 452 道开放题的分。
# 这里统一按 verify 过滤，判别方式是行数：
#   行数 == 数据总行数 → 旧的全量口径，doc_id 是**原始下标**，需要筛；
#   行数 == verify 题数 → 已经是过滤后的，**全部都是** verify，不能再用下标去筛。
EGOTASKQA_DATA = {
    "egotaskqa_event_order": "egotaskqa_event_order.jsonl",
    "egotaskqa_counterfactual": "egotaskqa_counterfactual.jsonl",
    "egotaskqa_explanatory": "egotaskqa_explanatory.jsonl",
    # 2026-10-20 加：全是 verify 题，上界判据下永远走「已过滤」分支，不会被筛
    "egotaskqa_state_change": "egotaskqa_state_change.jsonl",
}
_EG_CACHE: dict[str, tuple[int, set[int]]] = {}


def egotaskqa_verify(task: str) -> tuple[int, set[int]] | None:
    """(数据总行数, verify 题的原始下标集合)；非 egotaskqa 任务返回 None。"""
    if task not in EGOTASKQA_DATA:
        return None
    if task not in _EG_CACHE:
        path = pathlib.Path(__file__).parent / "lmms_eval/tasks/egotaskqa/data" / EGOTASKQA_DATA[task]
        keep: set[int] = set()
        total = 0
        with open(path, errors="replace") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                total = i + 1
                if str(json.loads(line).get("structural", "")).strip().lower() == "verify":
                    keep.add(i)
        _EG_CACHE[task] = (total, keep)
    return _EG_CACHE[task]


# 成对计分：task -> 指标字段。同一 pair_id 的行都 is_correct 才算对；每对恰好两行
# （MM-UPD 的正常版 + 无解版），逐行平均就等于官方的双重准确率。
DUAL_TASKS: dict[str, str] = {t: "mmupd_dual_accuracy" for t in ("mmupd_aad", "mmupd_iasd", "mmupd_ivqd")}


def _dual_values(rows: list[tuple[str, dict]]) -> dict[tuple[str, int], tuple[float | None, bool]]:
    """{(task, doc_id): (该对的分数, 该对是否有失败行)}；不完整的对（少一行）给 None，不计分。"""
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for task, r in rows:
        if task not in DUAL_TASKS or (task, r["doc_id"]) in seen:
            continue
        seen.add((task, r["doc_id"]))
        e = r.get(DUAL_TASKS[task])
        if isinstance(e, dict) and "pair_id" in e:
            groups[(task, e["pair_id"])].append(r)
    out: dict[tuple[str, int], tuple[float | None, bool]] = {}
    for (task, _pid), rs in groups.items():
        failed = any(FAIL_MARK in str(x.get("filtered_resps", "")) for x in rs)
        v = float(all(x[DUAL_TASKS[task]]["is_correct"] for x in rs)) if len(rs) == 2 else None
        for x in rs:
            out[(task, x["doc_id"])] = (v, failed)
    return out


def summarise(d: pathlib.Path) -> dict | None:
    rows = load(d)
    if not rows:
        return None

    # C 档处理三件事，缺一不可：
    #  1) 重算 —— 落盘字段是修复前判分器写的（phyx/epic 恒 0），见 c_tier_specs 说明；
    #  2) 筛到采样子集 —— 跨模型横比用「类别均衡」口径，不筛就变成自然比例分；
    #  3) 还原 doc_id —— 归档里**并存两套口径**（见 ct_specs.doc_id_convention）：
    #     老的全量 run 里 doc_id 是原始下标，新跑的采样 run 里是紧凑下标 0..n-1。
    #     搞错就会拿紧凑下标去索引数据集，**取到错误的题**且不报错。
    c_task = None
    for task, _ in rows:
        if ct_specs.is_c_task(task):
            c_task = task
            break

    ds = grader = None
    convention = "original"
    idx_map = None
    if c_task is not None:
        try:
            ds, grader = ct_specs.load_task_data(c_task)
            if ds is None:
                pass  # 该任务不重算（robo_spatial 的 mask 依赖临时文件）
            else:
                # 传 rows 走指纹判定（input 与题面对拍），行数启发式在探路 run 上必然
                # 给 unknown，那样会回落读**过期落盘值**，把修前的错数和重算值混进同一张表。
                n_rows = len({r["doc_id"] for _, r in rows})
                convention = ct_specs.doc_id_convention(c_task, ds, n_rows, rows=[r for _, r in rows])
                if convention == "unknown":
                    # 判不出口径就**整个跳过**这个任务，不要回落到落盘值 ——
                    # 宁可在表里显示 "-"，也不能让过期数字冒充结果。
                    print(f"  [warn] {c_task}: doc_id 口径无法判定（{d}），跳过（不回落落盘值）", file=sys.stderr)
                    return None
                idx_map = ct_specs.original_index_map(c_task, ds, n_rows)
        except Exception as e:  # 数据集加载失败同样跳过，不退回过期值
            print(f"  [warn] {c_task}: 重算不可用（{type(e).__name__}: {str(e)[:60]}），跳过", file=sys.stderr)
            return None

    # 全量 run 要筛到采样子集；采样 run 已经是该子集，不能再筛（否则一条不剩）
    want_idx = None
    if ds is not None and convention == "original":
        want_idx = ct_specs.sampled_original_indices(c_task, ds)

    per: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: {"ok": [], "bad": []})
    seen: set[tuple[str, int]] = set()
    pending: set[str] = set()

    # egotaskqa 的口径统一：先按任务数出每个任务的 doc_id 上界，用来判别新旧口径。
    # 判别用 **doc_id 上界** 而不是行数：行数判据会把「部分补跑的旧口径 run」误判成已过滤
    # —— 例如 eval_A_egotaskqa_qwen3max_retry 只有 108 行，但 doc_id 范围是 2..405
    # （原始下标），行数 108 < verify 数 225，用行数比就被当成「已过滤」而放行开放题。
    # 而紧凑下标一定落在 0..verify数-1 内，所以「上界 >= verify 数」必然是全量口径。
    eg_maxid: dict[str, int] = {}
    for t, r in rows:
        eg_maxid[t] = max(eg_maxid.get(t, -1), r["doc_id"])

    # 成对计分的任务（MM-UPD 的双重准确率）：先把每对的结果算出来，再逐行套用
    dual = _dual_values(rows)

    for task, r in rows:
        key = (task, r["doc_id"])
        if key in seen:
            continue
        seen.add(key)

        # egotaskqa 只算 verify（是/否）题。上界 >= verify 数 → 旧的全量 run
        # （doc_id 是原始下标），要按 structural 筛；否则整份都是 verify，不能筛
        # （再筛会一条不剩）。
        eg = egotaskqa_verify(task)
        if eg is not None:
            _, keep = eg
            if eg_maxid.get(task, 0) >= len(keep) and r["doc_id"] not in keep:
                continue

        if ds is not None and grader is not None and ct_specs.is_c_task(task):
            did = r["doc_id"]
            if want_idx is not None and did not in want_idx:
                continue  # 全量 run：筛到采样子集
            # compact 口径（采样 run）要用映射表把紧凑下标还原成原始下标，
            # 否则 ds[did] 取到的是**另一道题**的答案，分数全错且不报错。
            real_idx = idx_map[did] if idx_map is not None else did
            if not (0 <= real_idx < len(ds)):
                continue
            try:
                f, k = ct_specs.VALUE_KEYS[task]
                v = ct_specs.dig_value(grader(dict(ds[real_idx]), [str(r.get("filtered_resps", ""))]), f, k)
            except Exception:
                continue
        elif task in DUAL_TASKS:
            v, pair_failed = dual.get(key, (None, False))
        else:
            v = value(task, r)
        if v is None:
            # judge 写回的 -1 有两种：没判过（无 rb_judge）= 待判；判过但 judge 连续 3 次
            # 输出都解析不了（rb_judge.parse_error）= 官方做法是不计入均值，不能让整个任务停分
            # （2026-10-20 mimo-v2.6-pro 的 plan20_single_arm 就因 1 道这样的题整列空白）。
            if task in AB_PRIMARY and AB_PRIMARY[task][0] == "rb_openended_score" \
                    and not (r.get("rb_judge") or {}).get("parse_error"):
                pending.add(task)
            continue
        bucket = "bad" if FAIL_MARK in str(r.get("filtered_resps", "")) else "ok"
        if task in DUAL_TASKS and pair_failed:
            bucket = "bad"  # 一对里任一行失败，整对按失败处理（剔失败口径下整对不计）
        per[task][bucket].append(v)

    # RoboBench 规划题只要还有待判题，整个任务就不出分：否则 implicit_goal 会只剩
    # 评测时直接判的 Q3（yes/no），冒充整个任务的分数。
    for task in sorted(pending):
        per.pop(task, None)
        print(f"  [warn] {task}: 有题未经 judge 判分（{d}），先跑 judge_robobench_openended.py；该任务不出分",
              file=sys.stderr)

    task_incl, task_excl, n, nfail = {}, {}, 0, 0
    for task, b in per.items():
        allv = b["ok"] + b["bad"]
        if not allv:
            continue
        n += len(allv)
        nfail += len(b["bad"])
        task_incl[task] = sum(allv) / len(allv)
        if b["ok"]:
            task_excl[task] = sum(b["ok"]) / len(b["ok"])

    return {
        "dir": str(d),
        "n": n,
        "fail": nfail,
        "subtasks": {k: len(v["ok"]) + len(v["bad"]) for k, v in sorted(per.items())},
        "incl": sum(task_incl.values()) / len(task_incl) if task_incl else None,
        "excl": sum(task_excl.values()) / len(task_excl) if task_excl else None,
        "per_task_incl": task_incl,
    }


# label -> {benchmark: [按优先级排列的候选目录]}
MODELS = {
    "qwen3.8-max-0902": "qwen3max",
    "qwen3.8-27b": "qwen3_8_27b",
    "qwen3.8-flash": "qwen3_8_flash",
    "qwen3.6-35B-A3B": "qwen3_6_35b",
    "doubao-seed-2-1-pro": "doubao_seed_2_1_pro_260915",
    # Google 的 26B-A4B（激活 4B）。不在内部路由器上，走 OpenRouter 同步端点跑的，
    # 只跑了图片类数据集（视频类的 causalphys/vsibench/egotaskqa 与 physbench 未跑）。
    # 因此它那几行显示 "-" 是**没跑**，不是跑挂了。
    "gemma-4-26b-a4b": "gemma4_26b_a4b",
    # 本地 3B 基线：目录后缀就是 "sampled"，A/B/C 档都按 eval_<档>_<bench>_sampled 命名
    "Qwen2.5-VL-3B 基线": "sampled",
    # 2026-10-09 加入的新模型批次（走 OpenRouter 同步端点）。当时按 A7+B3+C8
    # 全档补跑，所以未跑完的格子是**还在跑**，不是设计上不跑。
    "mimo-v2.6-flash": "mimo_v2_6_flash",
    "mimo-v2.6-pro": "mimo_v2_6_pro",
    "minimax-m3": "minimax_m3",
    # 注意后缀带 _fixed：glm 的首轮跑法见 eval_A_causalphys_glm_5_3_flash.before-403
    # （800/800 全是 403）。_fixed 才是可用的那份，不要退回不带后缀的目录名。
    "glm-5.3-flash": "glm_5_3_flash_fixed",
    # 2026-10-13 加入。走 OpenRouter 同步端点跑的三档全量（A8+B3+C8）。
    # 注意只收 text+image（OpenRouter 页面的 input_modalities 无 video），
    # 但视频任务照常能跑 —— 我们走的是客户端抽帧路径，视频在本地抽成帧图再发。
    "deepseek-v4.1-flash": "deepseek_v4_1_flash",
}
A_BENCH = ["causalphys", "physbench", "vsibench", "momagraph", "negvqa", "refspatial", "egotaskqa", "eriq"]
B_BENCH = ["robobench", "robobench_plan", "mvrobobench", "pio", "asimov", "eai", "mmupd", "aqua", "openeqa",
           "intphys2", "corecognition", "tvbench_cap13", "vlm4d_cap37"]
# C 档的目录词干和任务名不同（连写），与 run_a_tier_api.py / run_model.sh 的键一致
C_BENCH = ["crosspoint", "multihopspatial", "omnispatial", "robospatial",
           "sparbench", "embspatial", "phyx", "epic"]
EXTRA = {  # 手写候选（顺序 = 优先级）
    "qwen3.8-max-0902": {
        "refspatial": ["eval_A_refspatial_qwen3max"],
        "vsibench": ["eval_A_vsibench_qwen3max_merged", "eval_A_vsibench_qwen3max"],
        "negvqa": ["eval_A_negvqa_qwen3max_merged", "eval_A_negvqa_qwen3max"],
        "robobench": ["eval_B_robobench_qwen3max"],
        "mvrobobench": ["eval_B_mvrobobench_qwen3max"],
        "pio": ["eval_B_pio_qwen3max"],
    },
    "qwen3.8-flash": {"causalphys": ["eval_A_causalphys_qwen3_8_flash"], "refspatial": ["eval_A_refspatial_qwen3_8_flash"]},
    "qwen3.8-27b": {"refspatial": ["eval_A_refspatial_qwen3_8_27b"]},
    # 注：曾有一条 "Qwen2.5-VL-3B 基线": {"epic": []}，写于 EPIC 的 3B 基线跑出来之前。
    # 空候选列表会让 candidates() 返回 []，该 benchmark 被**整个跳过**（表格里显示 "-"）
    # 而不报错。归档里 eval_C_epic_sampled 现在有 6661 行实测数据，这条早已过期，2026-10-05 删除。
}


def candidates(label: str, suffix: str, tier: str, bench: str) -> list[str]:
    if label in EXTRA and bench in EXTRA[label]:
        return EXTRA[label][bench]
    return [f"eval_{tier}_{bench}_{suffix}_merged", f"eval_{tier}_{bench}_{suffix}"]


def resolve(path_or_name: str) -> pathlib.Path | None:
    """把候选目录名解析成真实路径。

    C 档的结果目录 2026-10-05 搬进了 ``eval_C_pre_sampling_fullrun/``，仓库根下已无
    ``eval_C_*``，所以 C 档不能再用相对仓库根的 ``pathlib.Path(name)``。
    A/B 档保持在仓库根，行为不变。
    """
    p = pathlib.Path(path_or_name)
    if p.exists():
        return p
    return ct_specs.resolve_dir(path_or_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report: dict[str, dict] = {}
    for label, suffix in MODELS.items():
        entry: dict[str, dict] = {}
        for tier, benches in (("A", A_BENCH), ("B", B_BENCH), ("C", C_BENCH)):
            for bench in benches:
                for c in candidates(label, suffix, tier, bench):
                    rp = resolve(c)
                    if rp is None:
                        continue
                    s = summarise(rp)
                    if s:
                        s["tier"] = tier
                        entry[bench] = s
                        break
        if entry:
            report[label] = entry

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return

    for label, benches in report.items():
        print(f"\n{'='*76}\n{label}\n{'='*76}")
        print(f"  {'benchmark':16s} {'题数':>6s} {'失败':>5s} {'含失败':>8s} {'剔失败':>8s}")
        for tier, tier_benches in (("A", A_BENCH), ("B", B_BENCH), ("C", C_BENCH)):
            for bench in tier_benches:
                s = benches.get(bench)
                if not s or s["tier"] != tier:
                    continue
                inc = f"{s['incl']*100:.2f}" if s["incl"] is not None else "  -  "
                exc = f"{s['excl']*100:.2f}" if s["excl"] is not None else "  -  "
                # C 档题数过少时（多数是只跑过 30 题探路、与采样集交集仅几条），
                # 百分比是巧合而非结论 —— 标出来，别让 "100.00" 被当成满分。
                flag = ""
                if tier == "C" and s["n"] < C_MIN_N:
                    flag = "   ⚠ 题数过少(疑仅探路样本)，不可比"
                print(f"  {tier}|{bench:14s} {s['n']:6d} {s['fail']:5d} {inc:>8s} {exc:>8s}{flag}")


if __name__ == "__main__":
    main()
