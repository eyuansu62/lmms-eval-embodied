#!/usr/bin/env python3
"""把 build_report.py 的数据落成 reports/*.csv（可复现）。

背景
----
`reports/` 下那三份 CSV 是 2026-10-03 一次性生成的，**仓库里没有任何脚本会
重写它们**（grep 全仓无命中），`build_report.py` 也只打印到 stdout。于是
2026-10-08 那批重跑（截断修复、帧数修复、pio 口径）落盘后，报表数字全部
过期，引用前只能手工核对。

用法
----
    python dump_reports.py                # 写 reports/
    python dump_reports.py --out DIR      # 写指定目录（先干跑对比用）
    python dump_reports.py --json         # 只打印，不落盘

产出与旧文件逐列对齐：
    summary_A_B.csv       模型 × benchmark 总分（含 A/B 档均值、两档合计）
    summary_failed.csv    各 benchmark 的失败题数
    summary_subtasks.csv  模型 × 子任务

vsibench 的子任务口径要注意
---------------------------
它的样本文件只有 3 个切片（rel_direction / rel_distance / size_estimation），
但报表里的子任务是 5 个（object_abs_distance、room_size_estimation …）——
后者来自每条样本 `vsibench_overall` 里的 `question_type` 字段，必须用
`build_report.vsibench_official()` 重算，直接取 `per_task_incl` 会得到 3 行而不是 5 行。

口径
----
用**剔失败**（``excl``：只在成功题上平均），不是含失败。这是从旧 CSV 反推的
——momagraph 三个模型的旧值 75.68 / 78.88 / 75.15 与 ``excl`` 逐位相同，
而 ``incl`` 是 75.23 / 78.63 / 74.91。旧表头没写口径，改回含失败会让这几格
看起来「变差了」，其实是换了个算法。改口径要显式改 ``CALIBER``。

C 档
----
C 档走 build_report 的重算路径（落盘值是修复前判分器写的），很慢（epic 一项
约 16 秒，因为它要逐题重跑判分器），且在数据集加载失败时会**整格跳过**而不是
回落到过期值。所以要加 `--with-c` 才纳入。

**C 档不并入「两档合计」**：口径不同 —— C 档是类别均衡采样、A/B 档是全量，
混算的均值没有意义。C 档有自己的「C档均值」行，且该行 note 会标出各模型实际
可用项数（本轮作业未跑完时，不同模型的项数会不一样）。

C 档的子任务键与 A/B 不同名（如 `crosspoint_bench` / `omni_spatial`），
`summary_subtasks.csv` 里按原键去前缀后写出行；`crosspoint` 一项会展开成
多个子任务行（fine_grained_grounding / visibility_reasoning / …）。
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import build_report as br  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "reports"

# 列顺序 = 表头顺序。基线的 4 个在前（保持与历史 CSV 逐列可比），
# 2026-10-09 新加的 4 个排在后面，gemma4 是 2026-10-13 加的第九列。
# label 必须都在 build_report.MODELS 里。
#
# ⚠️ gemma4 的 causalphys / physbench / vsibench / egotaskqa 是**没跑**（OpenRouter
#    那 4 个视频类任务当时未跑），不是跑挂了 —— 那几格显示空是预期的。
MODEL_COLS = {
    "doubao-seed-2-1-pro": "doubao-seed-2-1-pro",
    "qwen3.8-max-0902": "qwen3.8-max-0902",
    "qwen3.8-27b": "qwen3.8-27b",
    "qwen3.8-flash": "qwen3.8-flash",
    "mimo-v2.6-flash": "mimo-v2.6-flash",
    "mimo-v2.6-pro": "mimo-v2.6-pro",
    "minimax-m3": "minimax-m3",
    "glm-5.3-flash": "glm-5.3-flash",
    "gemma-4-26b-a4b": "gemma-4-26b-a4b",
    "deepseek-v4.1-flash": "deepseek-v4.1-flash",
}

# 报表口径：excl = 剔失败（与旧 CSV 一致，见模块 docstring），incl = 含失败
CALIBER = "excl"

# 旧表里的人工备注，按 benchmark 保留（数值列由本脚本重算，备注是人写的判断）
NOTES = {
    "refspatial": "仅7题，不参与解读",
    "egotaskqa": "仅是/否题(348/800)",
    "eriq": "800题按规模等比分配；单任务n=50~197，半宽±7~14pt；口径取官方严格",
    "openeqa": "#6 场景语义；16 帧/题(本地抽帧)，judge=qwen3.8-max；数据 2023 年，可能有污染",
    "corecognition": "#35 守恒 71 / 恒常性 98；frame-combined 单图，官方 prompt；是非题 YES/NO 等量采样",
    "tvbench_cap13": "#13 事件顺序；action_sequence 按 start/end 裁剪、A/B 各 100，egocentric 全量 200；16 帧；二选一盲猜 50 / 四选一 25",
    "vlm4d_cap37": "#37 当前运动方向；真实视频 300 题，16 帧；exo 盲测 30.7(随机25)，ego 多为左/右二选一、盲测 46.7(按50读)",
    "intphys2": "#35 恒常性 / #38 运动连续性；1 帧/秒(11–15 帧)，官方 prompt1；全答 Yes=50%，CC BY-NC 仅限评测",
    "aqua": "#47 询问必要信息；judge=qwen3.8-max 判策略等级(论文用GPT-5-mini)；策略性提示词，非零样本",
    "mmupd": "#47 自我检视；双重准确率(正常+无解两版都对)；只问主版本，非官方循环评测；n 为请求数=2×题数",
    "eai": "纯文本(官方原协议)；本表为逐题F1均值，官方榜单是micro F1(见results json的eai_f1)",
    "asimov": "决策题文本化(官方为tool-call)，不可与论文比；盲猜 constraints 53.2/其余50；gauge 两切片须等权读；safety 四类等权",
}

# vsibench 官方子任务的显示名：subscores 的键带指标后缀，去掉
VSIBENCH_DISPLAY = {
    "object_abs_distance": "object_abs_distance",
    "object_rel_direction": "object_rel_direction",
    "object_rel_distance": "object_rel_distance",
    "object_size_estimation": "object_size_estimation",
    "room_size_estimation": "room_size_estimation",
}


def collect(c_tier: bool) -> dict[str, dict]:
    """跑一遍 build_report 的取数逻辑，返回 {label: {bench: summary}}。"""
    tiers = [("A", br.A_BENCH), ("B", br.B_BENCH)]
    if c_tier:
        tiers.append(("C", br.C_BENCH))
    report: dict[str, dict] = {}
    for label, suffix in br.MODELS.items():
        if label not in MODEL_COLS:
            continue
        entry: dict[str, dict] = {}
        for tier, benches in tiers:
            for bench in benches:
                for c in br.candidates(label, suffix, tier, bench):
                    rp = br.resolve(c)
                    if rp is None:
                        continue
                    s = br.summarise(rp)
                    if s:
                        s["tier"] = tier
                        s["dir_used"] = c
                        entry[bench] = s
                        break
        if entry:
            report[label] = entry
    return report


def _fmt(v) -> str:
    return f"{v*100:.2f}" if isinstance(v, (int, float)) else ""


def has_c(report: dict) -> bool:
    """本次取数里是否有任何 C 档结果（没有就不该往 A/B 表里插空行）。"""
    return any(s.get("tier") == "C"
               for bs in report.values() for s in bs.values())


def c_rows(report: dict, labels: list[str]) -> list[dict]:
    """C 档的 benchmark 行。

    C 档**独立成行**，不与 A/B 混算：口径不同 —— C 档是类别均衡采样（且判分器
    要现重算），A/B 档是全量。混在一起算的均值没有意义。

    照用户 2026-10-09 的要求，**不再输出任何均值行**（含 C 档自己的均值）：
    表里只保留各 benchmark 的原始数值。作业未跑完时各模型项数不齐，均值那一行
    反而会把「前提不全」固化成看着像结论的数字。
    """
    rows = []
    for bench in br.C_BENCH:
        present = [l for l in labels if bench in report.get(l, {})
                   and report[l][bench].get("tier") == "C"]
        if not present:
            continue
        rows.append({
            "tier": "C",
            "benchmark": bench,
            "n_questions": max(report[l][bench]["n"] for l in present),
            **{MODEL_COLS[l]: _fmt(report[l][bench][CALIBER]) if l in present else ""
               for l in labels},
            "note": NOTES.get(bench, ""),
        })
    return rows


def write_summary_ab(report: dict, out: pathlib.Path) -> None:
    rows = []
    labels = list(MODEL_COLS)
    for tier, benches in (("A", br.A_BENCH), ("B", br.B_BENCH)):
        for bench in benches:
            present = [l for l in labels if bench in report.get(l, {})
                       and report[l][bench].get("tier") == tier]
            if not present:
                continue
            n = max(report[l][bench]["n"] for l in present)
            row = {
                "tier": tier,
                "benchmark": bench,
                "n_questions": n,
                **{MODEL_COLS[l]: _fmt(report[l][bench][CALIBER]) if l in present else ""
                   for l in labels},
                "note": NOTES.get(bench, ""),
            }
            rows.append(row)

    if has_c(report):
        rows.extend(c_rows(report, labels))
    path = out / "summary_A_B.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["tier", "benchmark", "n_questions",
                                           *MODEL_COLS.values(), "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"  写入 {path}  ({len(rows)} 行)")


def write_failed(report: dict, out: pathlib.Path) -> None:
    rows = []
    labels = list(MODEL_COLS)
    for bench in br.A_BENCH + br.B_BENCH + br.C_BENCH:
        present = [l for l in labels if bench in report.get(l, {})]
        if not present:
            continue
        rows.append({
            "benchmark": bench,
            "n_questions": max(report[l][bench]["n"] for l in present),
            **{MODEL_COLS[l]: report[l][bench]["fail"] if l in present else ""
               for l in labels},
        })
    path = out / "summary_failed.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["benchmark", "n_questions",
                                           *MODEL_COLS.values()])
        w.writeheader()
        w.writerows(rows)
    print(f"  写入 {path}  ({len(rows)} 行)")


def _vsibench_subscores(label: str, suffix: str) -> dict[str, tuple[float, int]]:
    """vsibench 的 5 个 question_type：{名字: (逐题平均分, 题数)}。

    2026-10-20 起能力表一律逐题口径（答对题数 / 总题数），rel_direction 的 easy/medium/hard
    三档并成一个池子算，不再是官方的三档等权；官方口径仍在 summary_A_B.csv 的 vsibench 总分里。
    """
    for c in br.candidates(label, suffix, "A", "vsibench"):
        rp = br.resolve(c)
        if rp is None:
            continue
        rows = br.load(rp)
        if not rows:
            continue
        off = br.vsibench_official(rows)
        out = {VSIBENCH_DISPLAY[k]: vn for k, vn in off["per_question"].items() if k in VSIBENCH_DISPLAY}
        if out:
            return out
    return {}


# PIO 序数指代（#20）的 18 个 doc_id：从 pio_attribute_reference 的 95 道 s1 题里
# 人工审出的序数子集，见 lmms_eval/tasks/pio/build_pio_ordinal.py。
_PIO_ORDINAL_PATH = HERE / "lmms_eval/tasks/pio/data/pio_ordinal_reference.jsonl"


def _pio_ordinal_scores(label: str, suffix: str) -> tuple[dict[str, float], int]:
    """#20 的 PIO 序数子集：不发新请求，从已有 pio_attribute_reference 样本按 doc_id 取分。

    题干里的 polygon / 图片和序数切片完全一致（切片就是从同一份数据里挑的），
    所以模型对这几道题的作答可以直接复用，只是**只在 18 道上平均**。
    """
    subs = []
    for c in br.candidates(label, suffix, "B", "pio"):
        rp = br.resolve(c)
        if rp is None:
            continue
        rows = [r for t, r in br.load(rp) if t == "pio_attribute_reference"]
        if not rows:
            continue
        by_id = {int(r["doc_id"]): r for r in rows}
        want = [json.loads(l)["src_doc_id"] for l in open(_PIO_ORDINAL_PATH)]
        vals = []
        for i in want:
            if i not in by_id:
                continue
            v = br.value("pio_ordinal_reference", by_id[i])
            if v is not None:
                vals.append(v)
        if vals:
            subs.append((sum(vals) / len(vals), len(vals)))
            break
    if not subs:
        return {}, 0
    val, n = subs[0]
    return {"ordinal_reference": val}, n


# ERIQ 进能力表的子任务一律用「宽松 + Yes/No 归一」口径（2026-10-20 用户定）：每题宽松抽取
# （eriq_accuracy.lenient_correct）与 Yes/No 归一（build_report._yn_value）任一判对即算对。
# 官方严格口径只认「回复开头就是字母」，先分析再给字母、或照提示词回 Yes/No 的模型会被整片判错：
#   qwen3.8-flash task_progress 严格 7.25 / 宽松 68.12；deepseek success_detection 严格 17.26 / 归一 60.41。
# 能力表测的是会不会做，不是守不守格式。ERIQ 主表（summary_subtasks 的原子任务名、summary_A_B）仍是严格口径，
# 与 ERIQ README 的 baseline 可比；三个口径并排见 summary_eriq_calibers.csv。
# {派生子任务名: (原任务, 只取判断题)}
ERIQ_CAP_SUBTASKS: dict[str, tuple[str, bool]] = {
    # #27：四选一那半不看视频就答对 68.8%，只取判断题（A.Yes / B.No），盲测 47.1%
    "action_understanding_yn": ("eriq_action_understanding", True),
    "mistake_classify_lenient": ("eriq_mistake_classify", False),     # #25
    "task_progress_lenient": ("eriq_task_progress", False),           # #32
    "mistake_existence_lenient": ("eriq_mistake_existence", False),   # #33
    "success_detection_lenient": ("eriq_success_detection", False),   # #33
    "mistake_recovery_lenient": ("eriq_mistake_recovery", False),     # #34
}


def _eriq_cap_value(task: str, r: dict) -> float | None:
    e = r.get("eriq_accuracy")
    if not isinstance(e, dict):
        return None
    if br.FAIL_MARK in str(r.get("filtered_resps", "")):
        return 0.0
    yn = br._yn_value(task, r)
    return float(bool(e.get("lenient_correct")) or bool(yn))


def _eriq_cap_scores(label: str, suffix: str) -> dict[str, tuple[float, int]]:
    """能力表用的 ERIQ 派生子任务：{名字: (逐题平均, 题数)}，不发新请求，从已有样本取分。"""
    for c in br.candidates(label, suffix, "A", "eriq"):
        rp = br.resolve(c)
        if rp is None:
            continue
        loaded = br.load(rp)
        if not loaded:
            continue
        out = {}
        for name, (task, yn_only) in ERIQ_CAP_SUBTASKS.items():
            vals = []
            for t, r in loaded:
                if t != task or (yn_only and "A.Yes" not in str(r.get("input", ""))):
                    continue
                v = _eriq_cap_value(task, r)
                if v is not None:
                    vals.append(v)
            if vals:
                out[name] = (sum(vals) / len(vals), len(vals))
        if out:
            return out
    return {}


def _loaded(label: str, suffix: str, tier: str, bench: str) -> list[tuple[str, dict]]:
    for c in br.candidates(label, suffix, tier, bench):
        rp = br.resolve(c)
        if rp is not None:
            loaded = br.load(rp)
            if loaded:
                return loaded
    return []


def _negvqa_paired_scores(label: str, suffix: str) -> dict[str, tuple[float, int]]:
    """#17 用：同一题原版与否定版**都答对**才算对（2026-10-21）。

    原版不含否定词，单看它测的是普通 VQA；否定版单看又混着「看没看懂图」。
    两版选项相同、答案互换，都对 = 看懂了图且理解了 not。按 doc_id 配对（两切片同序，已核对）。
    """
    loaded = _loaded(label, suffix, "A", "negvqa")
    acc = {"original": {}, "negated": {}}
    for t, r in loaded:
        for kind in acc:
            if t == f"negvqa_natural_scene_{kind}":
                v = r.get(f"negvqa_{kind}_accuracy")
                if isinstance(v, (int, float)):
                    acc[kind][r["doc_id"]] = float(v)
    ids = set(acc["original"]) & set(acc["negated"])
    if not ids:
        return {}
    vals = [float(acc["original"][i] == 1.0 and acc["negated"][i] == 1.0) for i in ids]
    return {"paired": (sum(vals) / len(vals), len(vals))}


def _asimov_gauge_judge_scores(label: str, suffix: str) -> dict[str, tuple[float, int]]:
    """#46 用：只看「能不能读」判断对不对，不看读数准不准（2026-10-21）。

    可读题答了任何数值即算判断正确（读数错是读表精度问题，不是视觉质量判断）；
    不可读题答 unreadable 算对。抽不出 / 请求失败记错。
    """
    out = {}
    loaded = _loaded(label, suffix, "B", "asimov")
    def _is_number(p) -> bool:  # 样本里 pred 存成字符串（'1020.0'）
        try:
            float(p)
            return p is not None and not isinstance(p, bool)
        except (TypeError, ValueError):
            return False

    for kind, ok in (("readable", _is_number),
                     ("unreadable", lambda p: p == "unreadable")):
        vals = []
        for t, r in loaded:
            if t != f"asimov_gauge_{kind}":
                continue
            e = r.get("asimov_accuracy")
            if not isinstance(e, dict) or br.FAIL_MARK in str(r.get("filtered_resps", "")):
                vals.append(0.0)
                continue
            vals.append(float(ok(e.get("pred"))))
        if vals:
            out[f"gauge_judge_{kind}"] = (sum(vals) / len(vals), len(vals))
    return out


DERIVED_SCORES = {"negvqa": _negvqa_paired_scores, "asimov": _asimov_gauge_judge_scores}


def write_subtasks(report: dict, out: pathlib.Path) -> None:
    labels = list(MODEL_COLS)
    # (benchmark, subtask) 顺序：沿用旧表的 A 档在前、B 档在后的排布
    seen: dict[tuple[str, str], dict] = collections.OrderedDict()

    for tier, benches in (("A", br.A_BENCH), ("B", br.B_BENCH), ("C", br.C_BENCH)):
        for bench in benches:
            # vsibench 特殊：子任务来自 question_type，不是样本文件名
            if bench == "vsibench":
                for l in labels:
                    if "vsibench" not in report.get(l, {}):
                        continue
                    for sub, (val, n_sub) in _vsibench_subscores(l, br.MODELS[l]).items():
                        k = seen.setdefault((bench, sub), {})
                        k[MODEL_COLS[l]] = _fmt(val)
                        k["n"] = max(k.get("n", 0), n_sub)
                continue
            # pio 额外派生一个序数指代子任务（#20）：数据不重跑，见 _pio_ordinal_scores
            if bench == "pio":
                for l in labels:
                    if "pio" not in report.get(l, {}):
                        continue
                    subs, n_sub = _pio_ordinal_scores(l, br.MODELS[l])
                    for sub, val in subs.items():
                        k = seen.setdefault((bench, sub), {})
                        k[MODEL_COLS[l]] = _fmt(val)
                        if n_sub:
                            k["n"] = max(k.get("n", 0), n_sub)
            # negvqa / asimov 额外派生能力表用的子任务（#17 配对口径、#46 判断口径）：见 DERIVED_SCORES
            if bench in DERIVED_SCORES:
                for l in labels:
                    if bench not in report.get(l, {}):
                        continue
                    for sub, (val, n_sub) in DERIVED_SCORES[bench](l, br.MODELS[l]).items():
                        k = seen.setdefault((bench, sub), {})
                        k[MODEL_COLS[l]] = _fmt(val)
                        k["n"] = max(k.get("n", 0), n_sub)
            # eriq 额外派生能力表用的宽松 + Yes/No 归一子任务：见 ERIQ_CAP_SUBTASKS
            if bench == "eriq":
                for l in labels:
                    if "eriq" not in report.get(l, {}):
                        continue
                    for sub, (val, n_sub) in _eriq_cap_scores(l, br.MODELS[l]).items():
                        k = seen.setdefault((bench, sub), {})
                        k[MODEL_COLS[l]] = _fmt(val)
                        k["n"] = max(k.get("n", 0), n_sub)
            for l in labels:
                s = report.get(l, {}).get(bench)
                if not s or s.get("tier") != tier:
                    continue
                for key, val in s["per_task_incl"].items():
                    n_sub = s.get("subtasks", {}).get(key)
                    if tier == "C":
                        # C 档的任务键本身就是最有意义的标识（crosspoint_bench /
                        # omni_spatial / robo_spatial_all / phyx_mc …），与
                        # run_a_tier_api.py 的 --tasks 取值一致，便于对照。
                        # **不能**走下面的去前缀逻辑：bench="crosspoint" 会把
                        # "crosspoint_bench" 削成 "bench"、phyx 削成 "mc"。
                        sub = key
                    else:
                        # 去掉 benchmark 前缀。mvrobobench 的样本文件名用的是
                        # mvrb_ 缩写（旧 CSV 里也写作 affordance/cross_view/…）。
                        sub = key
                        for pre in (bench + "_", bench + "s_", "mvrb_"):
                            if sub.startswith(pre):
                                sub = sub[len(pre):]
                                break
                    k = seen.setdefault((bench, sub), {})
                    k[MODEL_COLS[l]] = _fmt(val)
                    if n_sub is not None:
                        # 题数取各模型最大值：同一子集各模型题数相同，缺题/补跑的模型偏小
                        k["n"] = max(k.get("n", 0), n_sub)

    # 旧表在每个 benchmark 内按子任务名排序，保持同样的排布便于逐行对比
    order = {b: i for i, b in enumerate(br.A_BENCH + br.B_BENCH + br.C_BENCH)}
    items = sorted(seen.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1]))
    rows = [{"benchmark": b, "subtask": s, "n": cols.get("n", ""),
             **{c: cols.get(c, "") for c in MODEL_COLS.values()}}
            for (b, s), cols in items]
    path = out / "summary_subtasks.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["benchmark", "subtask", "n",
                                           *MODEL_COLS.values()])
        w.writeheader()
        w.writerows(rows)
    print(f"  写入 {path}  ({len(rows)} 行)")


def write_eriq_calib(report: dict, out: pathlib.Path) -> None:
    """ERIQ 的三个判分口径对照表（answer 中一列一个模型，三个口径分块）。

    为什么单独出一张表、而不是给 summary_subtasks.csv 再加一组列：
    那张表已经是「列 = 模型」，再加两组口径就是 30 列，宽度翻倍、没法看。
    这里改成「行 = 模型 × 口径」，ERIQ 那 8 个子任务横向铺开。

    三个口径：
      official  官方严格（`^([A-F])`），与 README 的 6 个 baseline 可比 —— **主口径**
      lenient   共享 mcq_extract 的优先级抽取，捞「答对但格式不合规」
      yn        Yes/No → 字母归一，只对选项字面是 A.Yes/B.No 的三个子任务生效

    最后一列 `换算题数` 是 yn 相对 official 多算对的题数。它 > 3 就说明该模型
    偏好回 Yes/No，读它的 official 分数时要知道这一点。
    """
    labels = [l for l in MODEL_COLS if any(
        t in br.AB_PRIMARY for t in ("eriq_human_intention",))]
    subs = [t[len("eriq_"):] for t in br.ERIQ_YN_TASKS]

    def score(label: str, suffix: str, task: str, mode: str):
        for c in br.candidates(label, suffix, "A", "eriq"):
            rp = br.resolve(c)
            if rp is None:
                continue
            # 每个子任务单独取文件：一个目录里 8 个子任务的 samples 是分开的
            rows = [r for t, r in br.load(rp) if t == task]
            if not rows:
                continue
            ok = n = 0
            for r in rows:
                if mode == "yn":
                    v = br._yn_value(task, r)
                    if v is None:
                        continue
                    ok += v
                else:
                    e = r.get("eriq_accuracy")
                    if not isinstance(e, dict):
                        continue
                    key = "is_correct" if mode == "official" else "lenient_correct"
                    if key not in e:
                        continue
                    ok += 1 if e[key] else 0
                n += 1
            return ok / n if n else None, n
        return None, 0

    rows = []
    for label, suffix in br.MODELS.items():
        if label not in MODEL_COLS:
            continue
        for mode in ("official", "lenient", "yn"):
            row = {"model": label, "caliber": mode}
            for task, sub in zip(br.ERIQ_YN_TASKS, subs):
                v, _ = score(label, suffix, task, mode)
                row[sub] = _fmt(v)
            if mode == "yn":
                # 换算实际多算对的题数（全 8 子任务合计）
                diff = 0
                for task in br.ERIQ_YN_TASKS:
                    vo, no = score(label, suffix, task, "official")
                    vy, ny = score(label, suffix, task, "yn")
                    if vo is None or vy is None or no != ny:
                        continue
                    diff += round((vy - vo) * no)
                row["yn_gain"] = diff
            else:
                row["yn_gain"] = ""
            rows.append(row)

    path = out / "summary_eriq_calibers.csv"
    cols = ["model", "caliber", *subs, "yn_gain"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  写入 {path}  ({len(rows)} 行)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--with-c", action="store_true", help="也重算 C 档（慢）")
    ap.add_argument("--json", action="store_true", help="只打印不落盘")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if not args.json:
        out.mkdir(parents=True, exist_ok=True)

    print("扫描结果目录（C 档重算较慢）…" if args.with_c else "扫描结果目录…")
    report = collect(args.with_c)
    print(f"  覆盖模型 {len(report)} 个")

    if args.json:
        slim = {l: {b: {"n": s["n"], "fail": s["fail"], "incl": s["incl"],
                        "excl": s["excl"], "dir": s["dir_used"], "tier": s["tier"]}
                    for b, s in bs.items()} for l, bs in report.items()}
        print(json.dumps(slim, ensure_ascii=False, indent=1))
        return

    write_summary_ab(report, out)
    write_failed(report, out)
    write_subtasks(report, out)
    write_eriq_calib(report, out)

    # 能力体系分数表：只走 map_subtask_to_capability 一份映射，避免两处实现漂移
    # （2026-10-16 之前 dump_capability_report.py 自己读 per_task_incl，vsibench
    # 那几行因此与 summary_subtasks.csv 的官方 question_type 口径对不上）。
    #
    # C 档子分数（#7 / #26 / #43 的来源）要先算，能力表会去读它。
    # 较慢且有网依赖（要拉 EPIC/PhyX/OmniSpatial 数据集），失败就沿用已有的旧文件。
    try:
        import c_tier_subscores as cts
        cts.write_outputs(out)
    except Exception as exc:
        print(f"  [warn] C 档子分数未重算（{type(exc).__name__}: {str(exc)[:60]}），"
              f"沿用已有的 summary_c_subscores.csv")

    try:
        import map_subtask_to_capability as msc
        n_detail, n_agg = msc.write_outputs(out / "summary_subtasks.csv", out)
        print(f"  写入 {out / 'summary_capability.csv'}  ({n_detail} 行)")
        print(f"  写入 {out / 'summary_capability_agg.csv'}  ({n_agg} 能力)")
    except SystemExit as exc:
        print(f"  [warn] 能力分数表生成失败：{exc}")
    except Exception as exc:
        print(f"  [warn] 能力分数表生成失败：{exc}")

    print("\n完成。这些 CSV 现在由本脚本重算，引用前先跑一次。")


if __name__ == "__main__":
    main()
