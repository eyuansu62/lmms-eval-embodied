#!/usr/bin/env python3
"""benchmark 子任务分数 → 能力分数（逐题加权）。

用法：python map_subtask_to_capability.py --in summary_subtasks.csv --out-dir out/
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

HIERARCHY: dict[int, tuple[str, str, str, str, str, str]] = {
    1: ("基础智能能力", "世界理解", "对象理解", "可供性理解", "本体能力可供性感知", "P0"),
    2: ("基础智能能力", "世界理解", "对象理解", "可供性理解", "客体可供性感知", "P0"),
    3: ("基础智能能力", "世界理解", "对象理解", "本体客体交互关系", "本体客体交互关系", "P0"),
    4: ("基础智能能力", "世界理解", "对象理解", "视觉属性", "客体视觉属性", "P0"),
    5: ("基础智能能力", "世界理解", "对象理解", "物理属性", "客体物理属性", "P1"),
    6: ("基础智能能力", "世界理解", "时空理解", "空间理解", "场景语义", "P1"),
    7: ("基础智能能力", "世界理解", "时空理解", "空间理解", "空间布局", "P1"),
    8: ("基础智能能力", "世界理解", "时空理解", "空间理解", "相对位置", "P0"),
    9: ("基础智能能力", "世界理解", "时空理解", "空间理解", "相对距离", "P0"),
    10: ("基础智能能力", "世界理解", "时空理解", "空间理解", "朝向和姿态", "P1"),
    11: ("基础智能能力", "世界理解", "时空理解", "结构理解", "拓扑关系", "P1"),
    12: ("基础智能能力", "世界理解", "时空理解", "时序理解", "时间指代定位", "P0"),
    13: ("基础智能能力", "世界理解", "时空理解", "时序理解", "事件顺序理解", "P0"),
    14: ("基础智能能力", "世界理解", "指令理解", "意图理解", "指令具化", "P0"),
    15: ("基础智能能力", "世界理解", "指令理解", "意图理解", "指令歧义识别", "P1"),
    16: ("基础智能能力", "世界理解", "指令理解", "内容理解", "目标理解", "P0"),
    17: ("基础智能能力", "世界理解", "指令理解", "内容理解", "否定理解", "P0"),
    18: ("基础智能能力", "世界理解", "指令理解", "内容理解", "条件指令理解", "P0"),
    19: ("基础智能能力", "世界理解", "指令理解", "指代理解", "属性指代", "P0"),
    20: ("基础智能能力", "世界理解", "指令理解", "指代理解", "序数指代", "P0"),
    21: ("基础智能能力", "世界理解", "因果理解", "逻辑因果", "因果链推理", "P0"),
    22: ("基础智能能力", "世界理解", "因果理解", "逻辑因果", "反事实推理", "P1"),
    23: ("基础智能能力", "世界理解", "因果理解", "逻辑因果", "干预推理", "P1"),
    24: ("基础智能能力", "世界理解", "因果理解", "逻辑因果", "归因理解", "P0"),
    25: ("基础智能能力", "世界理解", "因果理解", "逻辑因果", "失败定位与原因分析", "P0"),
    26: ("基础智能能力", "世界理解", "科学理解", "物理规律", "物理知识", "P1"),
    27: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "单体/多体状态理解", "P0"),
    28: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "子任务分解", "P0"),
    29: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "指令顺序理解", "P0"),
    30: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "物理可行性", "P0"),
    31: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "约束满足检查", "P0"),
    32: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "进度估计", "P0"),
    33: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "失败检测", "P0"),
    34: ("基础智能能力", "世界推演", "状态变化推演", "任务与语义状态", "恢复策略", "P0"),
    35: ("基础智能能力", "世界推演", "状态变化推演", "物理状态", "物体恒常性", "P1"),
    36: ("基础智能能力", "世界推演", "状态变化推演", "潜在状态", "安全监控", "P0"),
    37: ("基础智能能力", "世界推演", "运动学推演", "运动轨迹推演", "轨迹理解与预测", "P0"),
    38: ("基础智能能力", "世界推演", "运动学推演", "运动轨迹推演", "运动连续性", "P1"),
    39: ("基础智能能力", "世界推演", "动力学推演", "力学效果预测", "单体力学效果", "P0"),
    40: ("基础智能能力", "世界推演", "动力学推演", "力学效果预测", "多体交互预测", "P1"),
    41: ("基础智能能力", "世界推演", "动力学推演", "材料动力学特性", "刚体、流体、柔体", "P1"),
    42: ("基础智能能力", "世界推演", "空间几何推演", "多视角一致性推演", "多视角匹配", "P1(P0)"),
    43: ("基础智能能力", "世界推演", "空间几何推演", "多视角一致性推演", "第三方视角理解", "P1"),
    44: ("基础智能能力", "世界推演", "空间几何推演", "三维占用与重建推演", "2D/3D理解与转换", "P1"),
    45: ("适应学习能力", "学习进化", "自主探索", "信息收集与知识检索", "场景完整判断与处置", "P1"),
    46: ("适应学习能力", "学习进化", "自主探索", "信息收集与知识检索", "低视觉质量判断与处置", "P1"),
    47: ("适应学习能力", "学习进化", "自主探索", "信息收集与知识检索", "知识检索与求助", "P1"),
}

CAP_TASKS: dict[int, list[tuple[str, list[str]]]] = {
    1:  [("asimov",         ["vla_feasibility"])],
    2:  [("robobench",      ["static_affordance", "dynamic_affordance"]),
         ("pio",            ["affordance"])],
    3:  [("mvrobobench",    ["affordance"]),
         ("asimov",         ["constraints"])],
    4:  [("robobench",      ["visual_attribute"])],
    5:  [("physbench",      ["property_object"])],
    8:  [("robobench",      ["spatial_relation"]),
         ("vsibench",       ["object_rel_direction"])],
    9:  [("mvrobobench",    ["distance"]),
         ("vsibench",       ["object_rel_distance"]),
         ("vsibench",       ["object_abs_distance"])],
    11: [("causalphys",     ["containability"])],
    12: [("robobench",      ["timestamp_analysis"])],
    13: [("tvbench_cap13",  ["action_sequence", "egocentric_sequence"])],
    14: [("robobench_plan", ["robobench_implicit_goal_q1"])],
    15: [("asimov",         ["uncertainty"])],
    16: [("eai",            ["behavior_goal", "virtualhome_goal"])],
    6:  [("openeqa",        ["world_knowledge", "functional_reasoning"])],
    17: [("negvqa",         ["paired"])],
    18: [("momagraph",      ["effect_prediction"])],
    19: [("refspatial",     ["attribute_reference"]),
         ("pio",            ["attribute_reference"])],
    20: [("refspatial",     ["ordinal_reference"]),
         ("pio",            ["ordinal_reference"])],
    21: [("causalphys",     ["multi_hop"])],
    22: [("egotaskqa",      ["counterfactual"])],
    23: [("causalphys",     ["action_substitution", "spatial_manipulation", "temporal_shifting"])],
    24: [("causalphys",     ["mechanics_reasoning"]),
         ("egotaskqa",      ["explanatory"])],
    25: [("robobench",      ["high_level_error", "low_level_error"]),
         ("eriq",           ["mistake_classify_lenient"])],
    27: [("egotaskqa",      ["state_change"]),
         ("eriq",           ["action_understanding_yn"])],
    28: [("robobench_plan", [f"robobench_plan20_{s}" for s in (
             "single_arm", "dual_arm", "human", "mobile_manipulation",
             "articulated", "deformable", "multi_object", "rigid", "special",
             "color", "number", "shape", "size", "world_knowledge",
             "multi_view", "single_view")])],
    29: [("momagraph",      ["action_sequence"])],
    30: [("mvrobobench",    ["trajectory"])],
    31: [("eai",            ["action_seq"])],
    32: [("eriq",           ["task_progress_lenient"])],
    33: [("eriq",           ["mistake_existence_lenient", "success_detection_lenient"])],
    34: [("eriq",           ["mistake_recovery_lenient"])],
    35: [("intphys2",       ["permanence"]),
         ("corecognition",  ["permanence"]),
         ("corecognition",  ["conservation"])],
    36: [("pio",            ["safety"]),
         ("asimov",         ["hsm"])],
    37: [("causalphys",     ["collision_prediction"]),
         ("vlm4d_cap37",    ["ego", "exo"])],
    38: [("intphys2",       ["continuity"])],
    39: [("physbench",      ["single_body_dynamics"])],
    40: [("physbench",      ["multi_body_collision"])],
    41: [("physbench",      ["fluid_prediction"]),
         ("causalphys",     ["deformation", "fluid_flow"])],
    42: [("mvrobobench",    ["cross_view"])],
    44: [("vsibench",       ["object_size_estimation", "room_size_estimation"])],
    46: [("asimov",         ["gauge_judge_readable", "gauge_judge_unreadable"])],
    47: [("mmupd",          ["aad", "iasd", "ivqd"]),
         ("aqua",           ["level0", "level1", "level2", "level3"])],
}

EXCLUDED: dict[int, str] = {}

NO_AGG: dict[int, str] = {
    9: "能力体系 要求相对/绝对距离分开计分，合并会掩盖差异",
}

PARTS: dict[int, dict[str, list[tuple[str, str]]]] = {
    35: {"恒常性": [("intphys2", "permanence"), ("corecognition", "permanence")],
         "守恒": [("corecognition", "conservation")]},
}


EQUAL_WEIGHT_SUBTASKS: dict[int, str] = {
    46: "readable 31 / unreadable 131，全答 unreadable 逐题即 81%",
}

C_TASKS: dict[int, list[tuple[str, list[str]]]] = {
    7:  [("epic", ["GroundDetection", "AffordanceRegion"])],
    26: [("phyx", ["Mechanics_manip"])],
    43: [("omnispatial", ["Perspective_Taking"])],
    10: [("epic", ["Orientation_object"])],
    45: [("crosspoint", ["Visibility Reasoning"])],
    4:  [("epic", ["Color", "Material", "Geometry"])],
}


def _c_n(row: dict, models: list[str]) -> str:
    """C 档一行的题数：取第一个非空的 <模型>__n（各模型题数相同）。"""
    for m in models:
        v = str(row.get(f"{m}__n", "")).strip()
        if v:
            return v
    return ""


def _sum_n(xs) -> str:
    """题数求和；任一来源缺题数就返回空（宁可空白，也不要半截加总）。"""
    tot = 0
    for x in xs:
        try:
            tot += int(str(x).strip())
        except (TypeError, ValueError):
            return ""
    return str(tot) if tot else ""


def _wmean(pairs: list[tuple[str, float | None]], what: str) -> str:
    """[(题数, 分数)] → 逐题平均。分数为 None（该模型没跑）的项跳过；有题数缺失就等权并 warn。"""
    got = [(n, v) for n, v in pairs if v is not None]
    if not got:
        return ""
    try:
        ws = [int(str(n).strip()) for n, _ in got]
    except (TypeError, ValueError):
        print(f"  [warn] {what}：缺题数，退回等权平均", file=sys.stderr)
        return f"{sum(v for _, v in got) / len(got):.2f}"
    return f"{sum(w * v for w, (_, v) in zip(ws, got)) / sum(ws):.2f}"


def _f(x: str) -> float | None:
    """CSV 里的分数转 float；空串 = 该模型没跑，不是 0 分。"""
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_subtasks(path: pathlib.Path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} 是空的")
    models = [k for k in rows[0] if k not in ("benchmark", "subtask", "n")]
    return models, {(r["benchmark"], r["subtask"]): r for r in rows}


DETAIL_FIELDS = ["#", "一级能力", "二级能力", "三级能力", "四级能力", "五级能力",
                 "优先级", "benchmark", "subtask", "题数", "口径"]
AGG_FIELDS = ["#", "一级能力", "二级能力", "三级能力", "四级能力", "五级能力",
              "优先级", "benchmarks", "题数", "口径"]

CALIBER_AB = "A/B档·全量"
CALIBER_C = "C档·均衡"
CALIBER_C_BY_CAP = {10: "C档·题型全量"}
CALIBER_TEXTUALIZED = "B档·文本化"
CALIBER_BY_SUBTASK = {
    ("asimov", "vla_feasibility"): CALIBER_TEXTUALIZED,
    ("asimov", "constraints"): CALIBER_TEXTUALIZED,
    ("asimov", "uncertainty"): CALIBER_TEXTUALIZED,
    ("asimov", "safety"): CALIBER_TEXTUALIZED,
    ("eai", "behavior_goal"): "B档·纯文本",
    ("eai", "virtualhome_goal"): "B档·纯文本",
    ("eai", "action_seq"): "B档·纯文本",
    ("mmupd", "aad"): "B档·非循环",
    ("aqua", "level0"): "B档·judge",
    ("openeqa", "world_knowledge"): "B档·帧序列",
    ("intphys2", "permanence"): "B档·二分类",
    ("intphys2", "continuity"): "B档·二分类",
}

PENDING_BENCH: set[str] = set()


def load_c_subscores(path: pathlib.Path):
    """读 summary_c_subscores.csv → {(bench, group): {model: 分数字符串}}；不存在则 None。"""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    models = [k for k in rows[0] if k not in ("benchmark", "group", "subtask") and not k.endswith("__n")]
    return models, {(r["benchmark"], r["group"]): r for r in rows}


def _write(path: pathlib.Path, fields: list[str], models: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=[*fields, *models], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build(src: pathlib.Path, missing_ok: bool = False):
    """读 summary_subtasks.csv → (models, 明细行, 合并行)。抛 SystemExit 表示映射对不上。"""
    models, table = load_subtasks(src)

    missing = [(b, s) for num in CAP_TASKS for b, subs in CAP_TASKS[num]
               for s in subs if (b, s) not in table]
    pending = [(b, s) for b, s in missing if b in PENDING_BENCH]
    if pending:
        print(f"  [warn] 尚未出分（PENDING_BENCH）：{', '.join(f'{b}/{s}' for b, s in pending)}",
              file=sys.stderr)
        missing = [x for x in missing if x not in pending]
    if missing:
        msg = ", ".join(f"{b}/{s}" for b, s in missing)
        if missing_ok:
            print(f"  [warn] 输入里没有这些子任务，将略去：{msg}", file=sys.stderr)
        else:
            sys.exit(f"这些子任务在 {src} 里找不到：{msg}\n"
                     f"（加 --missing-ok 可跳过，或检查 CAP_TASKS 与 CSV 的命名）")

    rows = []
    for num in sorted(CAP_TASKS):
        if num in EXCLUDED:
            print(f"  [skip] #{num} {HIERARCHY[num][4]}：暂不出分（{EXCLUDED[num]}）", file=sys.stderr)
            continue
        l1, l2, l3, l4, l5, prio = HIERARCHY[num]
        for bench, subs in CAP_TASKS[num]:
            present = [s for s in subs if (bench, s) in table]
            if not present:
                continue
            row: dict = {
                "#": num, "一级能力": l1, "二级能力": l2, "三级能力": l3,
                "四级能力": l4, "五级能力": l5, "优先级": prio,
                "benchmark": bench, "subtask": ", ".join(present),
                "题数": _sum_n(table[(bench, s)].get("n") for s in present),
                "口径": CALIBER_BY_SUBTASK.get((bench, present[0]), CALIBER_AB),
            }
            eq = num in EQUAL_WEIGHT_SUBTASKS
            for m in models:
                row[m] = _wmean([(1 if eq else table[(bench, s)].get("n"), _f(table[(bench, s)].get(m, "")))
                                 for s in present], f"#{num} {bench}")
            rows.append(row)

    c_data = load_c_subscores(src.parent / "summary_c_subscores.csv")
    if c_data is None:
        if C_TASKS:
            print(f"  [warn] 找不到 summary_c_subscores.csv，跳过 "
                  f"#{[n for n in C_TASKS if n not in EXCLUDED]}（先跑 c_tier_subscores.py）",
                  file=sys.stderr)
    else:
        c_models, c_table = c_data
        for num in sorted(C_TASKS):
            if num in EXCLUDED:
                continue
            l1, l2, l3, l4, l5, prio = HIERARCHY[num]
            for bench, groups in C_TASKS[num]:
                present = [g for g in groups if (bench, g) in c_table]
                if not present:
                    continue
                row = {
                    "#": num, "一级能力": l1, "二级能力": l2, "三级能力": l3,
                    "四级能力": l4, "五级能力": l5, "优先级": prio,
                    "benchmark": bench, "subtask": ", ".join(present),
                    "题数": _sum_n(_c_n(c_table[(bench, g)], c_models) for g in present),
                    "口径": CALIBER_C_BY_CAP.get(num, CALIBER_C),
                }
                for m in models:
                    if m not in c_models:
                        row[m] = ""
                        continue
                    row[m] = _wmean([(c_table[(bench, g)].get(f"{m}__n"), _f(c_table[(bench, g)].get(m, "")))
                                     for g in present], f"#{num} {bench}")
                rows.append(row)

    rows.sort(key=lambda r: (r["#"], r["benchmark"]))

    by_cap: dict[int, list[dict]] = {}
    for r in rows:
        by_cap.setdefault(r["#"], []).append(r)
    agg_rows = []
    for num, grows in sorted(by_cap.items()):
        if num in PARTS:
            for part, members in PARTS[num].items():
                prow = [r for r in grows if (r["benchmark"], r["subtask"]) in members]
                if not prow:
                    continue
                agg = {k: prow[0][k] for k in
                       ("#", "一级能力", "二级能力", "三级能力", "四级能力", "五级能力", "优先级")}
                agg["benchmarks"] = f"{part}: " + ", ".join(r["benchmark"] for r in prow)
                agg["题数"] = _sum_n(r.get("题数") for r in prow)
                agg["口径"] = "+".join(dict.fromkeys(r["口径"] for r in prow))
                for m in models:
                    agg[m] = _wmean([(r.get("题数"), _f(r[m])) for r in prow], f"#{num} {part}")
                agg_rows.append(agg)
            continue
        if num in NO_AGG:
            print(f"  [no-agg] #{num} {HIERARCHY[num][4]}：{NO_AGG[num]}，各 benchmark 分列",
                  file=sys.stderr)
            for r in grows:
                agg = {k: r[k] for k in
                       ("#", "一级能力", "二级能力", "三级能力", "四级能力", "五级能力", "优先级")}
                agg["benchmarks"] = f"{r['benchmark']}/{r['subtask']}"
                agg["题数"] = r.get("题数", "")
                agg["口径"] = r["口径"]
                for m in models:
                    agg[m] = r[m]
                agg_rows.append(agg)
            continue

        by_caliber: dict[str, list[dict]] = {}
        for r in grows:
            by_caliber.setdefault(r["口径"], []).append(r)
        for caliber, crows in sorted(by_caliber.items()):
            full = [r for r in crows if all(_f(r[m]) is not None for m in models)]
            if full and len(full) < len(crows):
                dropped = [r["benchmark"] for r in crows if r not in full]
                print(f"  [agg] #{num} {HIERARCHY[num][4]}：{', '.join(dropped)} 未覆盖全部模型，"
                      f"不参与合并（明细表保留）", file=sys.stderr)
                crows = full
            agg = {k: crows[0][k] for k in
                   ("#", "一级能力", "二级能力", "三级能力", "四级能力", "五级能力", "优先级")}
            agg["benchmarks"] = ", ".join(r["benchmark"] for r in crows)
            agg["题数"] = _sum_n(r.get("题数") for r in crows)
            agg["口径"] = caliber
            for m in models:
                agg[m] = _wmean([(r.get("题数"), _f(r[m])) for r in crows], f"#{num} 合并")
            agg_rows.append(agg)

    return models, rows, agg_rows


def write_outputs(src: pathlib.Path, out_dir: pathlib.Path,
                  missing_ok: bool = False) -> tuple[int, int]:
    models, rows, agg_rows = build(src, missing_ok)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "summary_capability.csv", DETAIL_FIELDS, models, rows)
    _write(out_dir / "summary_capability_agg.csv", AGG_FIELDS, models, agg_rows)
    return len(rows), len(agg_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="summary_subtasks.csv")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--missing-ok", action="store_true",
                    help="映射里的子任务在输入中找不到时只警告，不报错")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    if not src.exists():
        sys.exit(f"找不到输入文件 {src}（先跑 dump_reports.py 生成 summary_subtasks.csv）")
    models, table_rows = load_subtasks(src)
    print(f"读入 {src}：{len(table_rows)} 个子任务 × {len(models)} 个模型")

    n_detail, n_agg = write_outputs(src, pathlib.Path(args.out_dir), args.missing_ok)
    print(f"写出 {pathlib.Path(args.out_dir) / 'summary_capability.csv'}  "
          f"（{n_detail} 行：能力 × benchmark）")
    print(f"写出 {pathlib.Path(args.out_dir) / 'summary_capability_agg.csv'}  "
          f"（{n_agg} 行：能力合并）")


if __name__ == "__main__":
    main()
