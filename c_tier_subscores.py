#!/usr/bin/env python3
"""按类别抽取 C 档子分数，供能力表使用。

背景
----
能力表（summary_capability*.csv）原来只吃 A/B 档的 summary_subtasks.csv，但
能力体系（CAPABILITIES.md） 的 G 列有三项能力引用的正是 **C 档** 的数据源：

    #7  空间布局        EPIC-Bench：GroundDetection / AffordanceRegion / FeasiblePath
    #26 物理知识        PhyX：Mechanics
    #43 第三方视角理解   OmniSpatial：Perspective Taking

这三项的分数在 C 档结果里本来就是分开计的（epic 有 epic_accuracy_by_task_type、
omni_spatial 有 perspective_taking、phyx 的 eval_results.category），所以不必重跑，
按类别取出来即可。

口径
----
与 build_report 一致，分三步：
  1. 用当前判分器**重算**（落盘分数含 4 个静默 bug，见 c_tier_specs 的 docstring）；
  2. 判定 doc_id 口径（归档里并存 original / compact 两套）；
  3. 全量 run 要筛到采样子集（C 档分数是类别均衡口径）。

**C 档分数与 A/B 档不可直接合并**：它是类别均衡采样下的准确率，不是自然分布水平。
能力表里用「口径」列标出（C档·均衡），跨档位不做平均。

用法：
    python c_tier_subscores.py                 # 生成 reports/summary_c_subscores.csv
    python c_tier_subscores.py --json          # 只打印
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import pathlib
import re
import sys
from typing import Callable

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_report as br                      # noqa: E402
from lmms_eval.tasks import c_tier_specs as ct  # noqa: E402

FAIL_MARK = "[LMMS_EVAL_FAILED_PLACEHOLDER"  # 真值见 build_report.FAIL_MARK
FAIL_MARK = br.FAIL_MARK

OUT = HERE / "reports"

# (C 档 bench, 分组字段, 要取的分组值)
# 分组值来自 能力体系（CAPABILITIES.md）对各能力的子集定义：
#   #7  → epic 的 GroundDetection / AffordanceRegion / FeasiblePath_Ego / FeasiblePath_Exo
#         （AffordanceRegion 属 Manipulation 包，其余属 Navigation 包）
#   #26 → phyx 的 Mechanics
#   #43 → omni_spatial 的 Perspective_Taking
#   #10 → epic 的 Orientation / OrientationHuman（2026-10-17 加）。采样集只有 20 题，另由
#         eval_C_epicorient_<后缀>/ 补跑其余 356 题（关采样、doc_id 为原始下标），合计全部 376 题，
#         见 SUPPLEMENT。这两组是「该题型全量」，不是均衡采样
#   #45 → crosspoint 的 Visibility Reasoning（2026-10-17 加；跨视角是否被遮挡，200 题）
#   #42 → epic 的 VisualMatching（2026-10-19 加；能力体系 指定 MV-RoboBench + EPIC 两源，
#         本地只挂了 MV-RoboBench。采样集里 78 题，与 #7/#10 的 task_type 不重叠）
#   #4  → epic 的 Color / Material / Geometry（2026-10-19 加；能力体系 指定 RoboBench + EPIC，
#         本地只有 RoboBench 31 题。这三个 task_type 就是 EPIC 的 BasicAttributes 颜色/几何题，
#         采样集里 18/19/12 题。与 VisualMatching、#7 的 task_type 都不重叠）
GROUPS: list[tuple[str, str, list[str]]] = [
    ("epic_bench",  "task_type", ["GroundDetection", "AffordanceRegion",
                                  "FeasiblePath_Ego", "FeasiblePath_Exo",
                                  "Orientation", "OrientationHuman",
                                  "VisualMatching",
                                  "Color", "Material", "Geometry"]),
    ("crosspoint_bench", "type", ["Visibility Reasoning"]),
    ("phyx_mc",     "category",  ["Mechanics"]),
    ("omni_spatial", "task_type", ["Perspective_Taking"]),
]


# 派生分组：task -> {原分组值: (派生分组名, 判定函数 doc -> bool)}。题仍计入原分组，另记一份派生分组。
#   #26 → phyx 的 Mechanics_manip（2026-10-21 加）。能力体系 要的是「PhyX Mechanics（具身操作相关题）」，
#         Mechanics 采样集 133 题多是交通工具 / 抛体 / 图像读数 / 天体，人工挑出 39 道接触力学题：
#         推拉方块、摩擦、绳与滑轮、斜面、杠杆与扳手、手指夹持、箱体搬运、弹簧。
#         按数据集自带的 index 字段筛（采样 run 与全量 run 的 doc_id 口径不同，index 两边一致）。
#   #10 → epic 的 Orientation_object（2026-10-21 加）。朝向题 376 题里 227 题是人（OrientationHuman）、
#         Orientation 149 题里 81 题是动物，真正问物体朝向（椅子 / 沙发 / 车 / 鞋 / 冰箱…）的只有 68 题。
#         按 text_label 排除动物（已逐条核对，排除的 81 条全是动物、留下的全是物体）。
_PHYX_MANIP = {
        24, 30, 42, 64, 86, 96, 97, 111, 116, 129, 515, 525, 529, 787, 797, 1755, 1819, 1843, 1862,
        2200, 2209, 2232, 2366, 2367, 2453, 2455, 2462, 2470, 2474, 2489, 2497, 2509, 2519, 2521,
        2522, 2525, 2533, 2993, 2999}
_ANIMAL = re.compile(r"animal|\bcows?\b|\bbirds?\b", re.I)
DERIVED: dict[str, dict[str, tuple[str, Callable[[dict], bool]]]] = {
    "phyx_mc": {"Mechanics": ("Mechanics_manip", lambda d: int(d.get("index", -1)) in _PHYX_MANIP)},
    "epic_bench": {"Orientation": ("Orientation_object",
                                   lambda d: not _ANIMAL.search(str(d.get("text_label", ""))))},
}


# 补跑目录：task -> (build_report 里的 bench 词干, 只取这些分组值)。
# 补跑 run 关了采样，doc_id 就是原始下标；与采样集按原始下标去重后合并。
SUPPLEMENT: dict[str, tuple[str, set[str]]] = {
    "epic_bench": ("epicorient", {"Orientation", "OrientationHuman"}),
}


def _dig(out, field, key):
    """取该题的标量分。与 c_tier_specs.dig_value 同一逻辑，但 C 档的字段名固定。"""
    v = out.get(field)
    if isinstance(v, dict):
        if key is not None:
            v = v.get(key)
        else:
            for k in ct.SPARBENCH_VALUE_KEYS:
                if k in v:
                    v = v[k]
                    break
            else:
                v = None
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) else None


def collect_task(task: str) -> dict[str, dict[str, dict[str, float]]]:
    """{model_label: {group_value: {score, n}}}；任务无法判定口径时返回 {}。"""
    field_name, key = ct.VALUE_KEYS[task]
    group_field = dict((t, g) for t, g, _ in GROUPS)[task]
    # 要保留的分组值集合（**值**，不是字段名——写成字典推导会只剩最后一个键）
    wanted = {v for t, _g, vs in GROUPS if t == task for v in vs}

    out: dict[str, dict[str, dict[str, float]]] = {}
    # BENCH_TO_TASK 是 bench 键 -> task 名（如 "epic" -> "epic_bench"），这里要反过来
    tier_key = {v: k for k, v in ct.BENCH_TO_TASK.items()}[task]
    for label, suffix in br.MODELS.items():
        # 找到该模型的 C 档结果目录（用 build_report 的候选表，避免另拼一份命名规则）
        d = None
        for c in br.candidates(label, suffix, "C", tier_key):
            d = br.resolve(c)
            if d:
                break
        if d is None:
            continue

        rows = [r for t, r in br.load(d) if t == task]
        if not rows:
            continue

        try:
            ds, grader = ct.load_task_data(task)
        except Exception as exc:
            print(f"  [warn] {task}/{label}: 数据集加载失败（{type(exc).__name__}），跳过",
                  file=sys.stderr)
            continue
        if ds is None or grader is None:
            print(f"  [warn] {task}/{label}: 该任务不重算，跳过", file=sys.stderr)
            continue

        n_rows = len({r["doc_id"] for r in rows})
        conv = ct.doc_id_convention(task, ds, n_rows, rows=rows)
        if conv == "unknown":
            print(f"  [warn] {task}/{label}: doc_id 口径无法判定，跳过", file=sys.stderr)
            continue
        idx_map = ct.original_index_map(task, ds, n_rows)
        want = ct.sampled_original_indices(task, ds) if conv == "original" else None

        per: dict[str, list[float]] = collections.defaultdict(list)
        seen_real: set[int] = set()
        items = []  # (原始下标, 回复)
        for r in rows:
            did = r["doc_id"]
            if want is not None and did not in want:
                continue
            real = idx_map[did] if idx_map is not None else did
            items.append((real, r))
        if task in SUPPLEMENT:
            stem, sup_groups = SUPPLEMENT[task]
            sd = None
            for c in br.candidates(label, suffix, "C", stem):
                sd = br.resolve(c)
                if sd:
                    break
            if sd is not None:
                for t, r in br.load(sd):
                    if t == task and str(ds[r["doc_id"]].get(group_field, "")) in sup_groups:
                        items.append((r["doc_id"], r))
        for real, r in items:
            if real in seen_real:
                continue
            seen_real.add(real)
            if not (0 <= real < len(ds)):
                continue
            doc = dict(ds[real])
            grp = str(doc.get(group_field, ""))
            if grp not in wanted:
                continue
            try:
                v = _dig(grader(doc, [str(r.get("filtered_resps", ""))]), field_name, key)
            except Exception:
                v = None
            if v is not None:
                per[grp].append(v)
                der = DERIVED.get(task, {}).get(grp)
                if der and der[1](doc):
                    per[der[0]].append(v)

        if per:
            out[label] = {g: {"score": sum(v) / len(v) * 100, "n": len(v)}
                          for g, v in sorted(per.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data: dict[str, dict] = {}
    for task, group_field, _ in GROUPS:
        print(f"重算 {task}（按 {group_field} 分组）…")
        data[task] = collect_task(task)
        for label, groups in data[task].items():
            gs = "  ".join(f"{g}={v['score']:.2f}(n={v['n']})" for g, v in groups.items())
            print(f"  {label:<22s} {gs}")

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
        return

    n = _write_summary(data, pathlib.Path(args.out))
    print(f"\n写入 {pathlib.Path(args.out) / 'summary_c_subscores.csv'}  （{n} 行）")


def _write_summary(data: dict, out: pathlib.Path) -> int:
    """把 collect_task 的结果写成能力表要读的长表。返回行数。"""
    rows = []
    task_to_bench = {v: k for k, v in ct.BENCH_TO_TASK.items()}
    for task, _gf, _vs in GROUPS:
        groups: set[str] = set()
        for per in data[task].values():
            groups |= set(per)
        for g in sorted(groups):
            row = {"benchmark": task_to_bench[task], "group": g}
            for label in br.MODELS:
                v = data[task].get(label, {}).get(g)
                row[label] = f"{v['score']:.2f}" if v else ""
                row[f"{label}__n"] = str(v["n"]) if v else ""
            rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary_c_subscores.csv"
    # __n 列必须写出来：能力表要拿它填「题数」。之前只写分数，extrasaction="ignore"
    # 把 __n 静默丢掉了，于是 #4/#7/#10/#26/#42/#43/#45 的题数一直是空。
    fields = ["benchmark", "group", *br.MODELS, *[f"{m}__n" for m in br.MODELS]]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_outputs(out_dir: pathlib.Path) -> int:
    """重算 C 档子分数并落盘。dump_reports.py 调这个（无参，静默重算）。"""
    data = {task: collect_task(task) for task, _gf, _vs in GROUPS}
    return _write_summary(data, out_dir)


if __name__ == "__main__":
    main()
