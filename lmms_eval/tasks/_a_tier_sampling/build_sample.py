#!/usr/bin/env python3
"""A 档各 benchmark 等比采样到统一预算。

为什么需要
----------
A 档接入的题量差异极大（RefSpatial 7 道 vs EgoTaskQA 33830 道）。
直接汇总时 EgoTaskQA 会主导整体指标，其它 benchmark 的信号被淹没。
本脚本对每个 benchmark 做**等比采样**，使各方样本量可比、总量可控。

方案（「切法 A」）
------------------
* **每个 benchmark 一个预算**（默认 800），内部按切片规模**等比分配**
* 不强制顶到上限 —— 小于预算的 benchmark 全留
* 每切片设**下限**，避免小切片被压到无统计意义
* 固定随机种子，可复现

分配用**最大余数法**，保证各切片之和精确等于预算。

数据文件从哪来
--------------
从各 task 的 yaml 里解析 `data_files` —— 这是**权威来源**。
不能直接 glob 目录：例如 `causalphys/data/` 下同时存在
`mechanics_reasoning.jsonl`（298 条，未被任何 task 使用）与
`mechanics_reasoning_attribution.jsonl`（153 条，实际在用）。

备份
----
执行前会把原始文件复制到 `<task>/data/_full/`，可随时还原。
`--dry-run` 只看方案不动文件。

⚠️ 注意事项
----------
* 已跑出的**全量成绩与采样后不可直接比较**，需重跑对齐。
* **派生切片**（如 `causalphys_multi_hop` 取自 spatial_manipulation /
  temporal_shifting）独立采样后，其行不再保证是其它切片采样结果的子集。
  这影响「从切片反查原行」，不影响各切片自身评测的有效性。
* EgoTaskQA 的三个切片本身**互相重叠**（#13 的判据跨越所有 type），
  独立采样后重叠结构在期望上保持，但精确配对会被打破。
  EgoTaskQA 当前无视频无法运行，此项暂无实际影响。

用法
----
    python build_sample.py --dry-run   # 只看分配方案
    python build_sample.py             # 执行
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
TASKS = HERE.parent          # lmms_eval/tasks
REPO = TASKS.parent.parent   # 仓库根

BUDGET = 800
SEED = 20260926
FLOOR = 50

# benchmark 名 -> task 目录名。用于分组与报告。
BENCH_OF_DIR = {
    # A 档
    "causalphys": "CausalPhys",
    "physbench": "PhysBench",
    "vsibench": "VSI-Bench",
    "egotaskqa": "EgoTaskQA",
    "momagraph_bench": "MomaGraph-Bench",
    "negvqa": "NegVQA",
    "refspatial": "RefSpatial",
    # B 档
    "robobench": "RoboBench",
    "mvrobobench": "MV-RoboBench",
    "pio": "Point-It-Out",
}

# 只处理白名单内的目录。
# **必须用白名单**：直接 glob `*/*.yaml` 会把仓库里其它 task 也扫进来
# （实测会拉到 tau2_bench / vending_bench2 / hd_epic 等），
# 其中 hd_epic 的 yaml 还写着外部机器的绝对路径。
A_TIER_DIRS = set(BENCH_OF_DIR)


def discover() -> dict[str, list[tuple[str, pathlib.Path]]]:
    """从 yaml 里解析各 task 实际使用的 jsonl。

    返回 ``{benchmark: [(task_name, 数据文件绝对路径), ...]}``。

    只扫 ``A_TIER_DIRS`` 白名单内的目录。
    """
    found: dict[str, list[tuple[str, pathlib.Path]]] = defaultdict(list)

    for dirname in sorted(A_TIER_DIRS):
        bench = BENCH_OF_DIR[dirname]
        for y in sorted((TASKS / dirname).glob("*.yaml")):
            txt = y.read_text(encoding="utf-8", errors="replace")
            if "data_files:" not in txt:
                continue
            m = re.search(r"data_files:\s*\n\s*\S+:\s*(\S+)", txt)
            if not m:
                continue
            rel = m.group(1).strip()
            if not rel.endswith(".jsonl"):
                continue  # parquet / indices.json 等另行处理

            tm = re.search(r'^task:\s*"?([^"\n]+?)"?\s*$', txt, re.M)
            name = tm.group(1).strip() if tm else y.stem

            path = pathlib.Path(rel)
            if not path.is_absolute():
                path = REPO / rel
            # 只接受仓库内的路径，挡掉 yaml 里可能写死的外部路径
            try:
                path.resolve().relative_to(REPO.resolve())
            except ValueError:
                print(f"  [跳过] 路径不在仓库内: {y.name} -> {rel}")
                continue
            found[bench].append((name, path))

    # EgoTaskQA 的切片 yaml 尚未生成（视频未授权，一直挂起），
    # 但切片 jsonl 已存在 —— 直接扫目录补上。
    ego_dir = TASKS / "egotaskqa" / "data"
    if ego_dir.is_dir():
        for j in sorted(ego_dir.glob("egotaskqa_*.jsonl")):
            found["EgoTaskQA"].append((j.stem, j))

    return found


def allocate(sizes: dict[str, int], budget: int, floor: int) -> dict[str, int]:
    """按规模等比分配，最大余数法，附下限。

    * 总和 <= budget 时全留
    * 否则等比分配；低于下限的切片抬到下限，
      多出的部分从规模最大的切片回收（迭代直至收敛）
    """
    total = sum(sizes.values())
    if total <= budget:
        return dict(sizes)

    alloc = {k: v * budget / total for k, v in sizes.items()}

    # 迭代应用下限
    for _ in range(20):
        short = {k: floor - a for k, a in alloc.items() if a < floor}
        if not short:
            break
        deficit = sum(short.values())
        donors = sorted(
            (k for k in alloc if k not in short), key=lambda k: -alloc[k]
        )
        if not donors:
            break
        for k, d in short.items():
            alloc[k] = floor
        # 从最大的切片按比例回收
        dsum = sum(alloc[k] for k in donors)
        for k in donors:
            alloc[k] = max(floor, alloc[k] - deficit * alloc[k] / dsum)

    # 最大余数法取整到总和 = budget
    base = {k: int(a) for k, a in alloc.items()}
    rem = budget - sum(base.values())
    frac = sorted(alloc, key=lambda k: -(alloc[k] - int(alloc[k])))
    i = 0
    while rem > 0 and frac:
        k = frac[i % len(frac)]
        if base[k] < sizes[k]:
            base[k] += 1
            rem -= 1
        i += 1
        if i > 10000:
            break
    return {k: min(v, sizes[k]) for k, v in base.items()}


def sample_jsonl(path: pathlib.Path, n: int, rng: random.Random) -> int:
    """就地采样 jsonl，返回实际写入行数。原文件备份到 data/_full/。

    幂等：若备份已存在，先从备份还原再采样，
    避免重复运行导致「在已采样的数据上再采」。
    """
    backup = path.parent / "_full" / path.name

    if backup.is_file():
        # 已采过 —— 从备份还原，保证每次采样都基于全量
        path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [l for l in path.open(encoding="utf-8") if l.strip()]
    if n >= len(lines):
        return len(lines)

    picked = rng.sample(lines, n)
    path.write_text("".join(picked), encoding="utf-8")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只看分配方案，不动文件")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    found = discover()
    if not found:
        raise SystemExit("未从任何 yaml 解析到 jsonl 数据文件")

    manifest = {"budget": args.budget, "seed": args.seed, "floor": FLOOR, "benchmarks": {}}

    print(f"预算 {args.budget}/benchmark   种子 {args.seed}   下限 {FLOOR}\n")

    for bench in sorted(found):
        items = found[bench]
        # 同一个数据文件可能被多个 task 引用（罕见），去重
        uniq: dict[pathlib.Path, list[str]] = defaultdict(list)
        for name, path in items:
            uniq[path].append(name)

        sizes = {}
        for path, names in uniq.items():
            if not path.is_file():
                print(f"  [跳过] 文件不存在: {path}")
                continue
            sizes[str(path)] = sum(1 for l in path.open(encoding="utf-8") if l.strip())

        if not sizes:
            continue

        total = sum(sizes.values())
        alloc = allocate(sizes, args.budget, FLOOR)

        print(f"### {bench}  （总 {total} → 采 {sum(alloc.values())}）")
        rng = random.Random(args.seed)
        bench_rec = {"original_total": total, "sampled_total": sum(alloc.values()), "slices": {}}

        for key in sorted(alloc, key=lambda k: -sizes[k]):
            path = pathlib.Path(key)
            n_orig, n_new = sizes[key], alloc[key]
            names = uniq[path]
            label = names[0]
            pct = n_new / n_orig * 100 if n_orig else 0
            # 95% CI 半宽（p=0.5 最坏情况）
            ci = 1.96 * (0.25 / n_new) ** 0.5 * 100 if n_new else 0
            print(
                f"   {label:<42s} {n_orig:>6d} → {n_new:>5d}  ({pct:>5.1f}%)  ±{ci:.1f}%"
            )
            bench_rec["slices"][label] = {
                "file": str(path.relative_to(REPO)),
                "original": n_orig,
                "sampled": n_new,
                "ci95_halfwidth_pct": round(ci, 2),
            }
            if not args.dry_run:
                sample_jsonl(path, n_new, rng)
        print()
        manifest["benchmarks"][bench] = bench_rec

    (HERE / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"清单 -> {HERE / 'sample_manifest.json'}")
    if args.dry_run:
        print("\n[DRY RUN] 未改动任何文件")
    else:
        print("\n完成。原文件已备份到各任务目录的 data/_full/")


if __name__ == "__main__":
    main()
