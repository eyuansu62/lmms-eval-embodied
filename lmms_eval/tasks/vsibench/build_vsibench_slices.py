#!/usr/bin/env python3
"""VSI-Bench 按文档 P0 条目筛选切片。

文档要求
--------
能力体系（CAPABILITIES.md）里，VSI-Bench 共 3 条 P0 条目，
都要求按 `question_type` 筛选，且明确"config full；debiased 仅作单独对照"。

实测 `test.jsonl`（5130 题）的字段与分布：

    id / dataset / scene_name / question_type / question / ground_truth / options

    question_type 分布：
      object_size_estimation      953      object_abs_distance        834
      object_rel_distance         710      obj_appearance_order       618
      object_counting             565      object_rel_direction_medium 378
      object_rel_direction_hard   373      room_size_estimation       288
      object_rel_direction_easy   217      route_planning             194

    dataset 分布：scannet(2071) arkitscenes(1601) scannetpp(1458)

条目判据
--------
| 条目 | 能力 | question_type 条件 | 命中 |
|---|---|---|---|
| #8  | 相对位置      | `object_rel_direction_{easy,medium,hard}` | 968 |
| #9  | 相对距离      | `object_rel_distance` + `object_abs_distance` | 1544 |
| #44 | 2D/3D理解与转换 | `object_size_estimation` + `room_size_estimation` | 1241 |

### #9 的"两类分开计分"

文档 #9 明确要求"两类分开计分"。现有 `utils.py` 的 `metric_list` 本就
分别输出 `object_abs_distance_mra` 与 `object_rel_distance_accuracy`，
且 `_compute_all_subscores` 按 `question_type` 分别统计 —— 所以
**一个切片 yaml 即可产出两个独立分数**，不必拆成两个 task。

### 媒体

VSI-Bench 的视频不在 parquet 里，需下载三个 zip 并解压到：

    ~/.cache/huggingface/vsibench/<dataset>/<scene_name>.mp4

（`utils._vsibench_video_path` 按此路径查找；缺失会抛 `FileExistsError`。）

用法
----
    python build_vsibench_slices.py --input /path/to/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent

# (输出名, 中文说明, 条目编号, question_type 集合)
SLICES = [
    (
        "rel_direction", "相对位置", "#8",
        {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"},
    ),
    (
        "rel_distance", "相对距离", "#9",
        {"object_rel_distance", "object_abs_distance"},
    ),
    (
        "size_estimation", "2D/3D理解与转换", "#44",
        {"object_size_estimation", "room_size_estimation"},
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="VSI-Bench 的 test.jsonl")
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    if not src.is_file():
        raise SystemExit(f"找不到 {src}")

    rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
    print(f"输入 {len(rows)} 题 <- {src}\n")

    used: Counter[int] = Counter()
    for name, cn, entry, qtypes in SLICES:
        hit = [r for r in rows if r.get("question_type") in qtypes]
        out = HERE / "data" / f"vsibench_{name}.jsonl"
        idx_out = HERE / "data" / f"vsibench_{name}_indices.json"
        with out.open("w", encoding="utf-8") as fh:
            for r in hit:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        idx_out.write_text(
            json.dumps(sorted(int(r["id"]) for r in hit)), encoding="utf-8"
        )
        for r in hit:
            used[int(r["id"])] += 1

        per = Counter(r["question_type"] for r in hit)
        print(f"{entry:<5s} {name:<16s} {len(hit):>5d} 题  ({cn})")
        for k, v in sorted(per.items()):
            print(f"        {v:>5d}  {k}")

    dup = {k: v for k, v in used.items() if v > 1}
    print(f"\n跨切片重复: {len(dup)} 个 {'(无重叠)' if not dup else ''}")
    print(f"三切片合计 {len(used)} 题")


if __name__ == "__main__":
    main()
