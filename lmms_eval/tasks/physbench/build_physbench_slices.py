#!/usr/bin/env python3
"""PhysBench 按文档 P0 条目筛选切片。

数据源
------
**官方**：`physical-superintelligence-lab/PhysBench`（ICLR 2025，⭐94）
的 `eval/physbench/test_answer.json`。

> 注意：能力体系（CAPABILITIES.md） 里写的 repo 是 `USC-PSI-Lab/PhysBench`，
> **该路径 404 不存在**。实际 repo 名见上。

**实际取数**：HF 上的 `lmms-lab-eval/PhysBench`。已核实与官方逐行等价：

    官方 test_answer.json          10002 条
    HF full split                  10002 条
    逐行比对 10002 × 4 字段        全部一致 ✓

⚠️ 本脚本此前误用 `test` split（9802 条），**漏了 200 条**（即 `val` split）。
   现改用 `full`，与官方口径一致。

字段（官方 JSON 与 HF 一致）
----------------------------
    idx / answer / task_type / sub_type / ability_type
    （HF 另有 split / question / file_name）

    task_type   : dynamics(3003) property(2578) scene(2283) relationships(2138)
    sub_type    : motion(1446) attribute(1409) light(1099) viewpoint(1061)
                  manipulation(829) collision(663) fluid(649) number(577)
                  throwing(419) others(352) color(299) location(292)
                  depth(287) mass(286) chemistry(67) temperature(59) distance(54) air(47)
    ability_type: reasoning(2087) static(1923) prediction(1454)
                  comparison(1433) identify(1145) perception(968)
                  judgment(777) dynamic(215)

条目判据
--------
| 条目 | 能力 | 判据 | 性质 |
|---|---|---|---|
| #5  | 客体物理属性   | `property` + `sub_type in {mass, attribute}` | 字段精确 |
| #35 | 物体恒常性     | `property` + `sub_type == mass` | 字段精确（**#5 的子集**）|
| #39 | 单体力学效果   | `dynamics` + `{manipulation, throwing}` + **排除多物体** | 启发式 |
| #40 | 多体交互预测   | `dynamics` + `collision` + `prediction` + **保留多物体** | 启发式 |
| #41a | 刚体/流体/柔体·预测 | `dynamics` + `fluid` + `prediction` | 字段精确 |
| #41b | 刚体/流体/柔体·感知 | `dynamics` + `fluid` + `{perception, reasoning, judgment}` | 字段精确 |
| #26 | 物理知识       | **不建** —— 文档未给可机械判据 | — |

### #39 / #40 的互斥与"启发式"说明

文档 #39 说"只评估题目指定**单个**目标的响应；不把所有 dynamics 题并入"；
#40 说"再保留**多个对象**响应/联动链问题"，同时承认"不存在已核验 multi_body 字段"。

故两者用**同一个代理判据**划分，方向相反：

* 判据：题干中出现 **≥2 个颜色词**（blue/brown/green/gray/...）视为"多物体"
* #39 取**不满足**者（单体）
* #40 取**满足**者（多体）

**这是启发式，不是文档认可的判据。** 局限：
* 题干未写颜色的多物体题会被误判为单体
* 颜色词可能指同一物体的不同部分

实测：
* #39 候选 1250 条 → 排除多物体后 **986**
* #40 候选 495 条 → 保留多物体后 **310**

### #41 为何拆两个

文档 #41 说"水位/流体可有 perception 问题，**不一律限 prediction**"。
实测 fluid 649 条的 ability_type：

    prediction 559 / perception 74 / reasoning 15 / judgment 1

故拆为 `fluid_prediction`（559）与 `fluid_perception`（90），不整取。

### #5 与 #35 的嵌套

`#35 ⊂ #5` —— 这是**文档定义如此**（#5 含 mass+attribute，#35 只要 mass）。
**报告时不可将两者相加**，会重复计数。

### #26 为何不建

文档只写"按原物理规律应用定义审查题意；先导出 task_type/sub_type/ability_type 分布"，
**没有给出可机械执行的判据**。实测 scene/relationships 共 4421 条，
跨 light/viewpoint/motion/depth/location/air/size 七类，无"物理规律应用"维度。
需人工审查或另行定义规则。

用法
----
    python build_physbench_slices.py
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from typing import Callable

HERE = pathlib.Path(__file__).resolve().parent
# HF 的 **full** split（10002 条），与官方 test_answer.json 逐行等价。
# 此前误用 test split（9802 条），漏了 val 的 200 条。
SRC = HERE / "data" / "full-00000-of-00001.parquet"

# 颜色词 —— 用于 #39/#40 的"多物体"代理判据
_COLORS = {
    "blue", "brown", "green", "gray", "grey", "red", "yellow",
    "white", "black", "pink", "purple", "orange", "cyan", "silver", "golden",
}


def n_colored_objects(question: str) -> int:
    """题干中出现的不同颜色词数量（"多物体"的代理判据）。"""
    q = str(question or "")
    return len({c for c in _COLORS if re.search(r"\b" + c + r"\b", q, re.I)})


def _is_multi(question: str) -> bool:
    return n_colored_objects(question) >= 2


# (输出名, 中文说明, 条目编号, 判据, 是否用启发式)
SLICES: list[tuple[str, str, str, Callable[[dict], bool], bool]] = [
    (
        "property_object", "客体物理属性", "#5",
        lambda d: d["task_type"] == "property" and d["sub_type"] in {"mass", "attribute"},
        False,
    ),
    (
        "single_body_dynamics", "单体力学效果", "#39",
        lambda d: (
            d["task_type"] == "dynamics"
            and d["sub_type"] in {"manipulation", "throwing"}
            and not _is_multi(d.get("question", ""))
        ),
        True,
    ),
    (
        "multi_body_collision", "多体交互预测", "#40",
        lambda d: (
            d["task_type"] == "dynamics"
            and d["sub_type"] == "collision"
            and d["ability_type"] == "prediction"
            and _is_multi(d.get("question", ""))
        ),
        True,
    ),
    (
        "fluid_prediction", "刚体/流体/柔体·预测", "#41",
        lambda d: (
            d["task_type"] == "dynamics"
            and d["sub_type"] == "fluid"
            and d["ability_type"] == "prediction"
        ),
        False,
    ),
    (
        "fluid_perception", "刚体/流体/柔体·感知", "#41",
        lambda d: (
            d["task_type"] == "dynamics"
            and d["sub_type"] == "fluid"
            and d["ability_type"] in {"perception", "reasoning", "judgment"}
        ),
        False,
    ),
]


def main() -> None:
    from datasets import load_dataset

    if not SRC.is_file():
        raise SystemExit(f"找不到 {SRC}")

    ds = load_dataset("parquet", data_files=str(SRC), split="train")
    print(f"输入 {len(ds)} 题 <- {SRC.name}（HF full split）\n")

    seen: Counter[int] = Counter()
    for name, cn, entry, pred, heuristic in SLICES:
        rows = [dict(d) for d in ds if pred(d)]
        out = HERE / "data" / f"physbench_{name}.jsonl"
        idx_out = HERE / "data" / f"physbench_{name}_indices.json"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        idx_out.write_text(
            json.dumps(sorted(int(r["idx"]) for r in rows)), encoding="utf-8"
        )
        for r in rows:
            seen[int(r["idx"])] += 1
        tag = "  ⚠启发式" if heuristic else ""
        print(f"{entry:<5s} {name:<24s} {len(rows):>5d} 题  ({cn}){tag}")

    dup = {k: v for k, v in seen.items() if v > 1}
    print(f"\n跨切片重复: {len(dup)} 个")
    if dup:
        print("   （#35 ⊂ #5 属文档定义如此，报告时不可相加）")
    print("\n未建：")
    print("   #26 物理知识 —— 文档未给可机械判据，需人工审题")


if __name__ == "__main__":
    main()
