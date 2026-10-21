#!/usr/bin/env python3
"""PIO · 序数指代切片（#20）—— 从 pio_attribute_reference 的 95 道 s1 题里人工审出的子集。

能力体系（CAPABILITIES.md）#20 指定的 PIO 来源写作：「s1_or_s2=="s1"；按 lang 审查序数表达」，
也就是**不能靠正则粗筛**，要逐条看题干。这里把 95 条全部过了一遍，留下明确带名次 /
极值 / 中间位表达的 18 条；边界情形（只含 left/right/top/bottom 方位而不是名次的）
一律不算，避免把「方位指代」混进「序数指代」。

判定口径
--------
算：rightmost / leftmost / highest / lowest / tallest / first / second / third / middle
不算：left / right / top / bottom / inside / nearest（这些是方位或距离，#19 / #9 管）
      含 top / bottom 但同时含名次的算（例：「the second section of the staircase」
      里的 top 不是方位词）

产物：data/pio_ordinal_reference.jsonl（18 行，id 沿用 s1 全量里的原始下标）。
分数**不发新请求**：dump_reports 直接从已有的 pio_attribute_reference 样本里按 doc_id 取。

用法：
    python build_pio_ordinal.py [--src /tmp/pio_s1s2.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# 人工审出的下标（相对 s1/restriction+object part 全量 95 条的顺序，= doc_id）
ORDINAL_IDS = [
    43,  # The rightmost slot on the toaster
    44,  # the toaster's leftmost slot
    48,  # the handle of the rightmost cabinet
    50,  # The cap of the rightmost jar on the red plate
    55,  # The handle of the second-level drawer
    58,  # The pure water on the middle level.
    61,  # the leftmost plate
    62,  # the highest piano key
    63,  # the lowest key on the piano
    70,  # The second layer of the drawer
    73,  # the first flight of the staircase
    74,  # the second section of the staircase
    78,  # the tallest pile of paper cups
    81,  # the tallest pile of paper cups（另一张图，题干相同，不是重复题）
    82,  # the lowest pile of paper cups
    83,  # the pillow at the leftmost position on the sofa
    84,  # the middle pillow on the sofa
    87,  # The middle shelf of the open cabinet
]

# 边界情形：带 top/bottom/upper/lower 但只有方位含义，没有名次 —— 不计入 #20
#   54 the handle on the top drawer / 56 the handle of the top-level drawer
#   69 the top compartment of the drawer / 36 the upper-left stove
#   37 the lower-right stove / 51 the stove in the upper-left corner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/pio_s1s2.json")
    args = ap.parse_args()

    raw = json.loads(pathlib.Path(args.src).read_text())
    s1 = [d for d in raw if d["s1_or_s2"] == "s1" and d["subclasses"] in ("restriction", "object part")]
    assert len(s1) == 95, len(s1)

    out = DATA_DIR / "pio_ordinal_reference.jsonl"
    with out.open("w") as fh:
        for i in ORDINAL_IDS:
            item = s1[i]
            fh.write(json.dumps({
                # id 用原始下标，dump_reports 靠它去 pio_attribute_reference 的样本里取分
                "id": f"pio_ordinal_reference/{i}",
                "src_doc_id": i,
                "image_path": item["image_path"],
                "height": item["height"],
                "width": item["width"],
                "lang": item["lang"],
                "polygon": item["polygon"],
                "s1_or_s2": item["s1_or_s2"],
                "subclasses": item["subclasses"],
                "dataset_name": item.get("dataset_name", ""),
            }, ensure_ascii=False) + "\n")
    print(f"pio_ordinal_reference: {len(ORDINAL_IDS)} 行 → {out}")


if __name__ == "__main__":
    main()
