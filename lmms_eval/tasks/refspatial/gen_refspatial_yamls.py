#!/usr/bin/env python3
"""生成 RefSpatial-Expand-Bench 各 P0 切片的 task yaml。

用法：
    python gen_refspatial_yamls.py
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

SLICES = {
    "attribute_reference": (
        "属性指代", "#19",
        "id IN {1,3,9}", 3,
        'id=1 "the brown car" / id=3 "the furthest brown car" / id=9 "the white car on the left"',
    ),
    "ordinal_reference": (
        "序数指代", "#20",
        "id IN {0,2,8,13}", 4,
        'id=0 "third car" / id=2 "fifth car" / id=8 "third building" / id=13 "second closest tree"',
    ),
}

TEMPLATE = '''# RefSpatial-Expand-Bench · {cn}（P0 {entry}）
#
# 筛选切片：{cond}
# 题数：{n}
# 样本：{sample}
#
# 数据：https://huggingface.co/datasets/JingkunAn/RefSpatial-Expand-Bench
# 论文：https://arxiv.org/abs/2507.22076
#
# ⚠️ **样本量极小（{n} 条），分数仅作定性参考**，单题波动 ±14% 以上。
#    文档「子集定位」#19/#20 只给了 id 白名单，并注明
#    "其余 location 需审标后加入" —— 那 233 条需人工审标，脚本无法代劳。
#
# ⚠️ **数据源与内置 task 不同**：
#    内置 refspatial_* 用 BAAI/RefSpatial-Bench，
#    本切片按文档改用 JingkunAn/RefSpatial-Expand-Bench（split=location）。
#
# 过滤逻辑见 utils_filter.py（按 id 白名单），判分复用本目录原有 utils.py。

dataset_path: JingkunAn/RefSpatial-Expand-Bench
dataset_kwargs:
  name: location
full_docs: true
output_type: generate_until
task: "refspatial_{name}"
test_split: location

process_docs: !function utils_filter.process_docs_{name}
doc_to_visual: !function utils.refspatial_doc_to_visual
doc_to_text: !function utils.refspatial_doc_to_text
doc_to_target: ""
process_results: !function utils.refspatial_process_results

metric_list:
  - metric: refspatial_acc
    aggregation: !function utils.refspatial_aggregate_results
    higher_is_better: true

generation_kwargs:
  max_new_tokens: 1024

metadata:
  version: 0.1
  prompt_suffix_type: original
  note: 'P0 {entry} {cn}；{n} 条；判据 {cond}；样本量极小仅供参考'
'''


def main() -> None:
    for name, (cn, entry, cond, n, sample) in SLICES.items():
        content = TEMPLATE.format(
            cn=cn, entry=entry, cond=cond, n=n, name=name, sample=sample
        )
        out = HERE / f"refspatial_{name}.yaml"
        out.write_text(content, encoding="utf-8")
        print(f"  {out.name}   {n} 条")

    print(f"\n生成 {len(SLICES)} 个 yaml")


if __name__ == "__main__":
    main()
