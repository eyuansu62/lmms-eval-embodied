#!/usr/bin/env python3
"""生成 PhysBench 各 P0 切片的 task yaml。

手写 5 份 yaml 容易抄错（task 名、数据路径、判据对应关系），统一由这里生成。
改了切片或元信息后重跑。

用法：
    python gen_physbench_yamls.py
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# 切片名 -> (中文说明, 条目编号, 判据描述, 题数)
SLICES = {
    "property_object": (
        "客体物理属性", "#5",
        'task_type=="property" AND sub_type IN {"mass","attribute"}', 1702,
    ),
    "single_body_dynamics": (
        "单体力学效果", "#39",
        'task_type=="dynamics" AND sub_type IN {"manipulation","throwing"} AND 排除多物体', 968,
    ),
    "multi_body_collision": (
        "多体交互预测", "#40",
        'task_type=="dynamics" AND sub_type=="collision" AND ability_type=="prediction" AND 保留多物体', 350,
    ),
    "fluid_prediction": (
        "刚体/流体/柔体·预测", "#41",
        'task_type=="dynamics" AND sub_type=="fluid" AND ability_type=="prediction"', 559,
    ),
    "fluid_perception": (
        "刚体/流体/柔体·感知", "#41",
        'task_type=="dynamics" AND sub_type=="fluid" AND ability_type IN {"perception","reasoning","judgment"}', 90,
    ),
}

# 判据是否含启发式（在 yaml 里明确标注，避免被当成字段精确筛选）
HEURISTIC = {"single_body_dynamics", "multi_body_collision"}

# 嵌套关系提示（报告时不可相加）
NESTED_NOTE = {
    "property_object": (
        "\n# 说明：原 #35 object_permanence（sub_type==mass，293 题）是本切片的子集，"
        "\n#     2026-10-16 已删除该切片——它 100% 含于本切片，且不含守恒/遮挡题，"
        "\n#     不构成独立的「物体恒常性」评测（详见 能力体系（CAPABILITIES.md） #35 行的边界说明）。\n"
    ),
}

TEMPLATE = '''# PhysBench · {cn}（P0 {entry}）
#
# 筛选切片：{cond}
# 题数：{n}
#
# 论文：https://arxiv.org/abs/2501.16411
# 代码：https://github.com/physical-superintelligence-lab/PhysBench
# 数据：https://huggingface.co/datasets/lmms-lab-eval/PhysBench
#
# ⚠️ 这是**按 能力体系（CAPABILITIES.md） 条目筛选的切片**，不是官方子集。
#    筛选逻辑见 build_physbench_slices.py。
{heur}{nested}#
# 数据源核对：HF 的 full split（10002 条）已与官方 test_answer.json
# 逐行比对 4 字段，**全部一致**。切片由 full split 生成。
#
# 媒体：PhysBench 的 image.zip / video.zip 需自行下载并解压到
#       HF cache 的 physbench/ 目录下（见 utils._resolve_media_path）。
#
# 注：切片字段与原 parquet 一致，直接复用本目录的 utils.py 判分。

dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/physbench/data/physbench_{name}.jsonl
task: "physbench_{name}"
test_split: test
output_type: generate_until

doc_to_visual: !function utils.physbench_doc_to_visual
doc_to_text: !function utils.physbench_doc_to_text
doc_to_target: "answer"

generation_kwargs:
  max_new_tokens: 65536
  temperature: 0
  do_sample: false

process_results: !function utils.physbench_process_results
metric_list:
  - metric: physbench_accuracy
    aggregation: !function utils.physbench_aggregate_results
    higher_is_better: true

lmms_eval_specific_kwargs:
  default:
    post_prompt: "\\nAnswer with the option's letter from the given choices directly. You can only answer one letter from A, B, C, or D."

metadata:
  version: 0.2
  note: 'P0 {entry} {cn}；{n} 题；判据 {cond}{heur_tag}'
'''

_HEUR_WARN = """
# ⚠️ **判据含启发式**：用"题干出现≥2 个颜色词"判断是否多物体。
#    这是代理判据，不是文档认可的字段 —— 未写颜色的多物体题会被误判。
"""

_NESTED_NOTE = {
    "property_object": """
# 说明：原 #35 object_permanence（sub_type==mass，293 题）是本切片的**子集**，
#    2026-10-16 已删除该切片：它 100% 含于本切片，且不含守恒/遮挡题，不构成
#    独立的「物体恒常性」评测（见 能力体系（CAPABILITIES.md） #35 行的边界说明）。
""",
}


def main() -> None:
    for name, (cn, entry, cond, n) in SLICES.items():
        content = TEMPLATE.format(
            cn=cn,
            entry=entry,
            cond=cond,
            n=n,
            name=name,
            heur=_HEUR_WARN if name in HEURISTIC else "",
            heur_tag="；判据含启发式" if name in HEURISTIC else "",
            nested=_NESTED_NOTE.get(name, ""),
        )
        out = HERE / f"physbench_{name}.yaml"
        out.write_text(content, encoding="utf-8")
        print(f"  {out.name}   {n:>5d} 题")

    print(f"\n生成 {len(SLICES)} 个 yaml")
    print("未生成：#26 物理知识 —— 文档未给可机械判据")


if __name__ == "__main__":
    main()
