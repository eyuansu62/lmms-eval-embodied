#!/usr/bin/env python3
"""生成 16 个子类的 task yaml。

手写 16 份 yaml 容易抄错（task 名、bound 函数名、data_files 路径），
所以统一由这里生成。改了子集或元信息后重跑本脚本。

用法：
    python gen_yamls.py
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# 子集名 -> (中文说明, 对应 P0 能力, 备注)
SUBSETS = {
    # --- Anticipation ---
    "collision_prediction": (
        "碰撞预测",
        "P0 #27 轨迹理解与预测",
        "300 题，当前帧判碰撞的二分类（A/B 各 150）",
    ),
    "fluid_flow": (
        "流体运动",
        "P0 #28 单体力学效果",
        "200 题，预测流体行为",
    ),
    "deformation": (
        "形变预测",
        "P0 #28 单体力学效果",
        "200 题，预测物体形变结果；媒体为 data/<id>/ 目录内多图",
    ),
    # --- Intervention ---
    "action_substitution": (
        "动作替换",
        "P0 #15 因果链推理",
        "99 题，替换动作后的结果",
    ),
    "spatial_manipulation": (
        "空间操作",
        "P0 #16 归因理解",
        "151 题，空间关系随操作的变化",
    ),
    "temporal_shifting": (
        "时间偏移",
        "P0 #16 归因理解",
        "149 题，时间尺度变化后的结果",
    ),
    # --- Perception ---
    "mechanics_reasoning": (
        "力学推理",
        "P0 #16 归因理解",
        "298 题，由结果反推原因；**需经归因筛选**，见 build_attribution.py",
    ),
    "containability": (
        "容置性",
        "P0 #4 客体视觉属性",
        "201 题，物体能否容纳/堆叠；媒体为单图",
    ),
}

TEMPLATE = '''# CausalPhys · {category}/{subcat}
#
# {cn}（{p0}）{note}
# 论文：https://arxiv.org/html/2606.05966v1
# 代码：https://github.com/haorentang/CausalPhys
# 数据：https://huggingface.co/datasets/haorentang/causalphys
#
# 媒体按需从 HF 镜像拉取并缓存，不预拉全量。
# 媒体形态的实测记录见 DATA_SHAPE.md。

dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/causalphys/data/{name}.jsonl
task: "causalphys_{name}"
test_split: test
output_type: generate_until

doc_to_visual: !function utils.causalphys_{name}_doc_to_visual
doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target

generation_kwargs:
  max_new_tokens: 64
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false

process_results: !function utils.process_results
metric_list:
  - metric: causalphys_accuracy
    aggregation: !function utils.aggregate_results
    higher_is_better: true

lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\\nAnswer with the option's letter from the given choices directly."

metadata:
  version: 0.2
  note: "{note}"
'''

# 子集 -> 仓库内目录（用于 yaml 注释里的 category/subcat）
#
# **只含文档显式点名的 8 个子类**（#11 #23 #24 #37 #41 五条条目）。
# 另有 4 个子类只出现在 #21 的通配池里（intention_speculation /
# viewpoint_transformation / scene_reconstruction / optics），
# 它们不作为独立 task 存在 —— 只经 #21 筛选后进入 causalphys_multi_hop。
# 数据保留在 data/ 下，供 build_multihop.py 重建池子。
REPO_DIRS = {
    "collision_prediction": ("Anticipation", "Collision_Prediction"),
    "fluid_flow": ("Anticipation", "Fluid_Flow"),
    "deformation": ("Anticipation", "deformation"),
    "action_substitution": ("Intervention", "action_substitution"),
    "spatial_manipulation": ("Intervention", "spatial_manipulation"),
    "temporal_shifting": ("Intervention", "temporal_shifting"),
    "mechanics_reasoning": ("Perception", "Mechanics_Reasoning"),
    "containability": ("Perception", "containability"),
}


def main() -> None:
    for name, (cn, p0, note) in SUBSETS.items():
        category, subcat = REPO_DIRS[name]
        content = TEMPLATE.format(
            category=category, subcat=subcat, cn=cn, p0=p0, note=note, name=name
        )
        out = HERE / f"causalphys_{name}.yaml"
        out.write_text(content, encoding="utf-8")
        print(f"  {out.name}")
    print(f"\n生成 {len(SUBSETS)} 个 yaml")


if __name__ == "__main__":
    main()
