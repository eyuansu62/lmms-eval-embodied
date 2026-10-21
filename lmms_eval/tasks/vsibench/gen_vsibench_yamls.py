#!/usr/bin/env python3
"""生成 VSI-Bench 各 P0 切片的 task yaml。

用法：
    python gen_vsibench_yamls.py
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# 切片名 -> (中文说明, 条目编号, 条件描述, 题数)
SLICES = {
    "rel_direction": (
        "相对位置", "#8",
        'question_type IN {object_rel_direction_easy, _medium, _hard}', 968,
    ),
    "rel_distance": (
        "相对距离", "#9",
        'question_type IN {object_rel_distance, object_abs_distance}', 1544,
    ),
    "size_estimation": (
        "2D/3D 理解与转换", "#44",
        'question_type IN {object_size_estimation, room_size_estimation}', 1241,
    ),
}

TEMPLATE = '''# VSI-Bench · {cn}（P0 {entry}）
#
# 筛选切片：{cond}
# 题数：{n}
#
# 论文：https://arxiv.org/abs/2412.14171
# 数据：https://huggingface.co/datasets/nyu-visionx/VSI-Bench
#
# ⚠️ 这是**按 能力体系（CAPABILITIES.md） 条目筛选的切片**，不是官方子集。
#    筛选逻辑见 build_vsibench_slices.py。
#
# 媒体：视频不在 parquet 里，需下载 arkitscenes.zip / scannet.zip /
#       scannetpp.zip 并解压到 ~/.cache/huggingface/vsibench/
#       使其形成 ~/.cache/huggingface/vsibench/<dataset>/<scene_name>.mp4
#
# 注：切片字段与原 test.jsonl 一致，直接复用本目录的 utils.py 判分。
#     doc_to_messages 也一并继承（部分模型后端走 messages 路径）。

dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/vsibench/data/vsibench_{name}.jsonl
  cache_dir: vsibench
  video: true
task: "vsibench_{name}"
test_split: test
output_type: generate_until

doc_to_visual: !function utils.vsibench_doc_to_visual
doc_to_text: !function utils.vsibench_doc_to_text
doc_to_messages: !function utils.vsibench_doc_to_messages
doc_to_target: "ground_truth"

generation_kwargs:
  max_new_tokens: 16
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false

process_results: !function utils.vsibench_process_results
metric_list:
  - metric: vsibench_overall
    aggregation: !function utils.vsibench_aggregate_overall
    higher_is_better: true
  - metric: object_abs_distance_mra
    aggregation: !function utils.vsibench_aggregate_object_abs_distance_mra
    higher_is_better: true
  - metric: object_rel_distance_accuracy
    aggregation: !function utils.vsibench_aggregate_object_rel_distance_accuracy
    higher_is_better: true
  - metric: object_size_estimation_mra
    aggregation: !function utils.vsibench_aggregate_object_size_estimation_mra
    higher_is_better: true

metadata:
  version: 0.1
  note: 'P0 {entry} {cn}；{n} 题；判据 {cond}'
'''


def main() -> None:
    for name, (cn, entry, cond, n) in SLICES.items():
        content = TEMPLATE.format(cn=cn, entry=entry, cond=cond, n=n, name=name)
        out = HERE / f"vsibench_{name}.yaml"
        out.write_text(content, encoding="utf-8")
        print(f"  {out.name}   {n:>5d} 题")

    print(f"\n生成 {len(SLICES)} 个 yaml")


if __name__ == "__main__":
    main()
