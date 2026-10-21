#!/usr/bin/env python3
"""Generate YAML task configs for RoboBench MCQ slices."""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

SLICES = [
    ("robobench_static_affordance",  "P0 #2 客体可供性感知（静态）"),
    ("robobench_dynamic_affordance", "P0 #2 客体可供性感知（动态）"),
    # 整组 static_attribute 已不对应任何能力（85% 是可变形性/易碎性/质量等功能属性），
    # 保留 task 只为读旧结果；#4 改用下面的 visual_attribute。
    ("robobench_static_attribute",   "static_attribute 全组（不对应 #4，仅供对照旧结果）"),
    ("robobench_visual_attribute",   "P0 #4 客体视觉属性（static_attribute ∩ material/transparency，31 题全量）"),
    ("robobench_spatial_relation",   "P0 #8 相对位置"),
    ("robobench_timestamp_analysis", "P0 #12 时间指代定位"),
    ("robobench_high_level_error",   "P0 #25 失败定位（规划层）"),
    ("robobench_low_level_error",    "P0 #25 失败定位（执行层）"),
]

TEMPLATE = """\
# RoboBench · {desc}
#
# 数据: LeoFan01/RoboBench (HF), 图片按需下载至 media/
# 论文: RoboBench (2025)  https://arxiv.org/html/2510.17801v2
# Prompt: 官方前缀（data/system_prompt.json），见 utils.doc_to_text
# 由 gen_robobench_yamls.py 生成，不要手改。

dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/robobench/data/{name}.jsonl
task: "{name}"
test_split: test
output_type: generate_until

doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target

generation_kwargs:
  max_new_tokens: 65536
  temperature: 0
  do_sample: false

process_results: !function utils.process_results
metric_list:
  - metric: robobench_accuracy
    aggregation: !function utils.aggregate_results
    higher_is_better: true

metadata:
  # 0.2：改用官方 prompt。0.1 的结果（只发裸题干）与之不可比。
  version: 0.2
  note: "B-tier RoboBench · {desc}"
"""


def main() -> None:
    for name, desc in SLICES:
        content = TEMPLATE.format(name=name, desc=desc)
        out = HERE / f"{name}.yaml"
        out.write_text(content)
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
