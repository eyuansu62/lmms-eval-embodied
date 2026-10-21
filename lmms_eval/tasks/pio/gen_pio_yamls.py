#!/usr/bin/env python3
"""Generate YAML task configs for Point-It-Out (PIO) slices."""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

SLICES = [
    ("pio_affordance",
     "P0 #2 客体可供性感知（s2/afd+contact）",
     124),
    ("pio_attribute_reference",
     "P0 #19 属性指代（s1/restriction+object_part）",
     95),
    ("pio_safety",
     "P0 #36 安全监控（s2/safety）",
     12),
]

TEMPLATE = """\
# Point-It-Out (PIO) · {desc}
#
# 输出类型: generate_until (坐标定位)
# 判分: 点落入目标 polygon 计为命中
# 数据: GitHub xavihart/PIO, s1s2.json
# 图片: 按需下载至 ~/.cache/lmms_eval/pio/images_s1s2/
# 题数: {n}

dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/pio/data/{name}.jsonl
task: "{name}"
test_split: test
output_type: generate_until

doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target

generation_kwargs:
  max_new_tokens: 32
  temperature: 0
  do_sample: false

process_results: !function utils.process_results
metric_list:
  - metric: pio_accuracy
    aggregation: !function utils.aggregate_accuracy
    higher_is_better: true
  - metric: pio_parse_ok
    aggregation: !function utils.aggregate_parse_rate
    higher_is_better: true

lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\\nProvide the coordinates of the target point as (x, y) pixel values. Output only the coordinates."

metadata:
  version: 0.1
  note: "B-tier PIO · {desc}"
"""


def main() -> None:
    for name, desc, n in SLICES:
        content = TEMPLATE.format(name=name, desc=desc, n=n)
        out = HERE / f"{name}.yaml"
        out.write_text(content)
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
