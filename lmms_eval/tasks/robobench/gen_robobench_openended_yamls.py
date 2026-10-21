#!/usr/bin/env python3
"""Generate YAML configs for RoboBench open-ended (planning) subtasks."""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_robobench_jsonl import (  # noqa: E402
    PLAN20_PER_SUBTASK, PLAN_PER_SUBTASK, PLAN_SUBTASKS, plan20_task_name, plan_task_name,
)

SLICES = [
    ("robobench_implicit_goal",
     "implicit_demand_goal，Q1/Q2/Q3 全量 255 题（#14 请用 robobench_implicit_goal_q1）"),
    ("robobench_implicit_goal_q1",
     "P0 #14 指令具化（implicit_demand_goal 的 Q1 全量 100 题）"),
    # 下面两个不对应能力表里的任何一项（#16 / #20 表里写明只是场景源），保留只为 3B 基线
    ("robobench_explicit_goal",
     "explicit_object_goal（不对应能力项，仅保留 3B 基线）"),
    ("robobench_number",
     "physical_attribute/number 全量 Q1（#28 请用 robobench_plan_number）"),
] + [
    (plan_task_name(sub),
     f"{sub}（Q1，每子任务抽 {PLAN_PER_SUBTASK} 题；#28 请用 {plan20_task_name(sub)}）")
    for sub in PLAN_SUBTASKS
] + [
    (plan20_task_name(sub),
     f"P0 #28 子任务分解 · {sub}（Q1，每子任务抽 {PLAN20_PER_SUBTASK} 题，是 {plan_task_name(sub)} 的子集）")
    for sub in PLAN_SUBTASKS
]

TEMPLATE = """\
# RoboBench · {desc}
#
# 输出类型: generate_until（开放式规划）
# Prompt: 官方 skill list + 本体类型 + 题干，见 utils.doc_to_text_openended
# 判分（与官方 evaluation/planning.py 一致）:
#   - Q3 yes/no: 评测时直接判
#   - Q1 多步规划: judge 看首帧 + GT 动作序列 + GT DAG，给 node_correctness 与
#                  task_completion（各 0-10），分数 = 两者之和 / 20
#   - Q2 下一步:   judge 抽取下一步 → 按技能/对象/参数三维比较，分数 = 三项之和 / 3
#   Q1/Q2 在评测时记 -1（待判），须事后跑 judge_robobench_openended.py
# 数据: LeoFan01/RoboBench (HF)；由 gen_robobench_openended_yamls.py 生成，不要手改。

dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/robobench/data/{name}.jsonl
task: "{name}"
test_split: test
output_type: generate_until

doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.doc_to_text_openended
doc_to_target: !function utils.doc_to_target

generation_kwargs:
  max_new_tokens: 65536
  temperature: 0
  do_sample: false

process_results: !function utils.process_results_openended
metric_list:
  - metric: rb_openended_score
    aggregation: !function utils.aggregate_openended
    higher_is_better: true

metadata:
  # 0.2：官方 prompt + 官方判分。0.1 只发裸题干、用自写的 F1/0-10 judge，不可比。
  version: 0.2
  note: "B-tier RoboBench · {desc} · judge needed post-hoc"
"""


def main() -> None:
    for name, desc in SLICES:
        out = HERE / f"{name}.yaml"
        out.write_text(TEMPLATE.format(name=name, desc=desc))
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
