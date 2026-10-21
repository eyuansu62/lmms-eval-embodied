#!/usr/bin/env python3
"""生成 EgoTaskQA 三个切片的 task yaml。

用法：
    python gen_egotaskqa_yamls.py
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# 切片名 -> (中文说明, 条目编号, 判据, 题数)
SLICES = {
    "event_order": (
        "事件顺序理解", "#13",
        'step >= 2 AND 题干含时序关键词(before/after/first/then/...)', 406,
    ),
    "counterfactual": (
        "反事实推理", "#22",
        'type 含 "counterfactual"', 159,
    ),
    "explanatory": (
        "归因理解", "#24",
        'type 含 "explanatory"', 235,
    ),
}

TEMPLATE = '''# EgoTaskQA · {cn}（P0 {entry}）
#
# 筛选切片：{cond}
# 题数：{n}（采样后；全量见 data/_full/）
#
# 数据：https://huggingface.co/datasets/eyuansu71/EgoTaskQA
# 论文：https://arxiv.org/abs/2201.05994
#
# ⚠️ 这是**按 能力体系（CAPABILITIES.md） 条目筛选的切片**，不是官方子集。
#    筛选逻辑见 build_egotaskqa_slices.py。
#
# ⚠️ **未使用内置 utils.py**：内置那份是给 nv-njb/EgoTaskQA-MCQ 的 MCQ 判分写的，
#    而官方数据（本切片所用）答案是**自由文本、无选项**；
#    且 MCQ 版的答案键与官方不一致（官方答案 53.7% 不在 MCQ 选项里），不能 join。
#    故改用 utils_qa.py 做自由文本判分。
#
# 视频：需下载官方的 data/qa_videos.zip，解压到
#       ~/.cache/lmms_eval/egotaskqa/videos/ 使之形成
#       ~/.cache/lmms_eval/egotaskqa/videos/<interval>.mp4
#       （可用 EGOTASKQA_VIDEO_DIR 覆盖）
#
# 判分：归一化后精确匹配（小写/去标点/去冠词）。严格判分，
#       语义等价但措辞不同会判错。详见 utils_qa.py。
#
# ⚠️ 只跑"是/否"题：process_docs 过滤掉 structural=="query" 的开放题。
#    开放题的 gold 是 1015 条封闭动作用语词表，而模型输出自然语言，
#    精确匹配必然判错（452 道开放题里仅 8.2% 落在词表内），四个模型
#    在开放题上得分全部塌到 2–6%、无区分度。是/否题不受影响且有真实信号。
#    设 EGOTASKQA_INCLUDE_OPEN=1 可恢复全量。详见 utils_filter.py。

dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/egotaskqa/data/egotaskqa_{name}.jsonl
task: "egotaskqa_{name}"
test_split: test
process_docs: !function utils_filter.process_docs_verify_only
output_type: generate_until

doc_to_visual: !function utils_qa.doc_to_visual
doc_to_text: !function utils_qa.doc_to_text
doc_to_target: !function utils_qa.doc_to_target

generation_kwargs:
  max_new_tokens: 32
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false

process_results: !function utils_qa.process_results
metric_list:
  - metric: egotaskqa_accuracy
    aggregation: !function utils_qa.aggregate_results
    higher_is_better: true

lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\\nAnswer with a short phrase only, without extra explanation."

metadata:
  version: 0.1
  note: 'P0 {entry} {cn}；{n} 题；判据 {cond}；自由文本判分'
'''


def main() -> None:
    for name, (cn, entry, cond, n) in SLICES.items():
        content = TEMPLATE.format(cn=cn, entry=entry, cond=cond, n=n, name=name)
        out = HERE / f"egotaskqa_{name}.yaml"
        out.write_text(content, encoding="utf-8")
        print(f"  {out.name}   {n:>4d} 题")

    print(f"\n生成 {len(SLICES)} 个 yaml")


if __name__ == "__main__":
    main()
