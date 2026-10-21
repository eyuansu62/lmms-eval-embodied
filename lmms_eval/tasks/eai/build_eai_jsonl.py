#!/usr/bin/env python3
"""Embodied Agent Interface（EAI）· BEHAVIOR 目标解释 → 本地 jsonl。

    pip download eai-eval==1.0.5 --no-deps -d /tmp/eai_whl     # 官方包，27 MB
    python -m zipfile -e /tmp/eai_whl/eai_eval-1.0.5-py3-none-any.whl /tmp/eai_whl/x
    python lmms_eval/tasks/eai/build_eai_jsonl.py --src /tmp/eai_whl/x

产物（Apache-2.0，进 git）：
    lmms_eval/tasks/eai/data/eai_behavior_goal.jsonl   100 题

数据源
------
官方仓库 embodied-agent-interface/embodied-agent-interface（PyPI 包 eai-eval 1.0.5）的
`behavior_eval/evaluation/goal_interpretation/resources/data/`：
    llm_prompts.json     demo id -> 官方已生成好的完整 prompt（含物体、状态与关系词表）
    all_conditions.json  demo id -> {"goal_conditions": {目标: [方案1, 方案2, …]}, …}
    all_objects.json     demo id -> 场景里合法的物体（判「物体幻觉」用）

HF 的 Inevitablevalor/EmbodiedAgentInterface 只有任务描述与 LTL 目标，**没有** prompt 和
逐条真值，判不了官方口径，所以不从那里取。
"""

from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REL = "behavior_eval/evaluation/goal_interpretation/resources/data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="解压后的 eai-eval 包根目录（含 behavior_eval/）")
    args = ap.parse_args()
    data = pathlib.Path(args.src) / REL

    prompts = json.loads((data / "llm_prompts.json").read_text())
    conds = json.loads((data / "all_conditions.json").read_text())
    objs = json.loads((data / "all_objects.json").read_text())

    out = HERE / "data" / "eai_behavior_goal.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        # 官方 evaluate_results 按 demo 名排序后逐个判；这里沿用 prompt 文件的 100 个 key
        for demo in sorted(prompts):
            goal = conds[demo]["goal_conditions"]
            assert isinstance(goal, dict) and goal, demo
            rec = {
                "id": demo,
                "task_name": demo.split("_0_")[0],
                "prompt": prompts[demo],
                "goal_conditions": goal,
                "legal_objects": sorted(objs[demo]),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"eai_behavior_goal: {n} 题 -> {out.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
