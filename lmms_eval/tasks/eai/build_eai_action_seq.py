#!/usr/bin/env python3
"""EAI · VirtualHome 动作序列 → 本地 jsonl（#31 约束满足检查）。

    pip install --no-deps eai_eval-1.0.5-py3-none-any.whl      # 判分时也要用到官方包
    python lmms_eval/tasks/eai/build_eai_action_seq.py          # 会自行以 PYTHONHASHSEED=0 重启

⚠️ 不能用包里自带的 prompts/helm_prompts.json：那份是旧模板，要求动作参数只写物体名
（{'FIND': ['sink']}），而官方判分（check_name_id_format / check_no_hallucination_in_arg）
要求「名字, ID」成对 —— 照旧 prompt 作答会全部判成解析失败（2026-10-20 冒烟 4/4）。
这里调官方 generate_prompts.py 的同一套代码，用现行模板 prompts/one_shot.py 现场生成。

prompt 里有 set 拼接，顺序依赖字符串哈希，所以同 build_eai_virtualhome.py 一样钉 PYTHONHASHSEED=0。

剔除：官方 evaluate_results 跳过「标准程序执行完都过不了终态判定」的题（program_num -= 1），
这里用同一判据（utils_as.passes_gold）在建数据时就去掉。判分见 utils_as.py。产物 Apache-2.0，进 git。
"""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.ignore_environment:
    env = dict(os.environ, PYTHONHASHSEED="0")
    os.execve(sys.executable, [sys.executable, "-s", __file__, *sys.argv[1:]], env)

import json  # noqa: E402
import logging  # noqa: E402
import pathlib  # noqa: E402

import virtualhome_eval.evaluation.action_sequencing.prompts.one_shot as one_shot  # noqa: E402

from lmms_eval.tasks.eai import utils_as  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    logging.disable(logging.CRITICAL)  # 官方模拟器每一步都打 INFO
    o = utils_as._official()
    kept, dropped = [], 0
    for task_name, scripts in o["task_dicts"].items():
        for file_id in scripts:
            if not utils_as.passes_gold(file_id):
                dropped += 1
                continue
            node, edge, _ = utils_as.gold_goals(file_id)
            raw_actions = scripts[file_id]["vh_goal"]["actions"]  # 生成 prompt 时官方用原始列表
            planner = utils_as._planner(file_id)[0]
            _, _, node_str, edge_str, action_str, _ = planner.get_symbolic_goal_nl(node, edge, action_goals=raw_actions)
            object_in_scene, cur_change, _ = planner.get_nl_goal_string()
            prompt = (one_shot.prompt
                      .replace("<object_in_scene>", object_in_scene)
                      .replace("<cur_change>", cur_change)
                      .replace("<node_goals>", node_str)
                      .replace("<edge_goals>", edge_str)
                      .replace("<action_goals>", action_str))
            kept.append({"identifier": file_id, "prompt": prompt})
    out = HERE / "data" / "eai_action_seq.jsonl"
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"保留 {len(kept)}，剔除 {dropped}（标准程序过不了终态判定）→ {out}")


if __name__ == "__main__":
    main()
