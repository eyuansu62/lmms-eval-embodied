#!/usr/bin/env python3
"""EAI · VirtualHome 目标解释 → 本地 jsonl（scene_1，338 题）。

    pip download eai-eval==1.0.5 --no-deps -d /tmp/eai_whl
    python -m zipfile -e /tmp/eai_whl/eai_eval-1.0.5-py3-none-any.whl /tmp/eai_whl/x
    python lmms_eval/tasks/eai/build_eai_virtualhome.py /tmp/eai_whl/x     # 会自行以 PYTHONHASHSEED=0 重启

与 BEHAVIOR 不同，VirtualHome 没有现成的 prompt 文件：要用官方 virtualhome_eval 的场景图
代码（纯 Python，依赖 numpy/scipy，不需要模拟器）现场生成。本脚本调用的就是官方
generate_prompts.py / evaluate_results.py 里的同一套函数，一次性算出 prompt、真值与
名称→ID 对，写成 jsonl；之后评测与判分不再依赖官方代码。产物 Apache-2.0，进 git。

⚠️ 三个实测坑
1. **prompt 本身不确定**：官方把 set 直接 str() 拼进 prompt（物体可能状态、关系-物体对），
   顺序随字符串哈希变，342 题全都受影响（只是顺序，字符多重集逐题相同）。
   所以必须以 PYTHONHASHSEED=0 运行（脚本会自行重启确保生效），否则每次生成的数据都不一样。
2. **4 题真值为空**（339_1 / 627_1 / 84_1 / 93_1）：没有任何可判的目标，剔除后正好 338 题，
   与 HF Inevitablevalor/EmbodiedAgentInterface 的 virtualhome split 逐个对得上。
3. **同名多 ID**：46 题里同一个物体名对应多个场景 ID。官方用 set 转 dict 取其一，结果随哈希
   顺序随机；这里把全部 (名, ID) 对存下来，判分口径见 utils.py。
"""

import json
import os
import pathlib
import sys

# 官方 prompt 里的 set（含 State 枚举，其哈希就是名字字符串的哈希）顺序依赖 PYTHONHASHSEED。
# 只看环境变量不够：`python -I` / `-E` 会让解释器**忽略** PYTHON* 变量 —— 2026-10-17 就这样
# 撞上过，检查通过但两次生成的 prompt 全部不同。所以看 sys.flags，没生效就带上种子重启自己。
if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.ignore_environment:
    env = dict(os.environ, PYTHONHASHSEED="0")
    os.execve(sys.executable, [sys.executable, "-s", __file__, *sys.argv[1:]], env)

pkg_root = sys.argv[1]
HERE = pathlib.Path(__file__).resolve().parent
out_path = HERE / "data" / "eai_virtualhome_goal.jsonl"
sys.path.insert(0, pkg_root)

import virtualhome_eval.simulation.evolving_graph.utils as utils  # noqa: E402
from virtualhome_eval.simulation.evolving_graph.eval_utils import (  # noqa: E402
    construct_planner, remove_duplicate_dicts,
)
import virtualhome_eval.evaluation.goal_interpretation.prompts.one_shot as one_shot  # noqa: E402

base = os.path.join(pkg_root, "virtualhome_eval")
resource_root = os.path.join(base, "resources", "virtualhome")
data_dir = os.path.join(base, "dataset", "programs_processed_precond_nograb_morepreconds")
scenegraph_id = 1

all_rel = json.load(open(os.path.join(resource_root, "relation_types.json")))
action_space = json.load(open(os.path.join(resource_root, "action_space.json")))
task_dict = json.load(open(os.path.join(resource_root, "task_state_LTL_formula_accurate.json")))[f"scene_{scenegraph_id}"]
rel_obj = json.load(open(os.path.join(resource_root, "relation_object_pairs.json")))
for k, v in rel_obj.items():
    rel_obj[k] = set(v)

properties_data = utils.load_properties_data()
object_states = utils.load_object_states()
object_placing = utils.load_object_placing()
name_equivalence = utils.load_name_equivalence()

records = []
for task, scripts in task_dict.items():
    for script_id, goal_dict in scripts.items():
        planner, _, _, task_name, task_description = construct_planner(
            name_equivalence, properties_data, object_placing,
            scenegraph_id=scenegraph_id, script_id=script_id, dataset_root=data_dir,
        )
        object_in_scene, goal_str, _ = planner.get_goal_describe_nl(task_name, task_description, object_states)
        prompt = (one_shot.prompt
                  .replace("<object_in_scene>", object_in_scene)
                  .replace("<goal_str>", goal_str)
                  .replace("<relation_types>", str(all_rel))
                  .replace("<action_space>", str(action_space))
                  .replace("<rel_obj_pairs>", str(rel_obj)))

        # 与官方 evaluate_results.py 相同的真值整理
        goals = goal_dict["vh_goal"]
        node, edge = [], []
        for g in goals["goal"]:
            if "id" in g and "class_name" in g and "state" in g:
                node.append(g)
            elif "from_id" in g and "to_id" in g and "relation_type" in g:
                edge.append(g)
            else:
                raise ValueError(g)
        # 名称→ID：官方用 set 转 dict，同名多 ID 时结果依赖哈希顺序；这里全部保留
        pairs = sorted(planner.get_relevant_nodes(script_id=script_id))
        records.append({
            "id": script_id,
            "task": task,
            "task_name": task_name,
            "prompt": prompt,
            "gold_node_goals": remove_duplicate_dicts(node),
            "gold_edge_goals": remove_duplicate_dicts(edge),
            "gold_action_goals": sorted(set(goals["actions"])),
            "name_id_pairs": [list(p) for p in pairs],
        })

empty = [r["id"] for r in records
         if not (r["gold_node_goals"] or r["gold_edge_goals"] or r["gold_action_goals"])]
records = [r for r in records if r["id"] not in empty]
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"eai_virtualhome_goal: {len(records)} 题（剔除空真值 {len(empty)} 题：{empty}）-> {out_path}")
