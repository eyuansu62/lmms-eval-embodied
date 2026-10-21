"""EAI · VirtualHome 动作序列（Action Sequencing）判分 —— #31 约束满足检查。

逐行移植 virtualhome_eval/evaluation/action_sequencing/scripts/evaluate_results.py 的单题逻辑：
解析 → 幻觉检查（动作名 / 物体名+ID 是否在场景里）→ 参数个数检查 → 在 VirtualHome 演化图
模拟器里逐步执行（纯 Python，不需要 Unity）→ 终态判定。

依赖官方包 eai-eval 1.0.5（``pip install --no-deps eai_eval-1.0.5-py3-none-any.whl``，
2026-10-20 已装进 lmms-eval 环境）。判分时现场调模拟器，不能像目标理解那样预先算好真值。

为什么挂 #31 而不是 #29
----------------------
能力体系 把 EAI Action Sequencing 列在 #29 指令顺序理解和 #31 约束满足检查下。实测：
* 342 题里 193 题没有动作要求、136 题只有 1 个、仅 13 题有 2 个（才谈得上先后）；官方提示词
  把每条动作目标打印两遍，按行数数会误以为有 149 题带顺序约束。官方判分还对动作目标做了
  ``list(set(...))``，那 13 题里 10 题的顺序会被打乱。所以它基本不测「遵守给定先后」。
* 模拟器逐步检查前置条件，不满足即中止并归类（缺步骤 / 可供性错误 / 顺序错误）——
  「整个计划没有一步违反前置条件」正是 #31 的定义。

报表主分 eai_as_executable（执行成功率）：解析失败 / 幻觉 / 参数错误 / 某步不可执行都记 0。
另落 eai_as_success（终态全部达成，官方 task_success_rate）与各类错误计数。

偏离官方之处：
  1. 先去掉 </think> 之前的推理再解析（同目标理解）；
  2. 输出与标准程序逐条相同时，官方走捷径分支只记「可执行」、**不记成功**（task_success
     少算）；这里照常对终态判定。执行成功率、错误分类与官方逐题一致（2026-10-20 用 342 条
     构造回复对拍：执行率 56.7 / 幻觉 8.20 / 缺步骤 32.79 / 顺序 0.33 / 可供性 1.97，全同）；
  3. 动作目标按原顺序去重。官方 ``list(set(...))`` 让顺序随 PYTHONHASHSEED 变，而顺序检查
     check_order_with_or_score 对顺序敏感：有 2 个动作目标的题（如 1003_2 的 TYPE→SWITCHOFF）
     换个种子成功与否就翻转。只影响 eai_as_success，不影响主分 eai_as_executable。
"""

from __future__ import annotations

import copy
import functools
import json
import os
from typing import Any, Dict, List

from lmms_eval.tasks.eai.utils import _strip_think

ERROR_TYPES = {0: "wrong_temporal_order", 1: "missing_step", 2: "affordance_error", 4: "additional_step"}


@functools.lru_cache(maxsize=1)
def _official():
    """延迟导入：只有跑这个任务才需要官方包；资源只加载一次。"""
    import virtualhome_eval
    import virtualhome_eval.simulation.evolving_graph.utils as vh_utils
    from virtualhome_eval.simulation.evolving_graph import eval_utils as eu
    from virtualhome_eval.simulation.evolving_graph.checker import TemporalOrderChecker

    root = os.path.dirname(virtualhome_eval.__file__)
    res = os.path.join(root, "resources", "virtualhome")
    return {
        "eu": eu,
        "checker": TemporalOrderChecker,
        "data_dir": os.path.join(root, "dataset", "programs_processed_precond_nograb_morepreconds"),
        "task_dicts": json.load(open(os.path.join(res, "task_state_LTL_formula_accurate.json")))["scene_1"],
        "id2task": json.load(open(os.path.join(res, "id2task.json"))),
        "properties": vh_utils.load_properties_data(),
        "placing": vh_utils.load_object_placing(),
        "name_eq": vh_utils.load_name_equivalence(),
    }


def gold_goals(file_id: str):
    """与官方相同的真值整理：(node, edge, action) 目标。"""
    o = _official()
    goals = o["task_dicts"][o["id2task"][file_id]][file_id]["vh_goal"]
    node, edge = [], []
    for g in goals["goal"]:
        if "id" in g and "class_name" in g and "state" in g:
            node.append(g)
        elif "from_id" in g and "to_id" in g and "relation_type" in g:
            edge.append(g)
        else:
            raise ValueError(f"{file_id}: 目标格式不对 {g}")
    return (o["eu"].remove_duplicate_dicts(node), o["eu"].remove_duplicate_dicts(edge),
            list(dict.fromkeys(goals["actions"])))  # 官方是 list(set(...))，见文件头偏离第 3 条


def _planner(file_id: str):
    o = _official()
    return o["eu"].construct_planner(o["name_eq"], o["properties"], o["placing"],
                                     scenegraph_id=1, script_id=file_id, dataset_root=o["data_dir"])


def passes_gold(file_id: str) -> bool:
    """官方跳过「标准答案自己都过不了终态判定」的题，这里用同一判据在 build 时筛掉。"""
    o = _official()
    node, edge, _ = gold_goals(file_id)
    planner = _planner(file_id)[0]
    return bool(o["eu"].scene_evaluate_wID(planner.final_state_dict, node, edge, planner.acting_char_id)[3])


def score(file_id: str, response: str) -> Dict[str, Any]:
    o, eu = _official(), _official()["eu"]
    node, edge, action = gold_goals(file_id)
    planner, _, gd_actions, _, _ = _planner(file_id)
    relevant_name_to_id = planner.get_symbolic_goal_nl(node, edge, action_goals=action)[5]
    out = {"parse_ok": False, "hallucination": False, "param_error": False, "executable": False,
           "success": False, "error_type": None, "n_goals": len(node) + len(edge) + len(action),
           "goals_matched": 0}

    text = _strip_think(response or "")
    if text.startswith("```json"):
        text = text[7:]
    text = text.strip().replace("\n", "").replace("'", '"')
    try:
        actions = eu.load_json_preserving_order(text)
    except Exception:
        actions = None
    if not actions or not eu.check_name_id_format(actions)[0]:
        return out
    out["parse_ok"] = True

    if not (eu.check_no_hallucination_in_action(actions)
            and eu.check_no_hallucination_in_arg(actions, relevant_name_to_id)):
        out["hallucination"] = True
        return out
    if not eu.check_action_grammar(actions)[0]:
        out["param_error"] = True
        return out
    actions = eu.json_to_action(actions, relevant_name_to_id=relevant_name_to_id)

    history: List[str] = []
    executable = True
    out["gold_identical"] = actions == gd_actions
    if actions != gd_actions:
        planner.reset()
        states = [copy.deepcopy(planner.env_state.to_dict())]
        for act in actions:
            snap = copy.deepcopy(states)
            ok, info = planner.my_execute_primitive_action_eval(act)
            if ok:
                history.append(act)
                states.append(copy.deepcopy(planner.env_state.to_dict()))
                continue
            code = o["checker"](info, snap).run_checker().get_error_type()
            if code != 4:  # 4 = 多余步骤：官方不中止、不算不可执行
                out["error_type"] = ERROR_TYPES.get(code, "unknown")
                executable = False
                break
        final = planner.env_state.to_dict()
    else:
        history = list(gd_actions)
        final = planner.final_state_dict
    out["executable"] = executable
    # 与官方一致：中途不可执行也对「执行到中止那一步」的状态判终态
    n_m, e_m, a_m, ok, *_ = eu.scene_evaluate_wID(final, node, edge, planner.acting_char_id,
                                                 action_seq=history, action_goals=action)
    out["goals_matched"] = n_m + e_m + max(a_m, 0)
    out["success"] = bool(ok)
    return out


# ── lmms-eval 接口 ────────────────────────────────────────────────────────────

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["prompt"]


def doc_to_target(doc):
    return doc["identifier"]


def process_results(doc, results):
    r = score(doc["identifier"], results[0] if results else "")
    return {"eai_as_executable": r, "eai_as_success": r, "eai_as_parse_rate": r}


def agg_executable(rs):
    return sum(r["executable"] for r in rs) / len(rs) if rs else 0.0


def agg_success(rs):
    return sum(r["success"] for r in rs) / len(rs) if rs else 0.0


def agg_parse_rate(rs):
    return sum(r["parse_ok"] for r in rs) / len(rs) if rs else 0.0
