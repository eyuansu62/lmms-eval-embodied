"""Embodied Agent Interface（EAI）· 目标解释（lmms-eval task）。

挂 能力体系 能力行 #16 目标理解（「理解任务的最终目标状态是什么」），两个子集等权平均：

    eai_behavior_goal      100 题  给任务名 + 一句自然语言目标 + 场景物体 / 状态 / 关系词表，
                                   输出符号化目标 {"node goals": [...], "edge goals": [...]}
    eai_virtualhome_goal   338 题  同上，另有 action goals；判分细节见文件后半「VirtualHome」一节

数据：官方 eai-eval 1.0.5 包内资源（见 build_eai_jsonl.py），Apache-2.0
论文：https://arxiv.org/abs/2410.07166  代码：github.com/embodied-agent-interface/embodied-agent-interface

⚠️ 纯文本，不看图
-----------------
EAI 原协议就是符号 / 文本接口（能力体系：「与 VLM 图像输入不同，不能直接混算标准成绩」），
这里按原协议原样跑，能力表口径标「B档·纯文本」。测的是从文字理解目标状态，不涉及视觉。

判分：逐行移植 behavior_eval/evaluation/goal_interpretation/scripts/evaluate_results.py
------------------------------------------------------------------------------------
* 抽取（parse_json）：整段转小写、' → "、toggledon → toggled_on，取**第一个**不含嵌套的
  `{...}` 做 json.loads；失败则当作什么都没预测（该题 F1 = 0）。
* 每个目标若有多种满足方式，取命中条数最多的那一种（并列取第一个）；
  TP = 命中的真值条件，FN = 未命中的，FP = 预测里不在命中集合中的。
* **官方分数是 micro F1**：全数据集的 TP/FP/FN 合并后再算 P/R/F1（eai_f1）。
  报表 summary_*.csv 是逐题取平均的口径，用的是每题 F1 的均值（eai_f1_macro），
  两者不相等，引用时注意。

偏离官方的只有两处，都只影响「官方会直接崩溃」的输入：
  1. 先去掉 </think> 之前的推理再抽取（官方没处理推理模型）；
  2. 解析出的 JSON 里，值不是列表的键忽略（官方会把字符串逐字符当条件，或抛异常）。
另落一个宽松口径 eai_f1_lenient：取**最后一个** `{...}`，对付推理过程里先出现花括号的回复；
与官方口径的差值就是「格式问题」造成的损失。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

NODE_STATES = ["Cooked", "Dusty", "Frozen", "Open", "Sliced", "Soaked", "Stained", "Toggled_On"]
EDGE_STATES = ["Inside", "NextTo", "OnFloor", "OnTop", "Touching", "Under"]

_OBJ_RE = re.compile(r"{[^{}]*}", re.DOTALL)


def eai_doc_to_visual(doc: Dict[str, Any]) -> list:
    return []


def eai_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{doc['prompt']}{kwargs.get('post_prompt', '')}"


def eai_doc_to_target(doc: Dict[str, Any]) -> str:
    return json.dumps(doc["goal_conditions"], ensure_ascii=False)


# ── 抽取 ──────────────────────────────────────────────────────────────────


def _strip_think(text: str) -> str:
    for tag in ("</think>", "</thought>"):
        if tag in text:
            text = text.rsplit(tag, 1)[1]
    return text


def parse_goals(text: str, last: bool = False) -> Tuple[Dict[str, Any], bool]:
    """官方 parse_json（last=False）；last=True 为宽松口径，取最后一个 {...}。

    返回 (解析结果, 是否解析成功)。失败时返回空预测，与官方一致。
    """
    empty = {"node goals": [], "edge goals": []}
    if not isinstance(text, str):
        return empty, False
    raw = _strip_think(text).lower().replace("'", '"').replace("toggledon", "toggled_on")
    hits = _OBJ_RE.findall(raw)
    if not hits:
        return empty, False
    try:
        parsed = json.loads(hits[-1] if last else hits[0])
    except Exception:
        return empty, False
    if not isinstance(parsed, dict):
        return empty, False
    return parsed, True


def _flatten(goal_data: Dict[str, Any]) -> list:
    return [c for v in goal_data.values() if isinstance(v, list) for c in v]


# ── 官方判分（逐行移植） ──────────────────────────────────────────────────


def _is_state(state: str, cond) -> bool:
    if cond[0] == "not":
        return cond[1][0] == state.lower()
    return cond[0] == state.lower()


def _is_kind(cond, states: List[str], arity: int) -> bool:
    """官方 is_node_condition / is_edge_condition：先按状态名判，异常时退回按长度判。"""
    try:
        try:
            return any(_is_state(s, cond) for s in states)
        except Exception:
            if cond[0] == "not":
                return len(cond[1]) == arity
            return len(cond) == arity
    except Exception:  # 官方此处会抛异常；这里按「不属于该类」处理
        return False


def evaluate_goals(pred: Dict[str, Any], gt_goals: Dict[str, list]):
    predicted = _flatten(pred)
    sat, unsat = [], []
    for options in gt_goals.values():
        if len(options) == 1:
            best = options[0]
        else:
            nums = [len([c for c in opt if c in predicted]) for opt in options]
            best = options[nums.index(max(nums))]
        for c in best:
            (sat if c in predicted else unsat).append(c)
    fp = [c for c in predicted if c not in sat]
    return sat, unsat, fp, predicted


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def _counts(pred: Dict[str, Any], gt: Dict[str, list]) -> Dict[str, int]:
    sat, unsat, fp, _ = evaluate_goals(pred, gt)
    out = {"tp": len(sat), "fn": len(unsat), "fp": len(fp)}
    for kind, states, arity in (("state", NODE_STATES, 2), ("relation", EDGE_STATES, 3)):
        out[f"{kind}_tp"] = sum(_is_kind(c, states, arity) for c in sat)
        out[f"{kind}_fn"] = sum(_is_kind(c, states, arity) for c in unsat)
        out[f"{kind}_fp"] = sum(_is_kind(c, states, arity) for c in fp)
    return out


def eai_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = results[0] if results else ""
    gt = doc["goal_conditions"]
    pred, ok = parse_goals(response)
    pred_l, ok_l = parse_goals(response, last=True)
    c = _counts(pred, gt)
    cl = _counts(pred_l, gt)
    entry = {
        "id": doc.get("id", ""),
        "parsed": ok,
        **c,
        "f1": _prf(c["tp"], c["fp"], c["fn"])[2],
        "lenient_tp": cl["tp"], "lenient_fp": cl["fp"], "lenient_fn": cl["fn"],
    }
    return {k: entry for k in (
        "eai_f1", "eai_precision", "eai_recall", "eai_state_f1", "eai_relation_f1",
        "eai_f1_macro", "eai_f1_lenient", "eai_parse_rate",
    )}


# ── 聚合 ──────────────────────────────────────────────────────────────────


def _micro(results: List[Dict[str, Any]], prefix: str = "") -> Tuple[float, float, float]:
    tp = sum(r[f"{prefix}tp"] for r in results)
    fp = sum(r[f"{prefix}fp"] for r in results)
    fn = sum(r[f"{prefix}fn"] for r in results)
    return _prf(tp, fp, fn)


def eai_aggregate_f1(results):
    """官方主分：micro F1（全数据集合并 TP/FP/FN）。"""
    return _micro(results)[2]


def eai_aggregate_precision(results):
    return _micro(results)[0]


def eai_aggregate_recall(results):
    return _micro(results)[1]


def eai_aggregate_state_f1(results):
    """官方 state_goal：只看一元状态条件（cooked / open / …）。"""
    return _micro(results, "state_")[2]


def eai_aggregate_relation_f1(results):
    """官方 relation_goal：只看二元关系条件（inside / ontop / …）。"""
    return _micro(results, "relation_")[2]


def eai_aggregate_f1_macro(results):
    """每题 F1 的均值 —— 报表 summary_*.csv 用的就是这个口径，不是官方 micro。"""
    return sum(r["f1"] for r in results) / len(results) if results else 0.0


def eai_aggregate_f1_lenient(results):
    return _micro(results, "lenient_")[2]


def eai_aggregate_parse_rate(results):
    return sum(r["parsed"] for r in results) / len(results) if results else 0.0


# ══ VirtualHome（eai_virtualhome_goal，338 题）══════════════════════════════
#
# 逐行移植 virtualhome_eval/evaluation/goal_interpretation/scripts/evaluate_results.py。
# 与 BEHAVIOR 的差别（都是官方实现如此）：
#   * 解析极严：去 <|eot_id|>、去开头 ```json、去换行、' → " 后整段 json.loads，
#     **不从回复里抽 JSON**；
#   * 解析失败的题被 `continue` 掉，**连 FN 都不计** —— 格式错误不扣分。官方主分
#     eai_f1 照此复现；eai_f1_strict 把这些题的真值全部记为 FN，不让格式错误占便宜；
#   * 预测用物体**名字**，按 名字→场景 ID 映射后与真值比；另有 action goals 一类。
#
# 偏离官方之处：
#   1. 先去掉 </think> 之前的推理（同 BEHAVIOR）；
#   2. 同名多 ID（46/338 题）：官方用 set 转 dict 随机取一个 ID，判分不可复现；
#      这里任一 ID 能对上真值即算命中（比官方的随机结果宽，但确定）；
#   3. 预测条目不是 dict 时记格式错误（官方会对字符串做子串判断，结果无意义）。
# 报表口径：summary_*.csv 的逐题分数用**宽松抽取**（eai_f1_lenient 那套解析；仍抽不到才记 0）。
#   官方解析要求整段回复就是 JSON，而多数模型会在 JSON 前后加解释：2026-10-17 实测
#   glm-5.3-flash 官方可解析率仅 17%、mimo 两个 44–49%、doubao 88%。按 strict（失败记全 FN）
#   glm 只剩 17.8，但宽松抽取后 44.5，与官方「跳过失败题」的 44.2 几乎一样 —— 答案其实写对了，
#   strict 罚的是格式而非目标理解。所以报表取宽松口径，三个口径都在结果 json 里。
#
# ⚠️ 满分到不了 100：预测只能写物体**名字**，同名物体（不同 ID）各有同一目标时（如两盏灯都要
# 打开）名字写出来是同一条，去重后只能命中一个。15 题共 42/800 条真值因此不可达，
# 把真值原样当预测喂进去，F1 = 0.973（recall 上限 0.9475）。官方实现同样如此。


def _vh_parse_official(text: str):
    if not isinstance(text, str):
        return None
    out = _strip_think(text).replace("<|eot_id|>", "")
    if out.startswith("```json"):
        out = out[7:].strip("```")
    out = out.strip().replace("\n", "").replace("'", '"')
    try:
        parsed = json.loads(out)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _vh_parse_lenient(text: str):
    """从回复里找**最后一个**能完整解码、且带目标键的 JSON 对象（允许嵌套）。"""
    if not isinstance(text, str):
        return None
    body = _strip_think(text).replace("<|eot_id|>", "")
    dec = json.JSONDecoder()
    found = None
    for variant in (body, body.replace("'", '"')):
        i = variant.find("{")
        while i != -1:
            try:
                obj, end = dec.raw_decode(variant, i)
            except Exception:
                i = variant.find("{", i + 1)
                continue
            if isinstance(obj, dict) and any(k in obj for k in ("node goals", "edge goals", "action goals")):
                found = obj
            i = variant.find("{", end)
        if found is not None:
            return found
    return None


def _dedup(xs) -> list:
    def h(x):
        if isinstance(x, dict):
            return frozenset((k, h(v)) for k, v in x.items())
        if isinstance(x, list):
            return tuple(h(e) for e in x)
        return x
    try:
        return list({h(x): x for x in xs}.values())
    except TypeError:
        return list(xs)


def _vh_counts(pred: Dict[str, Any] | None, doc: Dict[str, Any]) -> Dict[str, int]:
    """单题 TP/FP/FN（node / edge / action 三类）。pred=None 表示解析失败。"""
    gn, ge, ga = doc["gold_node_goals"], doc["gold_edge_goals"], doc["gold_action_goals"]
    c = {f"{k}_{m}": 0 for k in ("state", "relation", "action") for m in ("tp", "fp", "fn")}
    c.update(format_wrong=0, hallucination=0)
    if pred is None:
        return c
    name_ids: Dict[str, set] = {}
    for name, i in doc["name_id_pairs"]:
        name_ids.setdefault(name, set()).add(i)

    def as_list(v):
        return v if isinstance(v, list) else []

    tp = fp = 0
    for g in _dedup(as_list(pred.get("node goals", []))):
        if isinstance(g, (dict, list, str)) and len(g) == 0:
            continue
        if not isinstance(g, dict) or "name" not in g or "state" not in g:
            c["format_wrong"] += 1
            continue
        if g["name"] not in name_ids:
            c["hallucination"] += 1
            continue
        if any({"id": i, "class_name": g["name"], "state": g["state"]} in gn for i in name_ids[g["name"]]):
            tp += 1
        else:
            fp += 1
    c["state_tp"], c["state_fp"], c["state_fn"] = tp, fp, len(gn) - tp

    tp = fp = 0
    for g in _dedup(as_list(pred.get("edge goals", []))):
        if isinstance(g, (dict, list, str)) and len(g) == 0:
            continue
        if not isinstance(g, dict) or not all(k in g for k in ("from_name", "to_name", "relation")):
            c["format_wrong"] += 1
            continue
        if g["from_name"] not in name_ids or g["to_name"] not in name_ids:
            c["hallucination"] += 1
            continue
        if any({"from_id": a, "to_id": b, "relation_type": g["relation"]} in ge
               for a in name_ids[g["from_name"]] for b in name_ids[g["to_name"]]):
            tp += 1
        else:
            fp += 1
    c["relation_tp"], c["relation_fp"], c["relation_fn"] = tp, fp, len(ge) - tp

    # action：逐行照官方循环（含其「命中后从未真正移除」的写法，同一金标动作可被重复命中）
    tp = fp = 0
    for g in _dedup(as_list(pred.get("action goals", []))):
        if isinstance(g, (dict, list, str)) and len(g) == 0:
            continue
        if not isinstance(g, dict) or "action" not in g or not isinstance(g["action"], str):
            c["format_wrong"] += 1
            continue
        a = g["action"].upper()
        if any(a in [x.upper() for x in gold.split("|")] for gold in ga):
            tp += 1
        else:
            fp += 1
    c["action_tp"], c["action_fp"], c["action_fn"] = tp, fp, len(ga) - tp
    return c


def _vh_total(c: Dict[str, int], prefix: str = "") -> Dict[str, int]:
    return {f"{prefix}{m}": sum(c[f"{k}_{m}"] for k in ("state", "relation", "action")) for m in ("tp", "fp", "fn")}


def _vh_gold_n(doc) -> int:
    return len(doc["gold_node_goals"]) + len(doc["gold_edge_goals"]) + len(doc["gold_action_goals"])


def eai_vh_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = results[0] if results else ""
    pred = _vh_parse_official(response)
    c = _vh_counts(pred, doc)
    tot = _vh_total(c)
    # strict：解析失败时真值全部记 FN（官方此时什么都不计）
    strict = dict(tot) if pred is not None else {"tp": 0, "fp": 0, "fn": _vh_gold_n(doc)}
    pl = _vh_parse_lenient(response)
    lc = _vh_total(_vh_counts(pl, doc)) if pl is not None else {"tp": 0, "fp": 0, "fn": _vh_gold_n(doc)}
    entry = {
        "id": doc.get("id", ""),
        "parsed": pred is not None,
        **tot, **c,
        "strict_tp": strict["tp"], "strict_fp": strict["fp"], "strict_fn": strict["fn"],
        "lenient_tp": lc["tp"], "lenient_fp": lc["fp"], "lenient_fn": lc["fn"],
        # 报表用的逐题分数：宽松抽取（连宽松抽取都失败才记 0），理由见上方「报表口径」
        "f1": _prf(lc["tp"], lc["fp"], lc["fn"])[2],
    }
    return {k: entry for k in (
        "eai_f1", "eai_f1_strict", "eai_precision", "eai_recall", "eai_state_f1",
        "eai_relation_f1", "eai_action_f1", "eai_f1_macro", "eai_f1_lenient", "eai_parse_rate",
    )}


def eai_aggregate_f1_strict(results):
    """解析失败的题真值全记 FN 后的 micro F1（官方不这么算，见上）。"""
    return _micro(results, "strict_")[2]


def eai_aggregate_action_f1(results):
    return _micro(results, "action_")[2]


def eai_vh_doc_to_target(doc: Dict[str, Any]) -> str:
    return json.dumps({"node": doc["gold_node_goals"], "edge": doc["gold_edge_goals"],
                       "action": doc["gold_action_goals"]}, ensure_ascii=False)
