"""Theory of Space（ToS，ICLR 2026）主动探索评测（实验性，未进入能力表）。

环境、出题、判分全部调用官方代码（vendor/Theory-of-Space，固定提交，见 build_tos.py），这里只把它接到
lmms-eval 的多轮接口上，按官方 scripts/SpatialGym 的协议组织对话：

  active  —— 探索阶段：system + 初始说明（带 2 张示意图），模型每轮输出动作（JumpTo/Rotate/Observe/Query/Term），
             环境回一条观测（图在文字里内联）；Term 或 20 步用完结束。评测阶段：在完整探索对话之后提问，
             9 类题各 3 道，每道一次独立请求（官方 build_evaluation_from_combo），27 道在同一轮并发发出。
  passive —— 官方 scout 代理先替模型探索好（同一份观测日志），模型只答同样的 27 道题。

两种设定题目完全相同（由场景和 seed 决定），active 与 passive 的差就是「自己探索」的代价。
图片同官方：探索阶段缩到 384×384，评测阶段用原图（768×768）。认知地图探针与 false-belief 未接。
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from typing import Any, Dict, List

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
MEDIA = HERE / "media"
VENDOR = HERE / "vendor"
EXP_IMAGE_SIZE = (384, 384)
MAX_EXP_STEPS = 20
EVAL_TASK_COUNTS = {"dir": 3, "pov": 3, "bwd_pov_text": 3, "fwd_fov": 3, "bwd_nav_text": 3, "e2a": 3, "rot": 3, "fwd_loc": 3, "bwd_loc_text": 3}
# 论文分组：route-level（自我中心、动作相关）与 survey-level（全局地图）
ROUTE = {"pov", "bwd_pov_text", "fwd_fov", "bwd_nav_text"}
SURVEY = {"dir", "e2a", "rot", "fwd_loc", "bwd_loc_text"}

_OUT_ROOT = None
_EPISODES: Dict[tuple, "_Episode"] = {}


def _tos():
    """官方代码按需导入（vendor 目录追加在 sys.path 末尾，不覆盖主环境已装的包）。"""
    for p in (VENDOR / "Theory-of-Space", VENDOR / "deps"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    from vagen.env.spatial import SpatialGym, SpatialGymConfig
    from vagen.env.spatial.message_list_builder import build_evaluation_from_combo
    from vagen.env.spatial.Base.tos_base.evaluation.tasks import evaluate_from_dict
    from vagen.env.spatial.Base.tos_base.utils.utils import parse_llm_response

    return SpatialGym, SpatialGymConfig, build_evaluation_from_combo, evaluate_from_dict, parse_llm_response


def _image(path: str, size=None):
    im = Image.open(path).convert("RGB")
    return im.resize(size, Image.LANCZOS) if size else im


def _to_chat(messages: List[Dict], size=None) -> List[Dict]:
    """官方消息（文字里的 <image> 占位 + images 路径列表）→ lmms-eval 聊天格式，图片按占位位置内联。"""
    out = []
    for m in messages:
        images = list(m.get("images") or [])
        parts = m["content"].split("<image>") if images else [m["content"]]
        content = []
        for i, text in enumerate(parts):
            if text.strip():
                content.append({"type": "text", "text": text})
            if i < len(parts) - 1 and i < len(images):
                rel = str(pathlib.Path(images[i]).resolve().relative_to(MEDIA))
                content.append({"type": "image", "url": _image(images[i], size), "key": f"{rel}@{size}"})
        out.append({"role": m["role"], "content": content})
    return out


class _Episode:
    def __init__(self, mode: str, seed: int):
        global _OUT_ROOT
        if _OUT_ROOT is None:
            _OUT_ROOT = tempfile.mkdtemp(prefix="lmms_tos_")
        SpatialGym, SpatialGymConfig, self._build_eval, self._evaluate, self._parse = _tos()
        cfg = SpatialGymConfig(
            exp_type=mode, max_exp_steps=1 if mode == "passive" else MAX_EXP_STEPS, render_mode="vision",
            data_dir=str(MEDIA), proxy_agent="scout", prompt_config={"enable_think": True},
            kwargs={"output_dir": f"{_OUT_ROOT}/{mode}/{seed}", "model_config": {"model_name": "lmms_eval"}, "all_override": True},
        )
        self.env = SpatialGym(cfg)
        self.env.reset(seed=seed)
        self.mode = mode
        self.exploring = mode == "active"
        self.questions, self.meta, self.answers = [], [], []
        if not self.exploring:
            self._start_eval()

    def _start_eval(self):
        self.exploring = False
        self.questions, self.meta = self._build_eval(self.env.history_manager.output_dir, EVAL_TASK_COUNTS, eval_override=True)

    def current_messages(self) -> List:
        """探索阶段返回一段对话；评测阶段返回 27 段（每题一段，同一轮并发发出）。"""
        if self.exploring:
            return _to_chat(self.env.history_manager.messages, EXP_IMAGE_SIZE)
        return [_to_chat(q) for q in self.questions]

    def feed(self, outputs: List[str]) -> bool:
        """喂入模型上一轮的输出；返回 True 表示探索与问答都已结束。"""
        if self.exploring:
            _, _, done, _ = self.env.step(outputs[-1])
            if done:
                self._start_eval()
            return False
        self.answers = outputs[-len(self.questions):]
        return True

    def result(self) -> Dict[str, Any]:
        qs = []
        for meta, raw in zip(self.meta, self.answers):
            _, answer, _ = self._parse(raw)
            score, _ = self._evaluate(meta["evaluation_data"], answer)
            qs.append({"task_type": meta["task_type"], "question_id": meta["question_id"], "answer": answer, "score": float(score)})
        exp = self.env.get_exp_summary()
        return {"mode": self.mode, "questions": qs,
                "exploration": {k: exp.get(k) for k in ("node_coverage", "edge_coverage", "action_cost", "n_exploration_steps") if k in exp},
                "turns": self.env.current_turn_number}


def _doc_to_text(mode: str):
    def doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
        key = (mode, doc["seed"])
        if round_idx is None:  # 只在缓存键里出现；真正的首轮对话由 initial_messages 给出
            return f"tos_{mode}:{doc['id']}"
        ep = _EPISODES[key]
        if ep.feed(previous_output or [""]):
            # 转义 "<"：答案里若有 "</think>"，评测器默认的思考剥离会从那里截断整段 JSON（10-23 27b 有 3 个场景因此记 0）
            payload = json.dumps(ep.result(), ensure_ascii=False).replace("<", "\\u003c")
            del _EPISODES[key]
            return None, None, True, [payload], {"seed": doc["seed"]}
        return None, ep.current_messages(), False, None, {"seed": doc["seed"]}

    def initial_messages(doc):
        ep = _EPISODES[(mode, doc["seed"])] = _Episode(mode, doc["seed"])
        return ep.current_messages()

    doc_to_text.initial_messages = initial_messages
    return doc_to_text


active_doc_to_text = _doc_to_text("active")
passive_doc_to_text = _doc_to_text("passive")


def doc_to_visual(doc):
    return []


def doc_to_target(doc):
    return ""


# ---------- 判分 ----------

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    try:
        payload = json.loads(results[0] if results else "")
    except (TypeError, json.JSONDecodeError):
        payload = None
    if not (isinstance(payload, dict) and "questions" in payload):  # 步数上限内没走完：27 题全记 0 分
        payload = {"questions": [{"task_type": t, "score": 0.0} for t, n in EVAL_TASK_COUNTS.items() for _ in range(n)],
                   "exploration": {}, "failed": True}
    entry = {"id": doc["id"], **payload}
    return {k: entry for k in ("tos_accuracy", "tos_route", "tos_survey", "tos_action_cost", "tos_node_coverage", "tos_complete")}


def _scores(rs, types=None):
    s = [q["score"] for r in rs for q in r["questions"] if types is None or q["task_type"] in types]
    return sum(s) / len(s) if s else 0.0


def agg_accuracy(rs):
    return _scores(rs)


def agg_route(rs):
    return _scores(rs, ROUTE)


def agg_survey(rs):
    return _scores(rs, SURVEY)


def _exp_mean(rs, key):
    v = [r["exploration"].get(key) for r in rs if r["exploration"].get(key) is not None]
    return sum(v) / len(v) if v else 0.0


def agg_action_cost(rs):
    return _exp_mean(rs, "action_cost")


def agg_node_coverage(rs):
    return _exp_mean(rs, "node_coverage")


def agg_complete(rs):
    return sum(not r.get("failed") for r in rs) / len(rs) if rs else 0.0
