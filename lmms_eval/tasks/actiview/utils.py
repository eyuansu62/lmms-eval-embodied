"""ActiView 主动感知多轮评测（实验性，未进入能力表）。

提示词照抄论文附录 J（官方评测代码未公开）；选项改为字母并要求只答字母，便于与其他任务同一抽取口径。
  full   —— J.1，全图单轮，作为对照。
  zoom   —— J.2，两轮：看全图选要放大的切片（1–4），再看全图 + 所选切片作答。
  shift  —— J.3，多轮：从一个切片出发，每轮问还要不要看别的切片；要就补上它点名的切片
            （点名无效时补一个相邻切片，即论文的做法），答 NO 或四片看全后作答。三档初始切片取自 shifting_level.json。

终止轮返回 JSON：answer_raw、views（最终看过/选过的切片编号）、rounds、invalid（选片回复无法解析的轮数）。
图片长边限 1600：原图最大 6327×6066，5 张图 PNG 编码会超网关 10 MB 上限；切片本已放大到全图尺寸，
限幅后仍比全图清晰一倍，放大的意义保留。
"""

from __future__ import annotations

import json
import pathlib
import re
from functools import lru_cache
from typing import Any, Dict, List

from PIL import Image

MEDIA = pathlib.Path(__file__).resolve().parent / "media"
MAX_SIDE = 1600
NAMES = ["upper-left", "lower-left", "upper-right", "lower-right"]  # 编号 1–4
ADJACENT = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
SPLITS = "1 is the upper-left part, 2 is the lower-left part, 3 is the upper-right part, and 4 is the lower-right part."
LETTER_ONLY = "\nAnswer with the option letter only."


@lru_cache(maxsize=64)
def _img(name: str) -> Image.Image:
    im = Image.open(MEDIA / name).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def _full(doc):
    return _img(doc["full"])


def _view(doc, k: int):
    return _img(doc["views"][k - 1])


def _options(doc) -> str:
    return " ".join(f"({l}) {c}" for l, c in zip(doc["letters"], doc["candidates"]))


def _answer_text(text: str) -> str:
    return (text or "").split("</think>")[-1].strip()


def _done(info: dict, answer_raw: str):
    payload = json.dumps({"answer_raw": answer_raw, "views": info["views"], "rounds": info["rounds"], "invalid": info["invalid"]},
                         ensure_ascii=False)
    return None, None, True, [payload], info


# ---------- full ----------

def full_doc_to_visual(doc):
    return [_full(doc)]


def full_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ("Carefully analysis this image, and answer the question from the given options. "
            f"Question: {doc['question']} Options: {_options(doc)}. Answer:" + LETTER_ONLY)


# ---------- zoom ----------

def _parse_parts(text: str) -> List[int]:
    return sorted({int(x) for x in re.findall(r"\b([1-4])\b", _answer_text(text))})


def zoom_doc_to_visual(doc):
    return [_full(doc)]


def zoom_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
    if round_idx is None:
        return ("This is the full image, which is split in to 4 equal parts, numbered from 1 to 4, where " + SPLITS + "\n===\n"
                "Response with the number of part (at least one part, at most 4 parts), that must be used to answer the question. "
                f"The question is: {doc['question']}\n===\n"
                "Do not directly answer the given question. Response with the selected number of parts, "
                "split by ',' if there are multiple selections. Your Response:")
    last = previous_output[-1] if previous_output else ""
    if round_idx == 1:
        parts = _parse_parts(last)
        info = {"views": parts, "rounds": 1, "invalid": 0 if parts else 1}
        zoomed = " ".join(f"Image {i} is the zoomed {NAMES[k - 1]} part." for i, k in enumerate(parts, 1))
        text = (f"Image 0 is the full image. {zoomed} These are your selected part from the full image to be zoomed for details "
                "for answering the question. Please answer question according to the given images from the the given options. "
                f"Question: {doc['question']} Options: {_options(doc)}. Answer:" + LETTER_ONLY)
        return [_full(doc)] + [_view(doc, k) for k in parts], text, False, None, info
    info = dict(previous_round_info, rounds=previous_round_info["rounds"] + 1)
    return _done(info, last)


# ---------- shift ----------

def _requested(text: str, unseen: List[int]) -> List[int]:
    t = _answer_text(text).lower().replace("_", "-")
    t = re.sub(r"\btop\b", "upper", re.sub(r"\bbottom\b", "lower", t))
    t = re.sub(r"(upper|lower)\s+(left|right)", r"\1-\2", t)
    return [k for k in unseen if NAMES[k - 1] in t]


def _says_no(text: str) -> bool:
    return re.match(r"^\W*no\b", _answer_text(text), re.I) is not None


def _check_prompt(doc, seen: List[int]) -> str:
    unseen = [k for k in range(1, 5) if k not in seen]
    opts = ", ".join(NAMES[k - 1] for k in unseen)
    if len(seen) == 1:
        head = f"image 0 is the {NAMES[seen[0] - 1]} part of the full image. Given image 0"
    else:
        desc = ", ".join(f"image {i} is the {NAMES[k - 1]} part" for i, k in enumerate(seen))
        head = f"{desc} of the full image. Given these images"
    return ("You will be presented with a partial image and a question concerning the full image. "
            f"{head}, please determine if you need more visual information to answer the question: {doc['question']}\n===\n"
            "Do not directly answer the question. If you can answer the question without more visual information, response with NO. "
            f"Otherwise, response with other image parts you need to see given these parts, you can choose from these views: {opts}. "
            "Your Response:")


def _answer_prompt(doc, seen: List[int]) -> str:
    desc = " ".join(f"Image {i} is the {NAMES[k - 1]} part." for i, k in enumerate(seen))
    return (f"These are parts of an image. {desc} Carefully analysis these images and pay attention to their original position. "
            f"Answer the question from the given options. Question: {doc['question']}. Options: {_options(doc)}. Answer:" + LETTER_ONLY)


def _shift_visual(level):
    return lambda doc: [_view(doc, doc["init"][level] + 1)]


def _shift_text(level):
    def doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
        start = doc["init"][level] + 1
        if round_idx is None:
            return _check_prompt(doc, [start])
        info = previous_round_info or {"views": [start], "rounds": 0, "invalid": 0, "answering": False}
        info = dict(info, views=list(info["views"]), rounds=info["rounds"] + 1)
        last = previous_output[-1] if previous_output else ""
        if info["answering"]:
            return _done(info, last)
        unseen = [k for k in range(1, 5) if k not in info["views"]]
        if not _says_no(last):
            add = _requested(last, unseen)
            if not add:
                info["invalid"] += 1
                add = next([k] for v in info["views"] for k in ADJACENT[v] if k in unseen)
            info["views"] += add
        seen = info["views"]
        visuals = [_view(doc, k) for k in seen]
        if _says_no(last) or len(seen) == 4:
            info["answering"] = True
            return visuals, _answer_prompt(doc, seen), False, None, info
        return visuals, _check_prompt(doc, seen), False, None, info

    return doc_to_text


shift_easy_doc_to_visual, shift_easy_doc_to_text = _shift_visual("easy"), _shift_text("easy")
shift_medium_doc_to_visual, shift_medium_doc_to_text = _shift_visual("medium"), _shift_text("medium")
shift_hard_doc_to_visual, shift_hard_doc_to_text = _shift_visual("hard"), _shift_text("hard")


# ---------- 判分 ----------

def doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["answer"]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def extract(text: str, letters: List[str], candidates: List[str]) -> str | None:
    t = _answer_text(text)
    valid = "".join(letters)
    m = re.match(r"^\W*([A-G])(?:\b|[.):])", t)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"(?:answer|option)\s*(?:is|:)?\s*\(?\**([A-G])\b", t, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    nt = _norm(t)
    hits = [i for i, c in enumerate(candidates) if _norm(c) and re.search(rf"\b{re.escape(_norm(c))}\b", nt)]
    if len(hits) == 1:
        return letters[hits[0]]
    ms = [x for x in re.findall(r"\b([A-G])\b", t) if x in valid]
    return ms[-1] if ms else None


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    raw = results[0] if results else ""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and "answer_raw" in payload:
        answer_raw, views, rounds, invalid = payload["answer_raw"], payload["views"], payload["rounds"], payload["invalid"]
    elif isinstance(payload, dict):  # 多轮未在步数上限内结束
        answer_raw, views, rounds, invalid = "", [], 0, 1
    else:  # 单轮（full）
        answer_raw, views, rounds, invalid = raw, [], 1, 0
    pred = extract(answer_raw, doc["letters"], doc["candidates"])
    gt = set(doc["gt_views"])
    entry = {"id": doc["id"], "pred": pred, "answer": doc["answer"], "is_correct": pred == doc["answer"], "parsed": pred is not None,
             "views": views, "rounds": rounds, "invalid": invalid,
             "clue_recall": len(gt & set(views)) / len(gt) if views else 0.0}
    return {k: entry for k in ("actiview_accuracy", "actiview_parse_rate", "actiview_num_views", "actiview_clue_recall", "actiview_invalid_rate")}


def _mean(rs, f):
    return sum(f(r) for r in rs) / len(rs) if rs else 0.0


def agg_accuracy(rs):
    return _mean(rs, lambda r: r["is_correct"])


def agg_parse_rate(rs):
    return _mean(rs, lambda r: r["parsed"])


def agg_num_views(rs):
    return _mean(rs, lambda r: len(r["views"]))


def agg_clue_recall(rs):
    return _mean(rs, lambda r: r["clue_recall"])


def agg_invalid_rate(rs):
    return _mean(rs, lambda r: r["invalid"] > 0)
