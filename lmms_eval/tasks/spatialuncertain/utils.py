"""SpatialUncertain（#45 场景完整判断与处置）判分。抽取逻辑同 tvbench/utils_cap13.py。

配对口径（判断）不在这里算：occ_answerable 与 occ_abstain 两个切片按 pair_id 对齐，
两份都对才算对，见 dump_reports._spatialuncertain_paired_scores。
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

from PIL import Image

MEDIA = pathlib.Path(__file__).resolve().parent / "media"


def doc_to_visual(doc: Dict[str, Any]) -> list:
    return [Image.open(MEDIA / p).convert("RGB") for p in doc["images"]]


def doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    lines = [doc["question"]] + [f"{l}. {c}" for l, c in zip(doc["letters"], doc["candidates"])]
    lines.append("Answer with the option letter only.")
    return "\n".join(lines)


def doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["answer"]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def extract(text: str, letters: List[str], candidates: List[str]) -> str | None:
    t = (text or "").split("</think>")[-1].strip()
    valid = "".join(letters)
    m = re.match(r"^\W*([A-E])(?:\b|[.):])", t)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"(?:answer|option|view)\s*(?:is|:)?\s*\(?\**([A-E])\b", t, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    nt = _norm(t)
    hits = [i for i, c in enumerate(candidates) if _norm(c) and re.search(rf"\b{re.escape(_norm(c))}\b", nt)]
    if len(hits) == 1:
        return letters[hits[0]]
    ms = [x for x in re.findall(r"\b([A-E])\b", t) if x in valid]
    return ms[-1] if ms else None


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    pred = extract(results[0] if results else "", doc["letters"], doc["candidates"])
    entry = {"id": doc["id"], "pair_id": doc.get("pair_id"), "pred": pred, "answer": doc["answer"],
             "is_correct": pred == doc["answer"], "parsed": pred is not None}
    return {"spatialuncertain_accuracy": entry, "spatialuncertain_parse_rate": entry}


def agg_accuracy(rs):
    return sum(r["is_correct"] for r in rs) / len(rs) if rs else 0.0


def agg_parse_rate(rs):
    return sum(r["parsed"] for r in rs) / len(rs) if rs else 0.0
