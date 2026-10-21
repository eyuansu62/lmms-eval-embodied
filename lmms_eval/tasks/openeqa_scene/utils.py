"""OpenEQA · 场景语义两类（lmms-eval task，挂 #6）。

    openeqa_world_knowledge       75 题  房间类型 / 装修风格 / 环境属性（「客厅什么吊顶」「车库里什么车」）
    openeqa_functional_reasoning  75 题  区域用途 / 用什么做什么（「怎么凉快下来」「去哪里洗手」）

数据：thomas-yanxin/open-eqa-mirror（OpenEQA，CVPR 2024，论文 2023-11）。这里只取全量 1636 题里的
这两类（213 / 217 题），各抽 75 题。每题 32 帧探索视频的帧序列，本地取 16 帧、长边 960 作为图片输入。

⚠️ 口径
-------
* **帧序列**：输入是均匀抽的 16 帧，不是官方完整视频。官方 episodic-memory 设定要求看完整个视频；
  抽帧后信息少一些，而且帧数是离线的，不是「看视频」。
* **judge**：主指标 openeqa_accuracy 由内部路由的 qwen3.8-max 判断「模型答案与标准答案语义是否等价」
  （官方用 GPT-4），结果缓存在 `.response_cache/openeqa_judge.sqlite`。
* **数据老**：2023 年发布，题目与答案在互联网上已久，可能已被训练数据覆盖；答案是场景绑定的短句，
  靠背题不太可能直接命中，但仍需在报表里注明。口径标「B档·帧序列」。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sqlite3
import time
from typing import Any, Dict, List

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
MEDIA = HERE / "media"
JUDGE_MODEL = "qwen/qwen3.8-max-0902"
CACHE_DB = REPO / ".response_cache" / "openeqa_judge.sqlite"

ANSWER_PROMPT = (
    "You are an embodied agent exploring a home. Answer the question based on the images, "
    "which are frames from your exploration, in a short phrase or one sentence.\n\n"
    "Question: {question}\n"
    "Answer:"
)

JUDGE_PROMPT = (
    "You are grading an answer to an embodied question-answering task.\n"
    "The prediction is correct if it conveys the same content as the reference answer up to paraphrasing — "
    "synonyms, articles, singular/plural, and word order do not matter. It is wrong if it names a different "
    "object, a different room/place, contradicts the reference, or only gives a vague restatement without the "
    "referenced entity.\n\n"
    "Question: {question}\n"
    "Reference answer: {gold}\n"
    "Predicted answer: {pred}\n\n"
    "Reply with exactly one word: CORRECT or INCORRECT."
)


def openeqa_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    out = []
    for rel in doc["frames"]:
        with Image.open(MEDIA / rel) as im:
            out.append(im.convert("RGB"))
    return out


def openeqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    return ANSWER_PROMPT.format(question=doc["question"])


def openeqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["answer"])


# ── judge ────────────────────────────────────────────────────────────────

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        cfg = {}
        env_file = REPO / ".env"
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip().strip('"').strip("'")
        _CLIENT = OpenAI(api_key=cfg["EVAL_ROUTER_API_KEY"], base_url=cfg["EVAL_ROUTER_BASE_URL"],
                         timeout=600, max_retries=0)
    return _CLIENT


def _db():
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_DB, timeout=60)
    con.execute("CREATE TABLE IF NOT EXISTS j (k TEXT PRIMARY KEY, raw TEXT)")
    return con


def _strip_think(text: str) -> str:
    for tag in ("</think>", "</thought>"):
        if tag in text:
            text = text.rsplit(tag, 1)[1]
    return text.strip()


def parse_verdict(text: str) -> bool | None:
    """取**最后一个** CORRECT / INCORRECT；注意 INCORRECT 里也含 CORRECT，必须先匹配整词。"""
    if not isinstance(text, str):
        return None
    body = _strip_think(text)
    hits = re.findall(r"\b(INCORRECT|CORRECT)\b", body.upper())
    if not hits:
        return None
    return hits[-1] == "CORRECT"


def judge(question: str, gold: str, pred: str) -> tuple[bool | None, str]:
    key = hashlib.sha256(f"{JUDGE_MODEL}\0{question}\0{gold}\0{pred}".encode()).hexdigest()
    con = _db()
    try:
        row = con.execute("SELECT raw FROM j WHERE k=?", (key,)).fetchone()
        if row:
            return parse_verdict(row[0]), row[0]
        last = ""
        for attempt in range(5):
            try:
                r = _client().chat.completions.create(
                    model=JUDGE_MODEL, temperature=0, max_tokens=4096,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)}],
                )
                raw = r.choices[0].message.content or ""
                if parse_verdict(raw) is not None:
                    con.execute("INSERT OR REPLACE INTO j VALUES (?,?)", (key, raw))
                    con.commit()
                    return parse_verdict(raw), raw
                last = f"unparsable: {raw[:160]!r}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:160]}"
            time.sleep(3 * (attempt + 1))
        return None, last
    finally:
        con.close()


# ── 判分 / 聚合 ───────────────────────────────────────────────────────────


def openeqa_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = _strip_think(results[0] if results else "")
    if not response or response.startswith("[LMMS_EVAL_REQUEST_FAILED"):
        ok, raw = (0.0 if not response else None), "no response"
    else:
        verdict, raw = judge(doc["question"], doc["answer"], response)
        ok = None if verdict is None else float(verdict)
    entry = {
        "id": doc["id"],
        "category": doc["category"],
        "question": doc["question"],
        "gold": doc["answer"],
        "pred": response[:2000],
        "judge_raw": raw[-300:],
        "score": ok,
    }
    return {"openeqa_accuracy": entry, "openeqa_judge_rate": entry}


def openeqa_aggregate_accuracy(results):
    xs = [r["score"] for r in results if r["score"] is not None]
    return sum(xs) / len(xs) if xs else 0.0


def openeqa_aggregate_judge_rate(results):
    return sum(r["score"] is not None for r in results) / len(results) if results else 0.0
