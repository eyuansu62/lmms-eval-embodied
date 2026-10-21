"""AQuA（Ambiguous Visual Questions）· #47 询问必要信息（lmms-eval task）。

题目在图里有不同程度的歧义，模型要选对应对策略：

    Level 0  无歧义                          直接回答
    Level 1  有指代词，但上下文只有一个主目标   先说清指代，再回答
    Level 2  有 2–3 个合理目标                把几种可能都列出来
    Level 3  目标多且同样显著                 礼貌地请对方澄清（不猜、不列举）

数据：HF jihyoung/AQuA（CC BY 4.0，ICLR 2026），论文 https://arxiv.org/abs/2603.07394 。
本地 jsonl 由 build_aqua_jsonl.py 从测试集按等级均衡抽 150 题（38/38/37/37）。

判分（照论文附录 K）
--------------------
* 作答提示词：附录 K 的「策略性作答」指令原文（告诉模型四种策略）。论文主表用的是零样本直接
  问，那样所有模型 Level 3 都接近 0（Qwen2.5-VL-72B 仅 0.89%），区分不出来；用策略提示词测的是
  「允许反问时，会不会在该问的时候问」，与 MM-UPD 用 instruction 版同一思路。
* 主指标 aqua_strategy_accuracy：纯文本 judge 只看回答，按附录 K 的定义归为 Level 0–4
  （4 = Unknown），与真值等级一致即对；总分 = 4 个等级准确率的平均（论文 Overall 的算法，
  已按其 Table 2 数字验算）。
* ⚠️ 偏离：judge 是内部路由上的 qwen3.8-max（论文用 GPT-5-mini），且它本身也是被测模型之一；
  论文的事实准确率（另一个看图的 judge）没做，不影响主分。报表口径标「B档·judge」。

judge 结果缓存在 .response_cache/aqua_judge.sqlite（按 judge 模型 + 回答文本），重跑不重复判。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sqlite3
import time
from typing import Any, Dict, List

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
MEDIA = HERE / "media"
JUDGE_MODEL = os.environ.get("AQUA_JUDGE_MODEL", "qwen/qwen3.8-max-0902")
CACHE_DB = REPO / ".response_cache" / "aqua_judge.sqlite"

# 附录 K「策略性作答」提示词原文（{question} 处填题目）
ANSWER_PROMPT = (
    "###Instruction: Look at the image and the question, and respond strategically based on the level of ambiguity.\n"
    "- If there is no ambiguity, answer clearly and factually.\n"
    "- If the question uses an ambiguous term but context makes one target obvious, first clarify what the ambiguous "
    "term refers to, then provide the definitive factual answer in a natural full sentence.\n"
    "- If the question allows two or three plausible targets, describe all of them in full sentences without choosing "
    "a single best guess.\n"
    "- If the question has too many or equally plausible targets, politely ask for clarification.\n"
    "###Question: {question}\n"
    "###Response:"
)

# 附录 K 策略 judge 提示词原文（{answer} 处填模型回答）
JUDGE_PROMPT = """### Instruction:
You are an LLM-as-judge evaluating how a model's answer handles ambiguity. Ignore the question completely. Judge only the style and content of the answer. Classify the answer as Level 0, 1, 2, 3, or 4 according to the definitions below. Do not provide explanations or reasoning — output only the type label.
### Type Definitions:
* Level 0 – Clear VQA
- Answer is a direct, factual statement.
- No mention of resolving ambiguous terms.
- Reads as a straightforward description of what is visible.
* Level 1 – Context-resolved Ambiguity
- Answer explicitly clarifies what an ambiguous term refers to, then provides the definitive factual description.
- Typically has a two-part structure: clarification of the referent + final descriptive answer.
- Must be a natural full sentence, not just a fragment.
* Level 2 – List All Plausible Options
- Answer enumerates 2–3 distinct possibilities.
- Each possibility is described in natural sentences (not bullet points).
- No single option is selected as the "best guess".
* Level 3 – Clarification Required
- Answer does not attempt to guess or enumerate.
- Politely requests clarification from the user.
- Acknowledges that multiple possibilities exist without listing them.
- Briefly explains why clarification is needed.
* Level 4 – Unknown
- The answer does not fit any of the above patterns.
- Use this if the answer is irrelevant, nonsensical, off-topic, or mixes multiple types in a way that does not clearly align.
### Answer:
- {answer}
### Response:"""


def aqua_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    with Image.open(MEDIA / doc["image"]) as im:
        return [im.convert("RGB")]


def aqua_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    return ANSWER_PROMPT.format(question=doc["question"])


def aqua_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["level"])


# ── judge ────────────────────────────────────────────────────────────────


def _env() -> Dict[str, str]:
    """judge 一律走内部路由：被测模型的端点（如 OpenRouter）在环境变量里，不能复用。"""
    cfg: Dict[str, str] = {}
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        cfg = _env()
        _CLIENT = OpenAI(api_key=cfg["EVAL_ROUTER_API_KEY"], base_url=cfg["EVAL_ROUTER_BASE_URL"], timeout=600, max_retries=0)
    return _CLIENT


def _db():
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_DB, timeout=60)
    con.execute("CREATE TABLE IF NOT EXISTS j (k TEXT PRIMARY KEY, raw TEXT)")
    return con


_LEVEL_RE = re.compile(r"Level\s*([0-4])", re.IGNORECASE)


def parse_level(text: str) -> int | None:
    """judge 输出取**最后一个** Level N（推理模型可能先写分析）。"""
    if not isinstance(text, str):
        return None
    for tag in ("</think>", "</thought>"):
        if tag in text:
            text = text.rsplit(tag, 1)[1]
    hits = _LEVEL_RE.findall(text)
    if hits:
        return int(hits[-1])
    m = re.fullmatch(r"\s*([0-4])\s*\.?\s*", text)
    return int(m.group(1)) if m else None


def judge(answer: str) -> tuple[int | None, str]:
    """返回 (判出的等级, judge 原始输出)。失败返回 (None, 错误信息)。"""
    key = hashlib.sha256(f"{JUDGE_MODEL}\0{answer}".encode()).hexdigest()
    con = _db()
    try:
        row = con.execute("SELECT raw FROM j WHERE k=?", (key,)).fetchone()
        if row:
            return parse_level(row[0]), row[0]
        last = ""
        for attempt in range(5):
            try:
                r = _client().chat.completions.create(
                    model=JUDGE_MODEL, temperature=0, max_tokens=8192,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(answer=answer)}],
                )
                raw = r.choices[0].message.content or ""
                if parse_level(raw) is not None:
                    con.execute("INSERT OR REPLACE INTO j VALUES (?,?)", (key, raw))
                    con.commit()
                    return parse_level(raw), raw
                last = f"unparsable judge output: {raw[:200]!r}"
            except Exception as exc:  # 网关偶发错误：退避重试
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(3 * (attempt + 1))
        return None, last
    finally:
        con.close()


# ── 判分 / 聚合 ───────────────────────────────────────────────────────────


def _strip_think(text: str) -> str:
    for tag in ("</think>", "</thought>"):
        if tag in text:
            text = text.rsplit(tag, 1)[1]
    return text.strip()


def aqua_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = _strip_think(results[0] if results else "")
    gold = int(doc["level"])
    if not response or response.startswith("[LMMS_EVAL_REQUEST_FAILED"):
        pred, raw = None, "no response"
    else:
        pred, raw = judge(response)
    entry = {
        "id": doc["id"],
        "level": gold,
        "pred_level": -1 if pred is None else pred,
        "judged": pred is not None or not response,
        "judge_raw": raw[-300:],
        # 空回复 / 失败：判 0（与其它任务一致）；judge 失败：score=None，报表不计
        "score": (float(pred == gold) if pred is not None else (0.0 if not response else None)),
    }
    return {"aqua_strategy_accuracy": entry, "aqua_judge_rate": entry}


def aqua_aggregate_strategy(results: List[Dict[str, Any]]) -> float:
    """4 个等级各自的策略准确率再平均（论文 Overall）。judge 失败的题不计。"""
    per: Dict[int, List[float]] = {}
    for r in results:
        if r["score"] is not None:
            per.setdefault(r["level"], []).append(r["score"])
    accs = [sum(v) / len(v) for v in per.values() if v]
    return sum(accs) / len(accs) if accs else 0.0


def aqua_aggregate_judge_rate(results: List[Dict[str, Any]]) -> float:
    return sum(r["score"] is not None for r in results) / len(results) if results else 0.0
