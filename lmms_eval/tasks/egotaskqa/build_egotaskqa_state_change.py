#!/usr/bin/env python3
"""EgoTaskQA #27 单体/多体状态理解切片：descriptive 里的对象状态 / 状态变化是非题。

能力体系（CAPABILITIES.md）#27 的主测是「EgoTaskQA Descriptive 中对象状态/状态变化题（筛选）」，
ERIQ Action Understanding 只是补充。2026-10-20 之前 #27 只挂 ERIQ，本切片补上主测。

判据：type 含 descriptive AND semantic ∈ {state, change} AND structural == verify
      （全量 8279 题：change 7111 / state 1168；答案 yes/no 各半）
采样：state / change 各 100 题（seed 42），共 200 题，95% 半宽约 ±7 分。
      change 题按自然比例会占 86%，且几乎都是「Did the attribute of X changed because
      of the action Y?」一个模板，等量采样是为了让「对象状态」那一半不被淹没。
去重：排除已在 event_order / counterfactual / explanatory 采样切片里的 question_id，
      避免同一道题在两项能力里各算一次。
只取 verify：开放题的 gold 是封闭动作词表，生成式判分下几乎全错（见 utils_filter.py）。

用法：
    python build_egotaskqa_state_change.py
"""

from __future__ import annotations

import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
PER_SEMANTIC = 100
SEED = 42
SAMPLED_SLICES = ("event_order", "counterfactual", "explanatory")


def main() -> None:
    rows = json.loads((DATA / "tagged_qas.json").read_text(encoding="utf-8"))
    used: set[int] = set()
    for name in SAMPLED_SLICES:
        for line in (DATA / f"egotaskqa_{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                used.add(int(json.loads(line)["question_id"]))

    def types(r: dict) -> list:
        t = r.get("type")
        return t if isinstance(t, list) else [t]

    pool = [r for r in rows
            if "descriptive" in types(r)
            and r.get("semantic") in ("state", "change")
            and str(r.get("structural", "")).strip().lower() == "verify"
            and int(r["question_id"]) not in used]

    full = DATA / "_full" / "egotaskqa_state_change.jsonl"
    full.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool), encoding="utf-8")

    rng = random.Random(SEED)
    picked: list[dict] = []
    for sem in ("state", "change"):
        cand = [r for r in pool if r["semantic"] == sem]
        picked += rng.sample(cand, PER_SEMANTIC)
        print(f"  {sem:6s} {len(cand):5d} 题 → 取 {PER_SEMANTIC}")
    picked.sort(key=lambda r: int(r["question_id"]))

    out = DATA / "egotaskqa_state_change.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8")
    (DATA / "egotaskqa_state_change_indices.json").write_text(
        json.dumps([int(r["question_id"]) for r in picked]), encoding="utf-8")
    yes = sum(r["answer"] == "yes" for r in picked)
    print(f"全量池 {len(pool)} 题 → {full.relative_to(HERE)}")
    print(f"采样 {len(picked)} 题（yes {yes} / no {len(picked) - yes}）→ {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
