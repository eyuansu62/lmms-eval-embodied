#!/usr/bin/env python3
"""把 run_a_tier_api.py --retry-failed 的补跑结果合并回原结果。

    python merge_retry.py eval_A_causalphys_qwen3max

读取 <dir>/ 与 <dir>_retry/ 中每个任务最新的 samples 文件，按 doc_id 用补跑结果
替换原记录，写到 <dir>_merged/，并打印每个任务合并前后的分数。

分数是逐题指标的简单平均。vsibench 的官方聚合按题型分组再平均，与简单平均不同，
其数字只用于前后对比，不能替代官方分数。
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

FAIL_MARK = "[LMMS_EVAL_REQUEST_FAILED"
SKIP_KEYS = {"doc_id", "doc", "target", "arguments", "resps", "filtered_resps", "token_counts", "doc_hash", "input", "input_media"}
META_KEYS = {"id", "dataset", "scene_name", "question_type", "question", "ground_truth", "options", "prediction"}


def latest_samples(d: pathlib.Path) -> dict[str, pathlib.Path]:
    latest: dict[str, pathlib.Path] = {}
    for f in sorted(glob.glob(str(d / "**" / "*_samples_*.jsonl"), recursive=True)):
        latest[f.split("_samples_")[-1][: -len(".jsonl")]] = pathlib.Path(f)
    return latest


def read(f: pathlib.Path) -> dict[int, dict]:
    with open(f, errors="replace") as fh:
        return {r["doc_id"]: r for r in (json.loads(l) for l in fh if l.strip())}


def score(r: dict) -> float | None:
    for k, v in r.items():
        if k in SKIP_KEYS or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for key in ("score", "accuracy", "acc"):
                if isinstance(v.get(key), (int, float)) and not isinstance(v.get(key), bool):
                    return float(v[key])
            rest = [x for kk, x in v.items() if kk not in META_KEYS and isinstance(x, (int, float)) and not isinstance(x, bool)]
            if len(rest) == 1:
                return float(rest[0])
    return None


def failed(r: dict) -> bool:
    return FAIL_MARK in str(r.get("filtered_resps", ""))


def mean(rows) -> float:
    vals = [s for s in (score(r) for r in rows) if s is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    orig_dir = pathlib.Path(sys.argv[1].rstrip("/"))
    retry_dir = pathlib.Path(f"{orig_dir}_retry")
    out_dir = pathlib.Path(f"{orig_dir}_merged")
    orig, retry = latest_samples(orig_dir), latest_samples(retry_dir)
    if not retry:
        raise SystemExit(f"{retry_dir} 里没有 samples 文件")

    print(f"{'task':40s} {'题数':>5s} {'补跑':>5s} {'仍失败':>6s} {'原分':>7s} {'合并后':>7s}")
    for task, f in sorted(orig.items()):
        rows = read(f)
        before = mean(rows.values())
        redo = read(retry[task]) if task in retry else {}
        rows.update(redo)
        still = sum(failed(r) for r in rows.values())
        dest = out_dir / f.parent.name / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as fh:
            for doc_id in sorted(rows):
                fh.write(json.dumps(rows[doc_id], ensure_ascii=False) + "\n")
        print(f"{task:40s} {len(rows):5d} {len(redo):5d} {still:6d} {before:7.3f} {mean(rows.values()):7.3f}")
    print(f"\n合并结果 → {out_dir}/")


if __name__ == "__main__":
    main()
