#!/usr/bin/env python3
"""把原题与否定题的评测结果按 index 配对，算出成对全对率。

背景
----
lmms-eval 的 task 是「一 doc 一问」，没法让同一条样本在一次运行里问两遍。
而 NegVQA 的价值恰恰在于**配对对照**：同一张图，换个否定表述，模型还答得对吗？

所以两个 task 分开跑（与官方 evaluate_models.sh 的做法一致），
再用这个脚本按 ``index`` 把两边配对，补出第三个指标。

三个指标
--------
* 原题准确率    negvqa_original_accuracy
* 否定题准确率  negvqa_negated_accuracy
* **成对全对率** 同一视觉内容下两问都答对 —— 本脚本计算

成对全对率的意义：它排除「靠猜」。若模型在两问上独立乱猜（各 50%），
期望成对全对率约 25%；把它和单问准确率一起看，能区分
「真的理解了否定」和「运气好猜中一道」。

用法
----
    python pair_negvqa_results.py <original_samples.jsonl> <negated_samples.jsonl>
    python pair_negvqa_results.py results/  # 传目录则自动找两个 samples 文件

输出：终端表格 + 同目录下 pair_negvqa_summary.json
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter


def _load_samples(path: pathlib.Path) -> dict[int, dict]:
    """读 lmms-eval 的 samples jsonl，键为 index。

    samples 文件每行的 doc_id 是该 task 内的序号，不是 NegVQA 的 index。
    这里靠 ``input`` 文本回查原始 index 不可靠，所以改用两种方式之一：
      1. 若行内含 'index' 字段（有些 task 会带上），直接用
      2. 否则按 doc_id 顺序对齐 —— 两个 task 的筛选与顺序一致，序号可对齐
    本实现用方式 2，并在调用处校验两边样本数相同。
    """
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "index" in r:
                key = int(r["index"])
            else:
                key = int(r.get("doc_id", len(rows)))
            rows[key] = r
    return rows


def _find_samples(dir_or_file: pathlib.Path) -> list[pathlib.Path]:
    if dir_or_file.is_file():
        return [dir_or_file]
    return sorted(dir_or_file.rglob("*samples*.jsonl"))


def _pick(files: list[pathlib.Path], keyword: str) -> pathlib.Path | None:
    for f in files:
        if keyword in f.name:
            return f
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    target = pathlib.Path(sys.argv[1])
    files = _find_samples(target)
    if len(target.is_file() and [target] or files) == 1 and len(files) == 1:
        print("需要两个 samples 文件（原题与否定题）。传目录会自动匹配。")
        return 1

    orig_f = _pick(files, "original")
    neg_f = _pick(files, "negated")
    if not orig_f or not neg_f:
        print(f"没找到成对文件。候选：{[f.name for f in files]}")
        print("文件名需分别含 'original' 与 'negated'。")
        return 1

    print(f"原题  : {orig_f}")
    print(f"否定题: {neg_f}\n")

    orig = _load_samples(orig_f)
    neg = _load_samples(neg_f)

    if len(orig) != len(neg):
        print(f"⚠️ 两边样本数不同（{len(orig)} vs {len(neg)}），按交集配对")

    keys = sorted(set(orig) & set(neg))
    if not keys:
        print("没有可配对的 index。")
        return 1

    n = len(keys)
    o_hit = n_hit = both = 0
    combo: Counter[str] = Counter()

    for k in keys:
        o = orig[k]
        g = neg[k]
        # samples 行里 filtered_resps 是模型输出，target 是真值
        o_pred = str(o.get("filtered_resps") or "").strip().upper()[:1]
        o_tgt = str(o.get("target") or "").strip().upper()[:1]
        n_pred = str(g.get("filtered_resps") or "").strip().upper()[:1]
        n_tgt = str(g.get("target") or "").strip().upper()[:1]

        o_ok = o_pred == o_tgt
        n_ok = n_pred == n_tgt
        o_hit += o_ok
        n_hit += n_ok
        both += o_ok and n_ok
        combo[f"原题{'✓' if o_ok else '✗'} 否定{'✓' if n_ok else '✗'}"] += 1

    print(f"配对样本 {n} 条\n")
    print(f"{'指标':<24s} {'值':>8s}")
    print("-" * 34)
    print(f"{'原题准确率':<24s} {o_hit / n:>8.4f}")
    print(f"{'否定题准确率':<24s} {n_hit / n:>8.4f}")
    print(f"{'成对全对率':<24s} {both / n:>8.4f}")
    print(f"{'落差（原题-否定）':<24s} {(o_hit - n_hit) / n:>+8.4f}")

    print("\n四象限分布（排除随机猜测的干扰）:")
    for k in ["原题✓ 否定✓", "原题✓ 否定✗", "原题✗ 否定✓", "原题✗ 否定✗"]:
        v = combo.get(k, 0)
        print(f"   {k:<16s} {v:>4d}  {v / n:>6.1%}")

    summary = {
        "n_paired": n,
        "original_accuracy": o_hit / n,
        "negated_accuracy": n_hit / n,
        "paired_accuracy": both / n,
        "gap": (o_hit - n_hit) / n,
        "quadrants": dict(combo),
    }
    out = orig_f.parent / "pair_negvqa_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
