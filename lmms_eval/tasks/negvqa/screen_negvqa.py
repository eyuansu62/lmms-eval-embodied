#!/usr/bin/env python3
"""NegVQA 严格筛选脚本（全部自动，规则从严）。

背景
----
NegBench 不适配生成式 VLM（评测协议是 CLIP 嵌入相似度，且答案恒定放首位）。
改用 NegVQA（ACL 2025 Findings）—— 原生两选项 VQA，模型直接生成选项字母，
官方在 VLMEvalKit 上评过 Qwen2-VL / LLaVA / Molmo 等。

但 NegVQA 的否定是**用 LLM 自动改写**生成的，答案则机械翻转。实测确认存在缺陷：

    index=326 (MMMU)
      original: ...Predict the overtime wages if 6,500 invoices are processed.  -> B
      negated : ...Do not predict the overtime wages if 6,500 invoices are processed. -> A

"不要预测"是禁止*预测行为*，不等于"预测一个错误数值"，答案翻转在语义上不成立。

dev 全量 1000 例的答案是**完美一一翻转**（488A/512B -> 488B/512A），
说明整批都是自动改写 + 机械对调，缺陷不会只出现在个别样本。

筛选策略
--------
**全部自动，规则从严**（宁可少留）。三层过滤：

1. **来源档**：只保留自然场景类来源。排除文档/图表/OCR（InfoVQA、DocVQA、
   ChartQA、TableVQABench、OCRVQA、TextVQA）与学科推理
   （AI2D、ScienceQA、MathVision、MathVista、MMMU、MMVet）。
   依据：离线用官方预测实测，三个档位的"原题-否定"落差差 15 倍
   （自然场景 0.264 / 文档图表 0.097 / 学科推理 0.018），
   学科推理档落差近零，混入只会稀释"否定理解"这个信号。

2. **否定句式**：只保留否定构成**可判定二选一**的形式。
   - 保留：`how many/of` + `are not in/on/present/visible/parked/...`
     （存在性补集，有唯一答案）；`what` + `is not on/in/visible/shown/...`；
     极性问句（是/否）且选项为 Yes/No。
   - 排除：`is not the [X]` 这类——"不是什么"开放无唯一解；
     `What type of book is this not?` 这类——否定形式语法不通。

3. **数据自洽**：要求 `original_answer != negated_answer`（否定确实翻转了答案），
   且两者都在选项集合 {A, B} 内。

4. **排除数量类**：`how many/much` 或以纯数字为选项的题一律不要。
   理由：**"图里有几个 X"的补集不是另一个数字**。
   例如 ``How many bananas are there in the image? 4 / 2``，
   否定后变成"有几根香蕉不在图里"，答案标为 4 —— 这在语义上不成立。
   这类题语法通顺，比 ``index=326`` 那种"禁止预测"句式更隐蔽，
   但与后者是同一类毛病：机械翻转答案。

   保留下来的只有**互斥二选一**：朝向、位置、类别、月相这类
   "选了 A 就不可能是 B"的属性题。只有在这些题上，
   "不是 A"才等价于"是 B"，答案翻转才成立。

筛选结果是**自定义评测切片**，不冒充官方子集。

用法
----
    python screen_negvqa.py --input dev.parquet --output data/negvqa_natural_scene.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

# --- 第 1 层：来源档 ---

# 自然场景类来源（保留）
NATURAL_SOURCES = {
    "OKVQA",
    "A-OKVQA",
    "GQA",
    "VQAv2",
    "RealWorldQA",
    "VizWiz",
    "SEEDBench",
    "MMStar",
}

# --- 第 2 层：否定句式 ---

_IMG_TAG = re.compile(r"<image \d+>")

# 极性问句（是/否）的开头
_POLAR_HEAD = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|has|have|will|would)\b", re.I
)

# 保留：存在性/可见性补集 —— 有唯一答案
# 形如 "how many X are not in/on/present/visible/..." 或 "what is not on/in/..."
_KEEP_NEG = re.compile(
    r"\bnot\s+(?:in|on|at|present|visible|shown|depicted|parked|there|"
    r"wearing|holding|carrying|using|contained|found|displayed|standing|"
    r"sitting|lying|located|open|closed|facing|belong)\b",
    re.I,
)

# 排除：`is not the [X]` —— "不是什么"开放，无唯一答案
_REJECT_NOT_THE = re.compile(r"\b(?:is|are|was|were)\s+not\s+the\b", re.I)

# 排除：禁止回答/预测/假设类句式（"不要预测"不等于"预测错值"）
_REJECT_FORBID = re.compile(
    r"\b(?:do not|don't|never)\s+(?:predict|answer|guess|tell|say|assume|"
    r"use|consider|mention|include|exclude)\b",
    re.I,
)

# 排除：句末直接缀 not（语法不通，如 "What type of book is this not?"）
_REJECT_TRAILING_NOT = re.compile(r"\bnot\s*\?\s*$", re.I)

# 排除：`what ... what is not the answer to` 这类元问题
_REJECT_META = re.compile(r"\bnot\s+the\s+answer\b", re.I)

# 排除：数量类。`how many/much` 或以纯数字为选项的题，否定后补集不是另一个数。
_HOW_MANY_HEAD = re.compile(r"^\s*how\s+(?:many|much)\b", re.I)
_PURE_NUMBER = re.compile(r"^\s*[$¥€£]?\s*[\d,]+(?:\.\d+)?\s*%?\s*$")


def _clean(s: Any) -> str:
    return _IMG_TAG.sub("", str(s or "")).strip()


def is_natural_source(category: str) -> bool:
    return str(category or "").strip() in NATURAL_SOURCES


def negation_is_well_formed(orig: str, neg: str) -> tuple[bool, str]:
    """判断这条否定改写是否构成可判定的二选一。

    返回 ``(是否保留, 原因)``，原因用于统计与回溯。
    """
    o = _clean(orig)
    n = _clean(neg)

    if not o or not n or o == n:
        return False, "未改写"

    # 明确的排除项优先
    if _REJECT_FORBID.search(n):
        return False, "禁止回答/预测句式"
    if _REJECT_TRAILING_NOT.search(n):
        return False, "句末缀 not，语法不通"
    if _REJECT_META.search(n):
        return False, "元问题（not the answer）"
    if _REJECT_NOT_THE.search(n):
        return False, "is not the X：开放无唯一解"

    # 极性问句：排除掉上面的问题后，只要 n 含否定就保留
    if _POLAR_HEAD.match(o):
        if re.search(r"\bnot\b|n't", n, re.I):
            return True, "极性问句"
        return False, "极性问句但未含否定"

    # 其余（what / how many / which / where / who）：要求命中存在性补集模式
    if _KEEP_NEG.search(n):
        return True, "存在性/可见性补集"

    return False, "否定形式无法判定唯一答案"


def answer_flipped(doc: dict) -> bool:
    """否定确实翻转了答案，且两者都在合法选项内。"""
    oa = str(doc.get("original_answer") or "").strip().upper()
    na = str(doc.get("negated_answer") or "").strip().upper()
    if oa not in {"A", "B"} or na not in {"A", "B"}:
        return False
    return oa != na


def is_cardinality(doc: dict) -> bool:
    """是否是数量类题（否定后补集无意义）。

    两种情况都算：
      * 题干以 ``how many`` / ``how much`` 开头
      * 两个选项都是纯数字

    这类题的"补集"不是另一个数：``How many bananas are there? 4 / 2``
    否定后变成"有几根香蕉不在图里"，答案标为 4 —— 语义不成立。
    """
    q = _clean(doc.get("original_question"))
    if _HOW_MANY_HEAD.match(q):
        return True
    a = str(doc.get("A") or "")
    b = str(doc.get("B") or "")
    return bool(_PURE_NUMBER.match(a) and _PURE_NUMBER.match(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 parquet（dev 或 test）")
    ap.add_argument("--output", required=True, help="输出 jsonl")
    ap.add_argument(
        "--include-image",
        action="store_true",
        help="把 image 一起写出（体积大；默认只写索引与题面，图像由 yaml 读 parquet）",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=args.input, split="train")
    print(f"输入 {len(ds)} 例 <- {args.input}\n")

    from collections import Counter

    stage1 = stage2 = stage3 = stage4 = 0
    reject_reasons: Counter[str] = Counter()
    kept: list[dict] = []

    for d in ds:
        if not is_natural_source(d.get("category")):
            continue
        stage1 += 1

        ok, why = negation_is_well_formed(d.get("original_question"), d.get("negated_question"))
        if not ok:
            reject_reasons[why] += 1
            continue
        stage2 += 1

        if not answer_flipped(d):
            reject_reasons["答案未翻转或越界"] += 1
            continue
        stage3 += 1

        if is_cardinality(d):
            reject_reasons["数量类：补集非另一个数"] += 1
            continue
        stage4 += 1

        row = {
            "index": int(d["index"]),
            "category": str(d.get("category") or ""),
            "original_question": str(d.get("original_question") or ""),
            "negated_question": str(d.get("negated_question") or ""),
            "A": str(d.get("A") or ""),
            "B": str(d.get("B") or ""),
            "original_answer": str(d.get("original_answer") or "").strip().upper(),
            "negated_answer": str(d.get("negated_answer") or "").strip().upper(),
            "negation_form": why,
        }
        if args.include_image:
            img = d.get("image")
            # PIL 不能直接 json 序列化；写出时转 base64 PNG
            import base64
            import io

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            row["image_png_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        kept.append(row)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 同时写出保留索引 —— yaml 用 parquet 作 dataset_path（图片内嵌其中），
    # 再靠 process_docs 按这份索引过滤。这样不必把图片单独落盘。
    idx_out = out.with_name(out.stem + "_indices.json")
    idx_out.write_text(
        json.dumps(sorted(r["index"] for r in kept)), encoding="utf-8"
    )

    print(f"第 1 层 自然场景来源     {stage1:>5d}")
    print(f"第 2 层 否定句式合格     {stage2:>5d}")
    print(f"第 3 层 答案正确翻转     {stage3:>5d}")
    print(f"\n保留 {len(kept)}/{len(ds)} 例 -> {out}")

    print("\n排除原因分布:")
    for k, v in reject_reasons.most_common():
        print(f"   {v:5d}  {k}")

    if kept:
        print("\n保留样本的来源分布:")
        for k, v in Counter(r["category"] for r in kept).most_common():
            print(f"   {v:5d}  {k}")
        print("\n保留样本的否定形式分布:")
        for k, v in Counter(r["negation_form"] for r in kept).most_common():
            print(f"   {v:5d}  {k}")


if __name__ == "__main__":
    main()
