#!/usr/bin/env python3
"""CausalPhys #24「归因理解」的筛题脚本。

文档要求
--------
能力体系（CAPABILITIES.md）条目 #24：

    文件/目录：Perception/Mechanics_Reasoning/annotations/*.json
    选取条件：Perception/Mechanics_Reasoning；**保留 question 询问已观察结果之原因的题**
    尚需处理：**100题可作归因种子**；**未把整个 Mechanics_Reasoning 当已筛选归因子集**

也就是说，整目录 298 题**不能**直接当归因子集用，必须筛出"由结果反推原因"的题。

判据
----
只保留题干**明确在问原因**的题 —— 即形如：

    What is the primary cause of the elephant figurine's movement ... ?
    What caused the ball to roll off the table ... ?
    Why did the box fall ... ?

排除的典型（同目录下的其他问法）：

    How the Paint tube will move relative to the Group of paint tubes?   <- 方位题
    What is the causal relationship between the hand and the object?     <- 关系描述，非归因
    Which object comes to rest first?                                    <- 顺序题

**注意**：判据从严。"relationship"（关系描述）不算归因 —— 它问的是两者关系，
不是"是什么导致了当前状态"。故 ``causal relationship`` 一律排除，
即使句中含有 "causal" 一词。

用法
----
    python build_attribution.py

输出 ``data/mechanics_reasoning_attribution.jsonl``（筛出的题）
与 ``data/mechanics_reasoning_attribution_indices.json``（保留的 id 列表）。
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent

SOURCE = HERE / "data" / "mechanics_reasoning.jsonl"
OUT_JSONL = HERE / "data" / "mechanics_reasoning_attribution.jsonl"
OUT_INDICES = HERE / "data" / "mechanics_reasoning_attribution_indices.json"

# 保留：题干明确询问原因的句式。
_KEEP = re.compile(
    r"(?:"
    r"primary\s+cause"
    r"|(?:what|which)\s+(?:is|are)\s+the\s+(?:primary\s+|main\s+|direct\s+|likely\s+)?cause"
    r"|what\s+(?:caused|causes|cause)"
    r"|why\s+(?:did|does|is|are|was|were|would)"
    r"|(?:the\s+)?reason\s+(?:for|behind|why)"
    r"|what\s+(?:makes|made)"
    r"|(?:is|are)\s+responsible\s+for"
    r")",
    re.I,
)

# 排除：关系描述类。即使含 "causal" 也不属归因（问的是关系，不是起因）。
_REJECT_RELATIONSHIP = re.compile(
    r"causal\s+relationship|relationship\s+between|what\s+is\s+the\s+relationship",
    re.I,
)


def is_attribution(question: str) -> bool:
    """题干是否在问"已观察结果的原因"。"""
    q = str(question or "")
    if _REJECT_RELATIONSHIP.search(q):
        return False
    return bool(_KEEP.search(q))


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"找不到 {SOURCE}，请先跑 build_causalphys_jsonl.py")

    rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    kept: list[dict[str, Any]] = []
    for r in rows:
        if is_attribution(r.get("question")):
            kept.append(r)

    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    OUT_INDICES.write_text(
        json.dumps(sorted(str(r["id"]) for r in kept)), encoding="utf-8"
    )

    print(f"输入 {len(rows)} 题（Mechanics_Reasoning 整目录）")
    print(f"筛出 {len(kept)} 题（{len(kept) / len(rows) * 100:.1f}%）")
    print("文档预期：约 100 题可作归因种子")
    print(f"\n-> {OUT_JSONL}")
    print(f"-> {OUT_INDICES}")

    print("\n保留样例：")
    for r in kept[:6]:
        print(f"   {str(r['question'])[:96]}")


if __name__ == "__main__":
    main()
