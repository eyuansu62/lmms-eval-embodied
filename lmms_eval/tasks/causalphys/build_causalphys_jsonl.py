#!/usr/bin/env python3
"""把 CausalPhys 的逐文件 annotation 归并成单个 JSONL（全部 16 个子类）。

为什么需要这一步
----------------
CausalPhys 是**文件式**数据集：每个样本一个 json（共 3062 个）。两个问题让
``datasets`` 无法直接读：

1. **schema 不一致** —— 不同样本的 ``graph.nodes`` 有的带 ``text`` 字段有的不带，
   列类型推断直接崩：

       DatasetGenerationError: An error occurred while generating the dataset
       ... because column names don't match

2. **仓库太大** —— 27k 文件，让 ``load_dataset`` 遍历整个 repo 不现实。

所以这里先归并：只保留任务需要的字段，``graph`` 序列化成字符串
（留给后续 judge 评分用）。

注意 ``path`` vs ``paths``
--------------------------
上游 46 个 ``optics`` 样本只写了 ``paths``（复数），其余写 ``path``。
这里统一归并到 ``media_paths`` 字段（始终是列表），下游只需读一个字段。

打散（重要）
------------
上游数据**按答案排序**，例如 ``collision_prediction`` 前 150 条全 A、后 150 条全 B，
切换点唯一。若原样保留，任何 ``--limit N (N≤150)`` 都只取到单一标签，
小样本调试的分数毫无意义 —— 实测 ``--limit 8`` 得到 0/8，
既不是判分 bug 也不是模型不会，纯粹是切片伪影。
这里用固定种子打散，保证可复现的同时让任意前缀都覆盖多种标签。

用法
----
    python build_causalphys_jsonl.py

改了上游数据或加了子集后要重跑。
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import unicodedata
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent

# 打散种子。固定值保证可复现。
SHUFFLE_SEED = 20260923

# 子集名 -> (仓库内的 annotation 目录, 本地目录名, 输出 jsonl)
#
# **这里含 12 个子类，但只有 8 个产出独立 task。** 区别：
#
#   8 个「显式点名」子类（#11 #23 #24 #37 #41 五条条目直接写全路径）
#     -> 各有一个 causalphys_<name>.yaml
#
#   4 个「仅通配池」子类（只出现在 #21 的 Perception/** + Anticipation/**
#     + Intervention/** 通配里，文档从未点名）：
#       intention_speculation / viewpoint_transformation /
#       scene_reconstruction / optics
#     -> **没有独立 yaml**，只作为 #21 多跳筛选的候选池
#        （实测筛完只剩 scene_reconstruction 的 5 题）
#
# 这 4 个仍要在这里归并，因为 build_multihop.py 需要读它们的 jsonl 重建池子。
SUBSETS = {
    # --- 8 个显式点名的子类（产出独立 task）---
    "collision_prediction": ("Anticipation/Collision_Prediction", "collision_prediction_annotations"),
    "fluid_flow": ("Anticipation/Fluid_Flow", "fluid_flow_annotations"),
    "deformation": ("Anticipation/deformation", "deformation_annotations"),
    "action_substitution": ("Intervention/action_substitution", "action_substitution_annotations"),
    "spatial_manipulation": ("Intervention/spatial_manipulation", "spatial_manipulation_annotations"),
    "temporal_shifting": ("Intervention/temporal_shifting", "temporal_shifting_annotations"),
    "mechanics_reasoning": ("Perception/Mechanics_Reasoning", "mechanics_reasoning_annotations"),
    "containability": ("Perception/containability", "containability_annotations"),
    # --- 4 个仅通配池的子类（无独立 task，仅供 #21 筛选）---
    "intention_speculation": ("Anticipation/Intention_Speculation", "intention_speculation_annotations"),
    "viewpoint_transformation": ("Intervention/viewpoint_transformation", "viewpoint_transformation_annotations"),
    "scene_reconstruction": ("Perception/Scene_Reconstruction", "scene_reconstruction_annotations"),
    "optics": ("Perception/optics", "optics_annotations"),
}


def _extract_options(question: str) -> str:
    """从 question 里切出选项部分。

    CausalPhys 的选项内嵌在题干里，形如：

        What is the primary cause of ... ? A: The gravitational force. B: The upward force.

    没有独立的 options 字段。这里把 ``A:`` 起始之后的部分切出来。
    切不出来就返回空串（下游会用整段 question）。
    """
    m = re.search(r"\s[A-Z]\s*:", question)
    return question[m.start() :].strip() if m else ""


def _raw_media_paths(ann: dict[str, Any]) -> list[str]:
    """取 ``paths`` → ``path``，统一成列表（与官方 loader 的尝试顺序一致）。

    实测 46 个 ``optics`` 样本只写了 ``paths``，漏掉就会当媒体缺失。
    同时做 NFKC 归一化：上游有 2 个样本把全角右括号 ``）``(U+FF09)
    写进了文件名，仓库里实际是半角 ``)``。
    """
    raw = ann.get("paths")
    if raw is None:
        raw = ann.get("path")
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out = []
    for x in items:
        s = str(x).strip() if x is not None else ""
        if s:
            out.append(unicodedata.normalize("NFKC", s))
    return out


def _normalize(ann: dict[str, Any]) -> dict[str, Any]:
    """挑出任务需要的字段，其余（graph 等）序列化后保留。"""
    question = str(ann.get("question") or "")
    return {
        # 注意：文件名 ≠ 该字段。必须用 JSON 里的 id 去拼 <id>_frames 目录。
        "id": str(ann.get("id") or ""),
        "question": question,
        "options": _extract_options(question),
        "answer": str(ann.get("ground_truth_answer") or ""),
        "category": str(ann.get("category") or ""),
        "sub_category": str(ann.get("sub_category") or ""),
        # 统一成列表：上游有的写 path、有的写 paths。
        "media_paths": json.dumps(_raw_media_paths(ann), ensure_ascii=False),
        # 因果图只给 judge 用，不喂模型；序列化避免 schema 冲突。
        "graph": json.dumps(ann.get("graph") or {}, ensure_ascii=False),
    }


def _build_media_index() -> None:
    """从本地 annotation 目录反推媒体索引。

    ``deformation`` 有 94/200 条样本的 ``path`` 指向**目录**，评测时要把目录
    展开成文件列表。展开需要知道目录里有什么，而仓库有 27k 文件 —— 用
    ``HfApi.list_repo_files`` 翻分页太慢，且它不读 ``HF_ENDPOINT``。

    所以这里生成一份本地索引随任务分发。索引来源是仓库的完整文件清单；
    本脚本假定该清单已存在于 ``/tmp/cp_files.txt``（由 git ls-tree 生成，
    见 DATA_SHAPE.md）。清单不在时跳过，评测会降级为
    不展开目录（仅影响 deformation 的目录形态样本）。
    """
    listing = pathlib.Path("/tmp/cp_files.txt")
    if not listing.is_file():
        print("[skip] 媒体索引：/tmp/cp_files.txt 不存在")
        print("       生成方式：cd <repo> && git ls-tree -r --name-only HEAD > /tmp/cp_files.txt")
        return
    out = HERE / "data" / "media_index.txt"
    n = 0
    with listing.open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as gh:
        for line in fh:
            p = line.strip()
            if p and "/data/" in p and not p.endswith(".mmd"):
                gh.write(p + "\n")
                n += 1
    print(f"media_index.txt            {n:>4d} 条媒体路径")


def main() -> None:
    total_rows = 0
    for name, (_repo_dir, local_dir) in SUBSETS.items():
        src = HERE / "data" / local_dir
        if not src.is_dir():
            print(f"[skip] {name}: {src} 不存在")
            continue

        rows = []
        skipped = []
        for f in sorted(src.glob("*.json")):
            try:
                ann = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                skipped.append((f.name, str(exc)))
                continue
            rows.append(_normalize(ann))

        # 打散：上游按答案排序，不打散则任何前缀都只含单一标签。
        random.Random(SHUFFLE_SEED).shuffle(rows)

        out = HERE / "data" / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_rows += len(rows)
        flag = f"  (跳过 {len(skipped)} 个损坏文件)" if skipped else ""
        print(f"{name:<26s} {len(rows):>4d} 条 -> {out.name}{flag}")

    print()
    _build_media_index()
    print(f"\n合计 {total_rows} 条")


if __name__ == "__main__":
    main()
