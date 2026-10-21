#!/usr/bin/env python3
"""基于因果图链长筛选 CausalPhys 的多跳题（P0 #23 的筛题标准）。

背景
----
能力体系（CAPABILITIES.md） 的条目 #23 要求从 ``Perception/**`` + ``Anticipation/**`` +
``Intervention/**`` 中"审查 question 与 graph，保留需跨 ≥2 个因果环节才能作答的题"。

这是**本方案的筛题标准，不是原生 multi_hop 标签** —— 文档自己提醒：

    仅 graph 有长度 2 路径不足以证明问题要求多跳。

所以阈值取 **≥3 条因果边**（链上存在中间环节，形如 A→B→C→D），
而不是 ≥2 条（A→B→C，即"长度 2 路径"）。

链长怎么算
----------
graph 的节点有三种 type：

* ``object``    —— 场景参与者。出现在链条上（如示例里的 ``cloth → a1``），
                  但不构成因果**环节**，所以不计入链长。
* ``event``     —— 动作/事件。
* ``attribute`` —— 状态/属性。

只保留 ``event``/``attribute`` 之间的边，在这些"因果节点"构成的 DAG 上求
最长路径的边数。示例（spatial_manipulation id=048）：

    obj   → e1    e1 → a1    a1 → a2      <- 因果链：e1→a1→a2，2 条边
    cloth → a1                             <- cloth 是 object，不计

用法
----
    python build_multihop.py

输出 ``data/multihop_pool.jsonl``（通过筛选的题，含 ``chain_len`` 字段），
供新增的 ``causalphys_multi_hop`` task 使用。
"""

from __future__ import annotations

import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# 因果链条上的节点类型。object 是场景参与者，不算环节。
_CAUSAL_TYPES = {"event", "attribute"}

# 阈值：链上至少这么多条边（= 至少这么多个因果环节的传递）。
MIN_CHAIN_EDGES = 3

# 参与筛选的子集。
#
# 文档 #23 的选取条件是**通配前缀**：
#     Perception/**/annotations/*.json
#     Anticipation/**/annotations/*.json
#     Intervention/**/annotations/*.json
# 所以这 12 个子类都在范围内（Goal_orientation 的 4 个不在，通配未提该目录）。
#
# 注意：其余 8 个各条 P0 条目显式点名的子类（Mechanics_Reasoning、
# Collision_Prediction 等）**不在此列** —— 它们走各自的独立 task。
# 见 README 的「与文档的对应关系」一节。
POOL_SUBSETS = [
    # Anticipation/**
    "intention_speculation",
    # Intervention/**
    "viewpoint_transformation",
    "spatial_manipulation",
    "temporal_shifting",
    "action_substitution",
    # Perception/**
    "scene_reconstruction",
    "optics",
]


def longest_causal_chain(graph: dict) -> int:
    """因果节点 DAG 上最长路径的**边数**。

    只走 ``event``/``attribute`` 之间的边。用 DFS + 访问集防环
    （数据里理论上应是 DAG，但上游标注未必严格）。
    """
    nodes = {n.get("id"): n for n in (graph.get("nodes") or [])}
    edges = [(e.get("from"), e.get("to")) for e in (graph.get("edges") or [])]

    causal_edges = [
        (a, b)
        for a, b in edges
        if nodes.get(a, {}).get("type") in _CAUSAL_TYPES
        and nodes.get(b, {}).get("type") in _CAUSAL_TYPES
    ]

    adj = collections.defaultdict(list)
    for a, b in causal_edges:
        adj[a].append(b)

    def dfs(u: str, seen: frozenset[str]) -> int:
        """从 u 出发的最长路径边数。"""
        if u in seen:
            return 0
        seen = seen | {u}
        best = 0
        for v in adj[u]:
            best = max(best, 1 + dfs(v, seen))
        return best

    return max((dfs(n, frozenset()) for n in nodes), default=0)


def main() -> None:
    pool: list[dict] = []
    per_subset = collections.Counter()
    kept = collections.Counter()

    for name in POOL_SUBSETS:
        src = HERE / "data" / f"{name}.jsonl"
        if not src.is_file():
            print(f"[skip] {src.name} 不存在")
            continue
        for line in src.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            per_subset[name] += 1
            try:
                graph = json.loads(row.get("graph") or "{}")
            except Exception:  # noqa: BLE001
                continue
            chain = longest_causal_chain(graph)
            if chain >= MIN_CHAIN_EDGES:
                row["chain_len"] = chain
                row["source_subset"] = name
                pool.append(row)
                kept[name] += 1

    # 打散：池子是按子类顺序拼的，且各子类内部已打散过。
    # 这里再打散一次，避免任何前缀只覆盖单一子类。
    import random

    random.Random(20260923).shuffle(pool)

    out = HERE / "data" / "multihop_pool.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in pool:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"阈值：因果链 >= {MIN_CHAIN_EDGES} 条边\n")
    print(f"{'子集':<26s} {'总数':>6s} {'入选':>6s} {'占比':>7s}")
    print("-" * 50)
    for name in POOL_SUBSETS:
        t = per_subset[name]
        k = kept[name]
        pct = f"{k / t * 100:.1f}%" if t else "—"
        print(f"{name:<26s} {t:>6d} {k:>6d} {pct:>7s}")
    print("-" * 50)
    print(f"{'合计':<26s} {sum(per_subset.values()):>6d} {len(pool):>6d}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
