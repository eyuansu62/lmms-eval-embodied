#!/usr/bin/env python3
"""RefSpatial-Expand-Bench 按文档 P0 条目筛选切片。

数据源
------
文档「子集定位」#19 / #20 指定的是 `JingkunAn/RefSpatial-Expand-Bench`，
split=location（241 条）。

**注意：内置 task 指向的不是同一个数据集。**
内置 `refspatial/_default_template_yaml` 用的是 `BAAI/RefSpatial-Bench`，
与文档要求不同。本脚本按文档改用前者的 location split。

字段（实测）：
    id / scene / image / mask / object / prompt / suffix / step

—— 与内置 `utils.py` 期望的字段**完全一致**，所以切片可直接复用其
`refspatial_doc_to_text` / `refspatial_doc_to_visual` /
`refspatial_process_results`，无需改动判分逻辑。

条目判据
--------
| 条目 | 能力 | 判据 | 命中 |
|---|---|---|---|
| #19 | 属性指代 | `id IN {1,3,9}` | 3 |
| #20 | 序数指代 | `id IN {0,2,8,13}` | 4 |

### 为何样本这么少

文档明确写了 `id IN {...}` 的白名单，并附注"**其余 location 需审标后加入**"。
也就是说文档本身只锁定了这 7 条作为种子，其余 233 条需要人工审标后才能纳入。

**这 7 条的样本量极小**，单题波动 ±14%，分数只能作定性参考，不能当作稳定指标。
若要更大样本量，需按文档所说先做审标 —— 那是人工工作，脚本无法代劳。

### id=3 的归属

`id=3` 的 prompt 是 "the furthest brown car" —— 同时含属性（brown）和
最高级序数（furthest）。文档把它归入 #19（属性指代），此处遵从文档。

用法
----
    python build_refspatial_slices.py
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# 文档指定的白名单
SLICES = {
    "attribute_reference": ("属性指代", "#19", {1, 3, 9}),
    "ordinal_reference": ("序数指代", "#20", {0, 2, 8, 13}),
}


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset(
        "JingkunAn/RefSpatial-Expand-Bench", split="location"
    )
    rows = [dict(d) for d in ds]
    print(f"输入 {len(rows)} 条（location split）\n")

    used: set[int] = set()
    for name, (cn, entry, ids) in SLICES.items():
        hit = [r for r in rows if int(r["id"]) in ids]
        out = HERE / "data" / f"refspatial_{name}.jsonl"
        idx_out = HERE / "data" / f"refspatial_{name}_indices.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in hit:
                # image / mask 是 PIL / ndarray，不能直接 json 序列化，
                # 故只落 id，图像由 yaml 读 HF dataset 时带上。
                fh.write(json.dumps({"id": int(r["id"])}, ensure_ascii=False) + "\n")
        idx_out.write_text(
            json.dumps(sorted(int(r["id"]) for r in hit)), encoding="utf-8"
        )
        used |= {int(r["id"]) for r in hit}

        print(f"{entry:<5s} {name:<22s} {len(hit):>3d} 条  ({cn})")
        for r in hit:
            print(f"        id={r['id']:<3d} {str(r['object'])[:52]}")

    print(f"\n合计 {len(used)} 条")
    print("⚠️ 样本量极小，分数仅作定性参考")
    print("   文档注：其余 location 需审标后加入")


if __name__ == "__main__":
    main()
