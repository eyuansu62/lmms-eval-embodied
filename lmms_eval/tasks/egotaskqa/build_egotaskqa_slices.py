#!/usr/bin/env python3
"""EgoTaskQA 按文档 P0 条目筛选切片。

数据源选择（重要）
------------------
文档「子集定位」里 EgoTaskQA 的 repo/config/split 三栏**全部标"未核验"**，
只给了三个能力名（#13 时序 / #22 Counterfactual / #24 Explanatory）。

内置 task 指向的 `nv-njb/EgoTaskQA-MCQ` **无法使用** —— 它只有
`video_path / q / option / a` 四个字段，**没有任何分类标签**：

    {"video_path": "27k-22-11-2|P1|4546|4986.mp4",
     "q": "What is the status of sandwich before ...",
     "option": {"A": "...", "B": "...", ...}, "a": "A"}

无法据此筛出"反事实""归因"题。

改用 `lotsoliu/EgoTaskQA` 的 `tagged_qas.json`（40323 条），它带完整标签：

    question / type / category / semantic / structural / step / program /
    answer / interval / reference / reasoning_type / video_id / question_id

    type: descriptive(20899) explanatory(9936) counterfactual(6739) prediction(4099)

**`type` 的取值正好就是文档写的命名**，故 #22 / #24 可精确匹配。

条目判据
--------
| 条目 | 能力 | 判据 | 命中 |
|---|---|---|---|
| #13 | 事件顺序理解 | `step >= 2` AND 题干含时序关键词 | 见下 |
| #22 | 反事实推理   | `type` 含 `counterfactual` | 6739 |
| #24 | 归因理解     | `type` 含 `explanatory` | 9936 |

### #13 的判据为何是"step>=2 + 时序关键词"

`type` 里**没有"时序"这个值**，只有 descriptive/explanatory/counterfactual/
prediction。而"时序/动作依赖"在 EgoTaskQA 里由两个信号表达：

* `step` —— 推理步数（1/2/3），步数 ≥2 表示需要跨多个事件环节
* 题干含 `before/after/first/then/next/order/sequence` 等时序词

单用关键词太宽（命中 21096 条，跨越全部 type）；叠加 `step>=2` 后收窄，
贴近"需多步推理的事件顺序"。

用法
----
    python build_egotaskqa_slices.py --input /path/to/tagged_qas.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent

_SEQ_WORDS = re.compile(
    r"\b(before|after|first|then|next|order|sequence|previous|last|following)\b",
    re.I,
)


def _has_type(row: dict, name: str) -> bool:
    t = row.get("type")
    if isinstance(t, list):
        return name in t
    return t == name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="lotsoliu/EgoTaskQA 的 tagged_qas.json")
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    if not src.is_file():
        raise SystemExit(f"找不到 {src}")

    rows = json.loads(src.read_text(encoding="utf-8"))
    print(f"输入 {len(rows)} 条 <- {src.name}\n")

    slices = {
        "event_order": (
            "事件顺序理解", "#13",
            lambda r: (r.get("step") or 0) >= 2 and bool(_SEQ_WORDS.search(r.get("question", ""))),
        ),
        "counterfactual": (
            "反事实推理", "#22",
            lambda r: _has_type(r, "counterfactual"),
        ),
        "explanatory": (
            "归因理解", "#24",
            lambda r: _has_type(r, "explanatory"),
        ),
    }

    used: Counter[int] = Counter()
    for name, (cn, entry, pred) in slices.items():
        hit = [r for r in rows if pred(r)]
        out = HERE / "data" / f"egotaskqa_{name}.jsonl"
        idx_out = HERE / "data" / f"egotaskqa_{name}_indices.json"
        with out.open("w", encoding="utf-8") as fh:
            for r in hit:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        idx_out.write_text(
            json.dumps(sorted(int(r["question_id"]) for r in hit)), encoding="utf-8"
        )
        for r in hit:
            used[int(r["question_id"])] += 1

        print(f"{entry:<5s} {name:<16s} {len(hit):>5d} 条  ({cn})")
        if name == "event_order":
            print(f"        step 分布: {dict(Counter(r.get('step') for r in hit).most_common())}")
            print(f"        type 分布: {dict(Counter(t for r in hit for t in (r.get('type') or [])).most_common())}")

    dup = {k: v for k, v in used.items() if v > 1}
    print(f"\n跨切片重复: {len(dup)} 个")
    print(f"三切片合计 {len(used)} 条（去重后）")


if __name__ == "__main__":
    main()
