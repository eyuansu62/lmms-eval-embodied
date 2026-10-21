#!/usr/bin/env python3
"""MM-UPD（Unsolvable Problem Detection）· #47 自我检视 → 本地 jsonl + 图片。

    python lmms_eval/tasks/mmupd_cap47/build_mmupd_jsonl.py

产物：
    lmms_eval/tasks/mmupd_cap47/data/mmupd_{aad,iasd,ivqd}.jsonl   50 道原题 × 2 版（进 git，Apache-2.0）
    lmms_eval/tasks/mmupd_cap47/media/<setting>/<index>_<type>.<ext>（不进 git，可由本脚本重建）

数据：HF MM-UPD/MM-UPD（Apache-2.0），论文 https://arxiv.org/abs/2403.20331 。
仓库里的上游 task（lmms_eval/tasks/mmupd/）用不了：数据集脚本 MM-UPD.py 已不被 datasets 支持，
判分又要在聚合阶段调 OpenAI。这里直接读官方 TSV。

结构（逐行核对过）
------------------
* 每道原题 g 有一对版本：standard（正常题）与 upd（无解题，正确答案恒为 F = None of the above）。
  三种无解构造：AAD 删掉正确选项；IASD 换成与题目无关的选项；IVQD 换成与题目不符的图片。
* index = g + k·1e6 是 MMBench 式**循环评测**的选项轮换版本（每版 2–4 个）。
  官方双重准确率要求所有轮换都对。**这里只取 k=0 的主版本**（#47 的请求预算约 400，
  官方循环评测 50 题就要 ~350 次请求）—— 口径因此偏离官方，报表标「非循环」。
* 抽样：每种设定在「主版本成对」的原题里固定种子抽取，AAD 17 / IASD 17 / IVQD 16 = 50。
"""

from __future__ import annotations

import base64
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
REPO_ID = "MM-UPD/MM-UPD"
SEED = 20260926  # 与 A 档 build_sample.py 同一个种子
SETTINGS = {"aad": 17, "iasd": 17, "ivqd": 16}


def _ext(b: bytes) -> str:
    return "png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    for setting, n in SETTINGS.items():
        tsv = hf_hub_download(REPO_ID, f"data/mm{setting}_20240303_base.tsv", repo_type="dataset")
        d = pd.read_csv(tsv, sep="\t")
        main_rows = d[d["index"] < 1_000_000]
        paired = sorted(set(main_rows[main_rows["type"] == "standard"]["index"])
                        & set(main_rows[main_rows["type"] == "upd"]["index"]))
        picked = sorted(random.Random(f"{SEED}:{setting}").sample(paired, n))

        media = HERE / "media" / setting
        media.mkdir(parents=True, exist_ok=True)
        out = HERE / "data" / f"mmupd_{setting}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for g in picked:
                for t in ("standard", "upd"):
                    r = main_rows[(main_rows["index"] == g) & (main_rows["type"] == t)]
                    assert len(r) == 1, (setting, g, t, len(r))
                    r = r.iloc[0]
                    img = base64.b64decode(r["image"])
                    rel = f"{setting}/{g}_{t}.{_ext(img)}"
                    (HERE / "media" / rel).write_bytes(img)
                    options = {k: str(r[k]) for k in "ABCDE"
                               if k in r and pd.notna(r[k]) and str(r[k]).strip() not in ("", "nan")}
                    hint = r.get("hint")
                    rec = {
                        "id": f"{setting}_{g}_{t}",
                        "pair_id": f"{setting}_{g}",
                        "setting": setting,
                        "type": t,
                        "index": int(g),
                        "question": str(r["question"]),
                        "hint": str(hint) if pd.notna(hint) and str(hint) != "nan" else "",
                        "options": options,
                        "answer": str(r["answer"]).strip(),
                        "category": str(r["category"]),
                        "l2_category": str(r["l2-category"]),
                        "image": rel,
                    }
                    assert (t == "upd") == (rec["answer"] == "F"), rec["id"]
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"mmupd_{setting}: {n} 道原题 × 2 版 -> {out.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
