#!/usr/bin/env python3
"""OpenEQA · 场景语义两类 → 本地 jsonl + 帧图（#6）。

    python lmms_eval/tasks/openeqa_scene/build_openeqa_jsonl.py --src /tmp/oeqa_shards

产物（图片不进 git，可由此脚本重建；jsonl 进 git）：
    lmms_eval/tasks/openeqa_scene/data/openeqa_{world_knowledge,functional_reasoning}.jsonl
    lmms_eval/tasks/openeqa_scene/media/<qid>/<i>.jpg

数据：HF thomas-yanxin/open-eqa-mirror（OpenEQA，CVPR 2024；论文 2023-11 arXiv 2312.17852 的镜像）。
17 个 parquet，1636 题 / 152 个场景；每题 `videos` 是 32 张 1920×1080 的 JPEG 帧。

只取两类（argparse 全量统计：world knowledge 213、functional reasoning 217，合计 430）：
    world knowledge        客厅什么吊顶 / 车库什么车 / 什么装修风格
    functional reasoning   怎么凉快下来 / 去哪里洗手
其余五类（物体识别/定位/属性/空间/状态）属于别的能力，不收。

帧：每题从 32 帧里均匀取 --frames 张（默认 16），长边压到 960、JPEG q=85。
32 帧原图约 6 MB，正好卡在网关单请求 6 MB 的体量上限，压到 16 帧 ×~80 KB 后 base64 约 1.7 MB。
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20260926
CATS = {"world knowledge": "world_knowledge", "functional reasoning": "functional_reasoning"}
PER_CAT = {"world knowledge": 75, "functional reasoning": 75}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="17 个 parquet 所在目录（<src>/data/*.parquet）")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=960)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from PIL import Image

    files = sorted(glob.glob(str(pathlib.Path(args.src) / "data" / "*.parquet")))
    assert len(files) == 17, files
    d = pd.concat([pd.read_parquet(f, columns=["question_id", "category", "question", "answer",
                                               "episode_history", "videos"]) for f in files])

    for cat, slug in CATS.items():
        xs = d[d["category"] == cat].sort_values("question_id").reset_index(drop=True)
        assert len(xs) >= PER_CAT[cat], (cat, len(xs))
        picked = sorted(random.Random(f"{SEED}:openeqa:{cat}").sample(range(len(xs)), PER_CAT[cat]))
        out = HERE / "data" / f"openeqa_{slug}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        n_frames = 0
        with open(out, "w", encoding="utf-8") as fh:
            for i in picked:
                r = xs.iloc[i]
                frames = r["videos"]
                assert len(frames) == 32, len(frames)
                idx = np.linspace(0, len(frames) - 1, args.frames, dtype=int)
                qdir = HERE / "media" / r["question_id"]
                qdir.mkdir(parents=True, exist_ok=True)
                rels = []
                for j, k in enumerate(idx):
                    im = Image.open(io.BytesIO(frames[int(k)]["bytes"])).convert("RGB")
                    im.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
                    rel = f"{r['question_id']}/{j:02d}.jpg"
                    im.save(HERE / "media" / rel, format="JPEG", quality=85)
                    rels.append(rel)
                n_frames += len(rels)
                fh.write(json.dumps({
                    "id": r["question_id"],
                    "category": cat,
                    "question": str(r["question"]),
                    "answer": str(r["answer"]),
                    "scene": str(r["episode_history"]),
                    "frames": rels,
                }, ensure_ascii=False) + "\n")
        print(f"openeqa_{slug}: {PER_CAT[cat]} 题 × {args.frames} 帧（共 {n_frames} 张）")


if __name__ == "__main__":
    main()
