#!/usr/bin/env python3
"""CoreCognition（williamium/CoreCognition，Apache-2.0，ICML 2025）→ #35 物体恒常性的两部分。

    python lmms_eval/tasks/corecognition/build_corecognition.py

用 frame-combined 配置：每题一张图（视频题已由官方合成多帧拼图），prompt 是官方原文
（结尾 "Please answer with 'YES' or 'NO' directly." 或选项字母）。不需要 6.4 GB 的完整 zip。

只取两个 concept：
    conservation  守恒（皮亚杰式：数量/长度/体积在变形后是否不变）→ #35 守恒部分
    permanence    恒常性（遮挡后物体是否仍在）                    → #35 恒常性第二来源
是非题偏 YES（conservation 72:32、permanence 31:19），全答 YES 即 69% / 62%，
所以是非题按 YES/NO **等量**采样（取少数类全量，seed 42）；选择题全取。
jsonl 进 git（Apache-2.0）；图片 media/ 不进（与其余任务一致，重跑本脚本即可还原）。
"""

import io
import json
import pathlib
import random


import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = "williamium/CoreCognition"
CONCEPTS = ("conservation", "permanence")
SEED = 42


def main() -> None:
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("frame-combined/") and f.endswith(".parquet"))
    pool = {c: [] for c in CONCEPTS}
    for f in files:
        for r in pq.read_table(hf_hub_download(REPO, f, repo_type="dataset")).to_pylist():
            if r["concept"] in pool:
                pool[r["concept"]].append(r)
    media = HERE / "media"
    media.mkdir(exist_ok=True)
    for concept, rows in pool.items():
        rows.sort(key=lambda r: int(r["id"]))
        yn = [r for r in rows if r["question_type"] == "YORN"]
        mc = [r for r in rows if r["question_type"] != "YORN"]
        yes = [r for r in yn if r["answer"].upper() == "YES"]
        no = [r for r in yn if r["answer"].upper() == "NO"]
        k = min(len(yes), len(no))
        rng = random.Random(SEED)
        picked = sorted(rng.sample(yes, k) + rng.sample(no, k) + mc, key=lambda r: int(r["id"]))
        out = HERE / "data" / f"corecognition_{concept}.jsonl"
        with out.open("w") as fh:
            for r in picked:
                im = r["images"]
                b = im["bytes"] if isinstance(im, dict) else im
                rel = f"{concept}_{r['id']}.jpg"
                Image.open(io.BytesIO(b)).convert("RGB").save(media / rel, quality=92)
                fh.write(json.dumps({"id": r["id"], "concept": concept, "question_type": r["question_type"],
                                     "prompt": r["prompt"], "answer": r["answer"].strip().upper(),
                                     "image": rel}, ensure_ascii=False) + "\n")
        print(f"{concept}: 全量 {len(rows)}（是非 {len(yn)}，YES {len(yes)} / NO {len(no)}；选择 {len(mc)}）"
              f" → {len(picked)}（是非 {2 * k} + 选择 {len(mc)}）")


if __name__ == "__main__":
    main()
