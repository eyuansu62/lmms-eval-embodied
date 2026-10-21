#!/usr/bin/env python3
"""AQuA 测试集 → 按等级均衡抽 150 题的本地 jsonl + 图片。

    python lmms_eval/tasks/aqua/build_aqua_jsonl.py [--src 已下载的 local_dir]

产物：
    lmms_eval/tasks/aqua/data/aqua_level{0,1,2,3}.jsonl   38/38/37/37 题（进 git，CC BY 4.0）
    lmms_eval/tasks/aqua/media/<filename>                 图片（COCO，不进 git，可由本脚本重建）

数据：HF jihyoung/AQuA 的 test split（3600 题，每级 900）。仓库以 datasets 的 arrow 分片存放
（test/data-0000{0..3}-of-00004.arrow，共约 1.7 GB），这里逐片下载后用 datasets 读。
#47 的请求预算：AQuA 150 题（MM-UPD 100 次请求、LiveVQA 150 题，合计约 400）。
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
REPO_ID = "jihyoung/AQuA"
SEED = 20260926
PER_LEVEL = {0: 38, 1: 38, 2: 37, 3: 37}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="已用 hf_hub_download(local_dir=...) 下好的目录，省掉再下 1.7 GB")
    args = ap.parse_args()
    import datasets
    from huggingface_hub import hf_hub_download

    names = [f"test/data-0000{i}-of-00004.arrow" for i in range(4)]
    if args.src:
        paths = [str(pathlib.Path(args.src) / n) for n in names]
    else:
        paths = [hf_hub_download(REPO_ID, n, repo_type="dataset") for n in names]
    ds = datasets.concatenate_datasets([datasets.Dataset.from_file(p) for p in paths])
    levels = [int(x) for x in ds["level"]]
    assert len(ds) == 3600, len(ds)

    (HERE / "media").mkdir(parents=True, exist_ok=True)
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    for lv, n in PER_LEVEL.items():
        idx = [i for i, l in enumerate(levels) if l == lv]
        picked = sorted(random.Random(f"{SEED}:aqua:{lv}").sample(idx, n))
        with open(HERE / "data" / f"aqua_level{lv}.jsonl", "w", encoding="utf-8") as fh:
            for i in picked:
                r = ds[i]
                img = r["image"]
                name = f"{i:04d}_{pathlib.Path(str(r['filename'])).stem}.jpg"
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=95)
                (HERE / "media" / name).write_bytes(buf.getvalue())
                fh.write(json.dumps({
                    "id": f"aqua_{i}",
                    "test_index": i,
                    "level": lv,
                    "question": str(r["question"]),
                    "answer": str(r["answer"]),
                    "filename": str(r["filename"]),
                    "image": name,
                }, ensure_ascii=False) + "\n")
        print(f"aqua_level{lv}: {n} 题")


if __name__ == "__main__":
    main()
