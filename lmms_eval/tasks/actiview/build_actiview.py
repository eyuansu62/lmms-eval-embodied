#!/usr/bin/env python3
"""ActiView（wangphoebe/ActiView，Apache-2.0）→ 主动感知多轮评测（实验性，未进入能力表）。

    python lmms_eval/tasks/actiview/build_actiview.py

325 题全取。每题用到 5 张图：全图（images_split/<id>_resized.jpg）和四个切片
（左上、左下、右上、右下，已由作者放大到全图尺寸）。原图、带分割线的图、bbox 图评测不用，不解压。
groundtruth_views.json 是含线索的切片编号（1–4，同上顺序）；shifting_level.json 是三档难度的
初始切片下标（0–3）。
"""

import json
import pathlib
import zipfile

from huggingface_hub import hf_hub_download

HERE = pathlib.Path(__file__).resolve().parent
REPO = "wangphoebe/ActiView"
MEDIA = HERE / "media"
LETTERS = "ABCDEFG"


def _dl(name: str) -> pathlib.Path:
    return pathlib.Path(hf_hub_download(REPO, name, repo_type="dataset"))


def main() -> None:
    items = json.loads(_dl("data.json").read_text())["data"]
    gt_views = json.loads(_dl("groundtruth_views.json").read_text())
    levels = json.loads(_dl("shifting_level.json").read_text())
    assert len(items) == len(gt_views) == 325 and all(len(v) == 325 for v in levels.values())

    rows, wanted = [], set()
    for i, it in enumerate(items):
        paths = [p.removeprefix("./") for p in it["images"]]
        full, views = paths[1], paths[3:7]
        wanted.update([full, *views])
        rows.append({
            "id": it["id"], "category": it["category"], "distraction": it["distraction"],
            "question": it["question"], "letters": list(LETTERS[: len(it["options"])]),
            "candidates": it["options"], "answer": LETTERS[it["answer"]],
            "full": full.removeprefix("images_split/"), "views": [v.removeprefix("images_split/") for v in views],
            "gt_views": gt_views[i], "init": {k: levels[k][i] for k in ("easy", "medium", "hard")},
        })

    MEDIA.mkdir(exist_ok=True)
    with zipfile.ZipFile(_dl("Archive_imgs.zip")) as z:
        for name in sorted(wanted):
            dst = MEDIA / name.removeprefix("images_split/")
            if not dst.exists():
                dst.write_bytes(z.read(name))

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "actiview.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"actiview: {len(rows)} 题，{len(wanted)} 张图")


if __name__ == "__main__":
    main()
