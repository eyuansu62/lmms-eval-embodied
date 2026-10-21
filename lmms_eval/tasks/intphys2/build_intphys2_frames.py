#!/usr/bin/env python3
"""IntPhys 2 · permanence / continuity → 本地 jsonl + 帧图（#35 / #38）。

    python lmms_eval/tasks/intphys2/build_intphys2_frames.py --src /tmp/intphys2_dl

产物（帧图不进 git，可由此脚本重建；jsonl 进 git）：
    lmms_eval/tasks/intphys2/data/intphys2_{permanence,continuity}.jsonl
    lmms_eval/tasks/intphys2/media/<name>/<i>.jpg

数据：HF facebook/IntPhys2（CC BY-NC 4.0，仅限评测用途），Main 集 1012 个视频 / 253 个场景，
每个场景 4 个视频（2 可能 + 2 不可能）。四个 condition 各取所需：
    permanence   240  → #35 物体恒常性
    continuity   240  → #38 运动连续性
    immutability / solidity 不收。
--src 目录结构同 HF 仓库：<src>/Main/metadata.csv + <src>/Main/Videos/*.mp4。

帧：照官方 IntPhys2_openai.py 的 process_video(seconds_per_frame=1)[:50] —— 从第 0 帧起
每 int(fps) 帧取一张，最多 50 张。视频是 10.65 s / 60 fps，所以每个视频 11 张 512×512。
官方用 cv2.imencode 默认 JPEG 质量（95）；这里同样存 q95。
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
CONDITIONS = ("permanence", "continuity")
SECONDS_PER_FRAME = 1
MAX_FRAMES = 50


def extract(video: pathlib.Path, out_dir: pathlib.Path) -> list[str]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(cap.get(cv2.CAP_PROP_FPS) * SECONDS_PER_FRAME)
    assert total > 0 and step > 0, (video, total, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    rels, cur = [], 0
    while cur < total - 1 and len(rels) < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        ok, frame = cap.read()
        if not ok:
            break
        name = f"{len(rels):02d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rels.append(f"{out_dir.name}/{name}")
        cur += step
    cap.release()
    return rels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    src = pathlib.Path(args.src) / "Main"

    with open(src / "metadata.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1012, len(rows)

    for cond in CONDITIONS:
        xs = sorted((r for r in rows if r["condition"] == cond),
                    key=lambda r: (int(r["SceneIndex"]), r["game_name"], r["type"]))
        assert len(xs) == 240, (cond, len(xs))
        out = HERE / "data" / f"intphys2_{cond}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        n_frames = []
        with open(out, "w", encoding="utf-8") as fh:
            for r in xs:
                frames = extract(src / r["file_name"], HERE / "media" / r["name"])
                n_frames.append(len(frames))
                fh.write(json.dumps({
                    "id": r["name"],
                    # 同一场景的 4 个视频 = (SceneIndex, game_name)；成对指标按这个分组
                    "scene": f"{r['game_name']}#{r['SceneIndex']}",
                    "condition": cond,
                    "type": r["type"],                       # 1_Possible / 2_Impossible …
                    "label": "Impossible" if "Impossible" in r["type"] else "Possible",
                    "difficulty": r["Difficulty"],
                    "camera": r["Camera"],
                    "env": r["env"],
                    "frames": frames,
                }, ensure_ascii=False) + "\n")
        print(f"intphys2_{cond}: {len(xs)} 个视频，每个 {min(n_frames)}–{max(n_frames)} 帧")


if __name__ == "__main__":
    main()
