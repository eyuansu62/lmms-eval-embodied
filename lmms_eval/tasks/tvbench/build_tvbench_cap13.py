#!/usr/bin/env python3
"""TVBench（FunAILab/TVBench，CC BY 4.0）→ #13 事件顺序理解。

    python lmms_eval/tasks/tvbench/build_tvbench_cap13.py

只取两个直接问「谁先谁后」的子任务：
    action_sequence      437 题  二选一「这个人先做了什么？」（STAR / Charades 片段）
    egocentric_sequence  200 题  四选一「视频里的动作顺序是？」，选项是同一组动作的不同排列
TVBench 的设计目标就是单帧 / 纯文本答不出来（两个候选动作都在片段里发生过）。

为什么不用 lmms-eval 自带的 tvbench_* 任务（同目录 utils.py，勿改）：
  * action_sequence 每题带 start / end，问的是这一段里谁先 —— 自带任务喂整段视频、不裁剪，
    片段外可能又出现同样动作，题目就有歧义。这里用 ffmpeg 按 [start, end] 预先裁好；
  * 自带抽取取「回复里第一个独立大写字母」，"A person first ..." 会被判成选 A。判分见 utils_cap13.py。

采样：action_sequence 按正确答案位置（A / B）各取 100（seed 42），egocentric_sequence 200 全取，共 400。
⚠️ action_sequence 的视频源自 Charades（非商用许可），仅限评测。
产物：data/tvbench_cap13_*.jsonl 进 git；裁好的视频在 media/（不进 git，重跑本脚本可还原）。
"""

import json
import pathlib
import random
import subprocess
import zipfile


from huggingface_hub import HfApi, hf_hub_download

HERE = pathlib.Path(__file__).resolve().parent
REPO = "FunAILab/TVBench"
MEDIA = HERE / "media"
SEED = 42
PER_POSITION = 100
LETTERS = "ABCD"


def _dl(f: str) -> pathlib.Path:
    return pathlib.Path(hf_hub_download(REPO, f, repo_type="dataset"))


def _extract(zips: list[str], dest: pathlib.Path) -> dict[str, pathlib.Path]:
    """解压到 dest，返回 {zip 内相对路径: 本地路径}。已解压过的跳过。"""
    dest.mkdir(parents=True, exist_ok=True)
    found: dict[str, pathlib.Path] = {}
    for z in zips:
        with zipfile.ZipFile(_dl(z)) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                out = dest / name
                if not out.exists():
                    zf.extract(name, dest)
                found[name] = out
    return found


def _by_tail(files: dict[str, pathlib.Path], video: str) -> pathlib.Path:
    """json 里的 video 字段与 zip 内路径前缀不一定一致，按路径尾部匹配。"""
    hits = [p for k, p in files.items() if k.endswith(video)]
    if len(hits) != 1:
        raise SystemExit(f"{video}: zip 里匹配到 {len(hits)} 个文件")
    return hits[0]


def main() -> None:
    all_files = HfApi().list_repo_files(REPO, repo_type="dataset")

    # ── action_sequence ──
    rows = json.loads(_dl("json/action_sequence.json").read_text())
    files = _extract([f for f in all_files if f.startswith("video/action_sequence")], MEDIA / "_raw_action")
    rng = random.Random(SEED)
    by_pos = {0: [], 1: []}
    for r in rows:
        by_pos[r["candidates"].index(r["answer"])].append(r)
    picked = sorted(rng.sample(by_pos[0], PER_POSITION) + rng.sample(by_pos[1], PER_POSITION),
                    key=lambda r: str(r["question_id"]))
    clip_dir = MEDIA / "action_sequence"
    clip_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in picked:
        src = _by_tail(files, r["video"])
        clip = clip_dir / f"{r['question_id']}.mp4"
        if not clip.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(r["start"]), "-to", str(r["end"]),
                            "-i", str(src), "-c:v", "libx264", "-an", str(clip)], check=True)
        out.append({"id": str(r["question_id"]), "question": r["question"], "candidates": r["candidates"],
                    "answer": LETTERS[r["candidates"].index(r["answer"])],
                    "video": str(clip.relative_to(MEDIA)), "start": r["start"], "end": r["end"]})
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "tvbench_cap13_action_sequence.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"action_sequence: {len(rows)} → {len(out)}（A/B 各 {PER_POSITION}，按 start/end 裁剪）")

    # ── egocentric_sequence ──
    rows = json.loads(_dl("json/egocentric_sequence.json").read_text())
    files = _extract([f for f in all_files if f.startswith("video/egocentric_sequence")], MEDIA / "egocentric_sequence")
    out = []
    for i, r in enumerate(rows):
        src = _by_tail(files, r["video"])
        out.append({"id": f"ego{i}", "question": r["question"], "candidates": r["candidates"],
                    "answer": LETTERS[r["candidates"].index(r["answer"])],
                    "video": str(src.relative_to(MEDIA))})
    (HERE / "data" / "tvbench_cap13_egocentric_sequence.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"egocentric_sequence: {len(out)}（全量）")


if __name__ == "__main__":
    main()
