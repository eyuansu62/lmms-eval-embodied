#!/usr/bin/env python3
"""MV-RoboBench (AaronFengZY24/MV_Robobench) -> data/_full/mvrb_*.jsonl for four sub-tasks.

    python lmms_eval/tasks/mvrobobench/build_mvrb_jsonl.py

Rows keep every question of the sub-task from both domains (agiworld, bridgev2). The evaluated
subsets and their order are restored from data/MANIFEST.json by prepare_data.py.
Images are fetched lazily at evaluation time (see utils.py).
"""

import json
import pathlib

from huggingface_hub import snapshot_download

HERE = pathlib.Path(__file__).resolve().parent
REPO = "AaronFengZY24/MV_Robobench"
TASKS = {
    "affordance": "robotic_affordance_recognition",
    "cross_view": "spatial_cross_view_matching",
    "distance": "spatial_distance_judgement",
    "trajectory": "robotic_trajectory_selection",
}
DOMAINS = ("agiworld", "bridgev2")


def main() -> None:
    root = pathlib.Path(snapshot_download(
        REPO, repo_type="dataset",
        allow_patterns=[f"{d}/{t}/*/QA.json" for d in DOMAINS for t in TASKS.values()]))
    out_dir = HERE / "data" / "_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    for short, task in TASKS.items():
        rows = []
        for domain in DOMAINS:
            for qa in sorted((root / domain / task).glob("*/QA.json")):
                d = json.loads(qa.read_text(encoding="utf-8"))
                rows.append({
                    "id": f"{domain}/{task}/{qa.parent.name}", "domain": domain, "task": task,
                    "question": d["question"], "options": d["options"], "answer": d["correct"],
                    "type": d["type"], "n_views": len(d["images"]),
                })
        (out_dir / f"mvrb_{short}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"mvrb_{short}: {len(rows)}")


if __name__ == "__main__":
    main()
