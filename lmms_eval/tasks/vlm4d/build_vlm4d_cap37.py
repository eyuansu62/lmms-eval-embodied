#!/usr/bin/env python3
"""VLM4D（shijiezhou/VLM4D，ICCV 2025，MIT）→ #37 轨迹理解与预测的「当前运动方向」部分。

    python lmms_eval/tasks/vlm4d/build_vlm4d_cap37.py

只用真实视频（QA/real_mc.json，1371 题，全是四选一）：
    ego4d        449 题  第一人称（「鸡蛋在碗的哪一侧磕开」「人把盖子往哪放」）
    davis        795 题  第三人称（「车往哪开」「火车有几节在往左走」）
    youtube-vos  127 题
不用合成视频（445 题，单物体平移，离具身场景远）。
VLM4D 测的是「物体当前怎么动」，基本不考预测（1816 题里约 29 题沾边），
所以 #37 的「预测」那半仍由 CausalPhys 碰撞题承担。

采样：ego4d 150 + 第三人称（davis + youtube-vos）150，seed 42，共 300。
剔除 1 道坏题（两个选项文字相同、答案对不回唯一字母）。
官方答案存的是选项文字，这里换成字母。
两部分分开出分：第三人称盲测 30.7%（随机 25）；ego4d 盲测 46.7% —— 多数题实为「左 / 右」二选一，
另两个选项（upwards / backward / straight）一眼可排除，读分按随机 50 对照。
产物：data/vlm4d_cap37_{ego,exo}.jsonl 进 git；视频在 media/（不进 git，重跑本脚本可还原）。
"""

import json
import pathlib
import random
import shutil


from huggingface_hub import hf_hub_download

HERE = pathlib.Path(__file__).resolve().parent
REPO = "shijiezhou/VLM4D"
MEDIA = HERE / "media"
SEED = 42
PER_GROUP = 150


def _norm(x) -> str:
    return str(x).strip().lower().rstrip(".")


def main() -> None:
    rows = json.loads(pathlib.Path(hf_hub_download(REPO, "QA/real_mc.json", repo_type="dataset")).read_text())
    groups: dict[str, list[dict]] = {"ego": [], "exo": []}
    dropped = 0
    for r in rows:
        letters = [k for k, v in r["choices"].items() if _norm(v) == _norm(r["answer"])]
        if len(letters) != 1:
            dropped += 1
            continue
        rel = r["video"].split("/resolve/main/", 1)[1]          # videos_real/<src>/<name>.mp4
        src = rel.split("/")[1]
        groups["ego" if src == "ego4d" else "exo"].append({
            "id": str(r["id"]), "question": r["question"],
            "candidates": [str(r["choices"][k]) for k in "ABCD"],
            "answer": letters[0], "source": src, "_rel": rel,
        })
    rng = random.Random(SEED)
    picked = []
    for g in ("ego", "exo"):
        picked += rng.sample(groups[g], PER_GROUP)
    picked.sort(key=lambda x: int(x["id"].rsplit("_", 1)[1]))

    out = []
    for x in picked:
        rel = x.pop("_rel")
        dst = MEDIA / rel.removeprefix("videos_real/")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download(REPO, rel, repo_type="dataset"), dst)
        x["video"] = str(dst.relative_to(MEDIA))
        out.append(x)
    (HERE / "data").mkdir(exist_ok=True)
    for name, keep in (("ego", lambda x: x["source"] == "ego4d"), ("exo", lambda x: x["source"] != "ego4d")):
        (HERE / "data" / f"vlm4d_cap37_{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out if keep(x)))
    print(f"real_mc: {len(rows)} 题，剔除坏题 {dropped}；ego4d {PER_GROUP} + 第三人称 {PER_GROUP} → {len(out)}")


if __name__ == "__main__":
    main()
