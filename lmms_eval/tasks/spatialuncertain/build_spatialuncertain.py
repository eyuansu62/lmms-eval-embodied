#!/usr/bin/env python3
"""SpatialUncertain（Yuezhangjoslin/spatialuncertain，CC BY 4.0，gated）→ #45 场景完整判断与处置。

    python lmms_eval/tasks/spatialuncertain/build_spatialuncertain.py

判断（能不能答）：occlusion 题里同一目标、同一问题的两份——清晰视角（应作答）与完全遮挡（应答 Cannot determine）。
  只取两份题干与选项文字完全一致的配对（1409 / 1719），不看图就无法区分两者；按配对计分，
  两份都对才算对，「一律拒答」「从不拒答」都拿不到分。题型 relative_position / depth_ordering /
  size_comparison 各 50 对，共 150 对（300 次请求），每题 1 张图（oracle 视角）。
处置（换视角）：view_selection 670 题，五个视角里选能判断真实大小的那个（正确项恒为等距中心视角）。
  按正确字母 A–E 各 40 题，共 200 题，每题 5 张图。

采样 seed 42；同一 (场景, 目标, 题型) 只取一对。图片按需下载到 media/（不进 git）。
需先在 Hugging Face 页面同意条款并登录（gated）。
"""

import json
import pathlib
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

HERE = pathlib.Path(__file__).resolve().parent
REPO = "Yuezhangjoslin/spatialuncertain"
MEDIA = HERE / "media"
SEED = 42
PAIRS_PER_TYPE = 50
VIEWSEL_PER_LETTER = 40
QTYPES = ("relative_position", "depth_ordering", "size_comparison")


def _dl(path: str) -> pathlib.Path:
    return pathlib.Path(hf_hub_download(REPO, path, repo_type="dataset"))


def _questions(name: str) -> dict:
    return json.loads(_dl(f"questions/{name}_benchmark.json").read_text())


class _Images:
    """题目里的路径是作者本机的绝对路径。本机的 <x>_layout 目录对应仓库 scenes/ 下的固定目录，
    其后的相对路径与仓库一致（不同目录下有同名的变体，不能只按路径末尾匹配）。"""

    LAYOUT = {"clean_scene_layout": "clean_scenes", "occlusion_scene_layout": "occlusion_scenes",
              "view_selection_scenes_layout": "view_selection_scenes", "distortion_scenes_layout": "perspective_scenes"}

    def __init__(self) -> None:
        self.files = {f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True)
                      if f.path.endswith(".png")}
        self.wanted: set[str] = set()

    def resolve(self, local_abs: str) -> str:
        parts = local_abs.split("/")
        i = next((k for k, x in enumerate(parts) if x in self.LAYOUT), None)
        if i is None:
            raise SystemExit(f"{local_abs}: 无法识别的本机目录")
        repo_path = "/".join(["scenes", self.LAYOUT[parts[i]], *parts[i + 1:]])
        if repo_path not in self.files:
            raise SystemExit(f"{local_abs}: 仓库里没有 {repo_path}")
        self.wanted.add(repo_path)
        return repo_path.removeprefix("scenes/")

    def download(self, workers: int = 16) -> None:
        def one(repo_path: str) -> None:
            dst = MEDIA / repo_path.removeprefix("scenes/")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(_dl(repo_path), dst)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, sorted(self.wanted)))


def _opts(q: dict) -> tuple[list[str], list[str]]:
    letters = [o.split(")", 1)[0].strip() for o in q["answer_options"]]
    texts = [o.split(")", 1)[1].strip() for o in q["answer_options"]]
    return letters, texts


def build_pairs(img: _Images, rng: random.Random) -> tuple[list[dict], list[dict]]:
    pool: dict[str, dict[tuple, tuple]] = {t: {} for t in QTYPES}
    for s in _questions("occlusion")["scenes"]:
        for t in s["targets"]:
            clean = {q["question_type"]: q for q in t["clean"]["questions"]}
            for oc in t["occlusions"]:
                if oc["occlusion_type"] != "full_occlusion":
                    continue
                for q in oc["questions"]:
                    c = clean.get(q["question_type"])
                    if q["question_type"] not in QTYPES or c is None or c["question"] != q["question"]:
                        continue
                    if sorted(_opts(c)[1]) != sorted(_opts(q)[1]):
                        continue
                    key = (s["scene_id"], t["target_id"], q["question_type"])
                    pool[q["question_type"]].setdefault(key, (c, q, t, oc))
    ans, abst = [], []
    for qt in QTYPES:
        keys = sorted(pool[qt])
        for key in sorted(rng.sample(keys, PAIRS_PER_TYPE)):
            c, q, t, oc = pool[qt][key]
            pair_id = f"{key[0]}|{key[1]}|{qt}"
            for out, item, frame in ((ans, c, t["clean"]), (abst, q, oc)):
                letters, texts = _opts(item)
                out.append({
                    "id": item["question_id"], "pair_id": pair_id, "question_type": qt,
                    "condition": item["scene_type"], "question": item["question"],
                    "letters": letters, "candidates": texts, "answer": item["gt_answer"],
                    "images": [img.resolve(frame["image_retrieval"]["oracle_frame"]["image_path"])],
                })
    return ans, abst


def build_viewsel(img: _Images, rng: random.Random) -> list[dict]:
    by_letter: dict[str, list[dict]] = defaultdict(list)
    for e in _questions("view_selection")["entries"]:
        by_letter[e["gt_answer"]].append(e)
    out = []
    for letter in "ABCDE":
        for e in sorted(rng.sample(by_letter[letter], VIEWSEL_PER_LETTER), key=lambda e: e["entry_id"]):
            letters, texts = _opts(e)
            views = sorted(e["candidate_views"], key=lambda c: c["label"])
            out.append({
                "id": e["entry_id"], "pair_category": e["pair_category"], "question": e["question"],
                "letters": letters, "candidates": texts, "answer": e["gt_answer"],
                "images": [img.resolve(v["image_path"]) for v in views],
            })
    return sorted(out, key=lambda x: x["id"])


def main() -> None:
    rng = random.Random(SEED)
    img = _Images()
    ans, abst = build_pairs(img, rng)
    viewsel = build_viewsel(img, rng)
    img.download()
    (HERE / "data").mkdir(exist_ok=True)
    for name, rows in (("occ_answerable", ans), ("occ_abstain", abst), ("view_select", viewsel)):
        (HERE / "data" / f"spatialuncertain_{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"spatialuncertain_{name}: {len(rows)}")


if __name__ == "__main__":
    main()
