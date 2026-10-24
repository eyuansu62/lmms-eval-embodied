#!/usr/bin/env python3
"""Download the original benchmark data and rebuild the exact evaluation subsets.

    python prepare_data.py                 # all tasks
    python prepare_data.py tvbench vlm4d   # selected tasks
    python prepare_data.py --list

Each task: fetch the original files (Hugging Face / GitHub / PyPI), run the task's builder
scripts, then select and verify the evaluated rows against ``lmms_eval/tasks/<task>/data/MANIFEST.json``
(see ``tools/data_manifest.py``). Media files are written next to each task and are not tracked by git.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ROOT / "lmms_eval" / "tasks"
CACHE = pathlib.Path.home() / ".cache" / "lmms_eval_prepare"
PY = sys.executable


def hf(repo: str, filename: str) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    return pathlib.Path(hf_hub_download(repo, filename, repo_type="dataset"))


def hf_snapshot(repo: str, patterns: list[str]) -> pathlib.Path:
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(repo, repo_type="dataset", allow_patterns=patterns))


def fetch_url(url: str, dst: pathlib.Path, tries: int = 5) -> pathlib.Path:
    import urllib.request

    if dst.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                tmp.write_bytes(r.read())
            tmp.replace(dst)
            return dst
        except OSError as e:
            print(f"  retry {i + 1}/{tries} {url}: {e}")
    raise SystemExit(f"download failed: {url}")


def run(task: str, script: str, *args: str) -> None:
    subprocess.run([PY, str(TASKS / task / script), *map(str, args)], check=True, cwd=ROOT)


def link(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir() and not dst.is_symlink():
        shutil.rmtree(dst)
    elif dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def restore(task: str) -> bool:
    r = subprocess.run([PY, str(ROOT / "tools" / "data_manifest.py"), "restore", str(TASKS / task)], cwd=ROOT)
    return r.returncode == 0


def prep_physbench() -> None:
    link(hf("lmms-lab-eval/PhysBench", "data/full-00000-of-00001.parquet"),
         TASKS / "physbench" / "data" / "full-00000-of-00001.parquet")
    run("physbench", "build_physbench_slices.py")


def prep_vsibench() -> None:
    src = hf("nyu-visionx/VSI-Bench", "test.jsonl")
    link(src, TASKS / "vsibench" / "data" / "test.jsonl")
    run("vsibench", "build_vsibench_slices.py", "--input", src)


def prep_egotaskqa() -> None:
    src = hf("eyuansu71/EgoTaskQA", "data/qa/direct/tagged_qas.json")
    link(src, TASKS / "egotaskqa" / "data" / "tagged_qas.json")
    run("egotaskqa", "build_egotaskqa_slices.py", "--input", src)
    restore("egotaskqa")
    run("egotaskqa", "build_egotaskqa_state_change.py")


def prep_negvqa() -> None:
    src = hf("yuhuizhang/NegVQA", "data/dev-00000-of-00001.parquet")
    link(src, TASKS / "negvqa" / "data" / "dev-00000-of-00001.parquet")
    run("negvqa", "screen_negvqa.py", "--input", src, "--output", TASKS / "negvqa" / "data" / "negvqa_natural_scene_dev.jsonl")


def prep_pio() -> None:
    import os

    src = fetch_url("https://raw.githubusercontent.com/xavihart/PIO/main/data/s1s2.json", CACHE / "pio" / "s1s2.json")
    os.environ["PIO_JSON"] = str(src)
    run("pio", "build_pio_jsonl.py")
    run("pio", "build_pio_ordinal.py", "--src", src)


def prep_causalphys() -> None:
    sys.path.insert(0, str(TASKS / "causalphys"))
    from build_causalphys_jsonl import SUBSETS

    root = hf_snapshot("haorentang/causalphys", [f"{d}/annotations/*.json" for d, _ in SUBSETS.values()])
    for repo_dir, local_dir in SUBSETS.values():
        link(root / repo_dir / "annotations", TASKS / "causalphys" / "data" / local_dir)
    run("causalphys", "build_causalphys_jsonl.py")
    run("causalphys", "build_attribution.py")
    run("causalphys", "build_multihop.py")


def prep_eai() -> None:
    import importlib.util
    import zipfile

    wheels = CACHE / "eai"
    if not list(wheels.glob("eai_eval-1.0.5-*.whl")):
        subprocess.run([PY, "-m", "pip", "download", "eai-eval==1.0.5", "--no-deps", "-d", str(wheels)], check=True)
    whl = next(wheels.glob("eai_eval-1.0.5-*.whl"))
    src = CACHE / "eai" / "pkg"
    if not src.is_dir():
        zipfile.ZipFile(whl).extractall(src)
    if importlib.util.find_spec("virtualhome_eval") is None:
        subprocess.run([PY, "-m", "pip", "install", "--no-deps", str(whl)], check=True)
    run("eai", "build_eai_jsonl.py", "--src", src)
    run("eai", "build_eai_virtualhome.py", src)
    run("eai", "build_eai_action_seq.py")


def prep_asimov() -> None:
    for script in ("build_asimov_jsonl.py", "build_asimov_safety.py", "build_asimov_hsm.py"):
        run("asimov", script)


def prep_intphys2() -> None:
    src = hf_snapshot("facebook/IntPhys2", ["Main/metadata.csv", "Main/Videos/*"])
    run("intphys2", "build_intphys2_frames.py", "--src", src)


def prep_openeqa() -> None:
    src = hf_snapshot("thomas-yanxin/open-eqa-mirror", ["data/*.parquet"])
    run("openeqa_scene", "build_openeqa_jsonl.py", "--src", src)


STEPS = {
    "aqua": lambda: run("aqua", "build_aqua_jsonl.py"),
    "corecognition": lambda: run("corecognition", "build_corecognition.py"),
    "mmupd_cap47": lambda: run("mmupd_cap47", "build_mmupd_jsonl.py"),
    "refspatial": lambda: run("refspatial", "build_refspatial_slices.py"),
    "tvbench": lambda: run("tvbench", "build_tvbench_cap13.py"),
    "vlm4d": lambda: run("vlm4d", "build_vlm4d_cap37.py"),
    "physbench": prep_physbench,
    "vsibench": prep_vsibench,
    "egotaskqa": prep_egotaskqa,
    "negvqa": prep_negvqa,
    "pio": prep_pio,
    "robobench": lambda: run("robobench", "build_robobench_jsonl.py", "--mcq"),
    "mvrobobench": lambda: run("mvrobobench", "build_mvrb_jsonl.py"),
    "causalphys": prep_causalphys,
    "eai": prep_eai,
    "asimov": prep_asimov,
    "spatialuncertain": lambda: run("spatialuncertain", "build_spatialuncertain.py"),
    "actiview": lambda: run("actiview", "build_actiview.py"),
    "tos": lambda: run("tos", "build_tos.py"),
    "intphys2": prep_intphys2,
    "openeqa_scene": prep_openeqa,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(STEPS))
        return
    todo = a.tasks or list(STEPS)
    unknown = [t for t in todo if t not in STEPS]
    if unknown:
        sys.exit(f"unknown task(s): {unknown}; see --list")
    failed = []
    for t in todo:
        print(f"\n=== {t}")
        STEPS[t]()
        if not restore(t):
            failed.append(t)
    if failed:
        sys.exit(f"\nverification failed: {failed}")
    print("\nall selected tasks rebuilt and verified")


if __name__ == "__main__":
    main()
