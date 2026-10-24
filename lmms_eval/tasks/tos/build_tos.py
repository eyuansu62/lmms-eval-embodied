#!/usr/bin/env python3
"""Theory of Space（ToS）→ 主动探索多轮评测（实验性，未进入能力表）。

    python lmms_eval/tasks/tos/build_tos.py

三件事：
  1. 下载数据 MLL-Lab/Theory-of-Space（CC-BY-4.0，单个 zip 约 4.6 GB，100 个三房间场景，全部预渲染），
     校验 sha256 后解压到 media/。
  2. 取官方代码 mll-lab-nu/Theory-of-Space（MIT）固定提交到 vendor/Theory-of-Space：环境、出题、判分都用它。
  3. 官方代码另需 gymnasium、omegaconf，用 pip --target 装到 vendor/deps，不进主环境
     （omegaconf 会把 antlr4 升到 4.9，与 latex2sympy2 要求的 4.7.2 冲突；这里只装 omegaconf 本体，
     运行时沿用主环境的 antlr4，会打一行版本不一致的提示，不影响 ToS 用到的功能）。
"""

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import zipfile

from huggingface_hub import hf_hub_download

HERE = pathlib.Path(__file__).resolve().parent
REPO = "MLL-Lab/Theory-of-Space"
ZIP = "tos_dataset_0127_3room_100runs.zip"
ZIP_SHA256 = "195723ca597be86a489baaf187709e20a5ff49246d1ec7ceebfb244a444c4dd4"
CODE = "https://github.com/mll-lab-nu/Theory-of-Space.git"
COMMIT = "d48f52fb2c07d7a440a5ce76486d115cd85f37a3"
DEPS = ["gymnasium==1.2.3", "omegaconf==2.3.0"]


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def data() -> None:
    media = HERE / "media"
    if (media / "run99" / "meta_data.json").exists():
        return
    z = pathlib.Path(hf_hub_download(REPO, ZIP, repo_type="dataset"))
    if _sha256(z) != ZIP_SHA256:
        raise SystemExit(f"{ZIP}: sha256 不符")
    tmp = HERE / "_unzip"
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    shutil.rmtree(media, ignore_errors=True)
    (tmp / ZIP.removesuffix(".zip")).rename(media)
    shutil.rmtree(tmp)


def code() -> None:
    dst = HERE / "vendor" / "Theory-of-Space"
    if dst.exists():
        head = subprocess.run(["git", "-C", str(dst), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == COMMIT:
            return
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    subprocess.run(["git", "-C", str(dst), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(dst), "remote", "add", "origin", CODE], check=True)
    for _ in range(3):  # 仓库约 90 MB（含论文图），连接不稳时重试
        if subprocess.run(["git", "-C", str(dst), "fetch", "-q", "--depth", "1", "origin", COMMIT]).returncode == 0:
            break
    else:
        raise SystemExit(f"拉取 {CODE}@{COMMIT} 失败")
    subprocess.run(["git", "-C", str(dst), "checkout", "-q", "FETCH_HEAD"], check=True)


def deps() -> None:
    target = HERE / "vendor" / "deps"
    if (target / "omegaconf").exists() and (target / "gymnasium").exists():
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(target), *DEPS], check=True)


def main() -> None:
    data()
    code()
    deps()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "tos.jsonl").write_text("".join(json.dumps({"id": f"run{s:02d}", "seed": s}) + "\n" for s in range(100)))
    print("tos: 100 个场景")


if __name__ == "__main__":
    main()
