#!/usr/bin/env python3
"""
C 档数据集预取 —— 先把 8 个 benchmark 的数据拉到本地，之后评测可以完全离线。

    python prefetch_c_tier.py                 # 预取全部（已缓存的会跳过）
    python prefetch_c_tier.py --tasks epic phyx
    python prefetch_c_tier.py --verify-only   # 只做离线校验，不下载

为什么要单独跑这一步：
  A/B 档的媒体已经本地化在 tasks/*/media/ 下，但 C 档是**运行时从 HF 拉**的。
  本机连不上 huggingface.co（Errno 101），只通 hf-mirror.com。直接开评测的话，
  C 档会卡在 HEAD 重试上，几十题之后整个 benchmark 被拖死，而且退出码可能是 0。

  预取一次之后每个任务都命中本地缓存，离线可跑，评测时也不再有下载抖动。

注意 robo_spatial：yaml 用 data_files 通配 + verification_mode=no_checks 拼出
`combined` 配置。datasets 5.0.1 在**从缓存重载**时会把 verification_mode 丢掉再算
config_id，于是离线去找 `default-49b60927b8cdc9c1`，而在线写盘的是
`default-d4dc482102d358f1`，对不上就报 "Couldn't find cache"。
数据是同一份，只是目录名不同 —— 本脚本会补一个软链把两者接上（见 _link_alias）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

MIRROR = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# benchmark -> 加载参数。必须和对应 yaml 完全一致，否则缓存的 config_id 对不上，
# 评测时不会复用这里的下载结果。
BENCHMARKS: dict[str, dict] = {
    "crosspoint": {
        "repo": "WangYipu2002/CrossPoint-Bench", "split": "train",
        "rows": 1000, "note": "另有 image/ 树需 snapshot_download",
    },
    "multihopspatial": {
        "repo": "etri-vilab/MultihopSpatial", "name": "benchmark", "split": "test",
        "rows": 4500,
    },
    "omnispatial": {
        "repo": "nv-njb/OmniSpatial-Test", "split": "test", "rows": 1533,
    },
    "robospatial": {
        "repo": "chanhee-luke/RoboSpatial-Home", "split": "combined", "rows": 350,
        "data_files": {"combined": [
            "data/compatibility-*.parquet",
            "data/configuration-*.parquet",
            "data/context-*.parquet",
        ]},
        "verification_mode": "no_checks",
    },
    "sparbench": {
        "repo": "jasonzhango/SPAR-Bench", "split": "test", "rows": 7211,
    },
    "embspatial": {
        "repo": "FlagEval/EmbSpatial-Bench", "split": "test", "rows": 3640,
    },
    "phyx": {
        "repo": "Cloudriver/PhyX", "name": "data_for_llms_eval", "split": "phyx_mc",
        "rows": 3000,
    },
    "epic": {
        "repo": "rxc205/EPIC-Bench", "split": "train", "rows": 6661,
    },
}


def _cached(ds):
    """True if this load came from disk rather than the network."""
    return getattr(getattr(ds, "info", None), "download_checksums", None) is not None


def prefetch(name: str, spec: dict, offline: bool = False) -> tuple[str, int]:
    """Load one benchmark so datasets/hub 把文件落到本地缓存。返回 (状态, 行数)。"""
    from datasets import load_dataset

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.setdefault("HF_ENDPOINT", MIRROR)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    kwargs = {"split": spec["split"]}
    if spec.get("name"):
        kwargs["name"] = spec["name"]
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]
    if spec.get("verification_mode"):
        kwargs["verification_mode"] = spec["verification_mode"]

    ds = load_dataset(spec["repo"], **kwargs)

    # CrossPoint 的图不在 parquet 里，是仓库里的 image/ 树，得单独拉
    if name == "crosspoint":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=spec["repo"], repo_type="dataset",
                          allow_patterns=["image/**", "CrossPoint-Bench.jsonl"])

    return len(ds)


def _link_alias() -> None:
    """补 robo_spatial 的缓存软链（见文件头说明）。幂等。"""
    import glob
    root = os.path.expanduser("~/.cache/huggingface/datasets/chanhee-luke___robo_spatial-home")
    if not os.path.isdir(root):
        return
    # datasets 从缓存重载时算出来的 config_id（verification_mode 丢了）
    want = "default-49b60927b8cdc9c1"
    if os.path.exists(os.path.join(root, want)):
        return
    # 在线写盘时用的 config_id
    for cand in ("default-d4dc482102d358f1", "default"):
        if os.path.isdir(os.path.join(root, cand)):
            src = cand
            break
    else:
        return
    os.symlink(src, os.path.join(root, want))
    print(f"    ↳ 补软链 {want} -> {src}（datasets 5.0.1 离线重算 config_id 的偏差）")


VERIFY_SNIPPET = r"""
import sys
from datasets import load_dataset
ok = True
for name, repo, cfg, split, fail in [
    ("crosspoint",      "WangYipu2002/CrossPoint-Bench", None, "train", None),
    ("multihopspatial", "etri-vilab/MultihopSpatial", "benchmark", "test", None),
    ("omnispatial",     "nv-njb/OmniSpatial-Test", None, "test", None),
    ("sparbench",       "jasonzhango/SPAR-Bench", None, "test", None),
    ("embspatial",      "FlagEval/EmbSpatial-Bench", None, "test", None),
    ("phyx",            "Cloudriver/PhyX", "data_for_llms_eval", "phyx_mc", None),
    ("epic",            "rxc205/EPIC-Bench", None, "train", None),
]:
    try:
        ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
        print(f"OK   {name:16s} rows={len(ds):6d}")
    except Exception as e:
        ok = False
        print(f"FAIL {name:16s} {type(e).__name__}: {str(e)[:120]}")
try:
    ds = load_dataset("chanhee-luke/RoboSpatial-Home",
        data_files={"combined": ["data/compatibility-*.parquet",
                                 "data/configuration-*.parquet",
                                 "data/context-*.parquet"]},
        verification_mode="no_checks", split="combined")
    print(f"OK   {'robospatial':16s} rows={len(ds):6d}")
except Exception as e:
    ok = False
    print(f"FAIL {'robospatial':16s} {type(e).__name__}: {str(e)[:120]}")
try:
    from lmms_eval.tasks.crosspoint_bench.utils import _image_root
    import os as _os
    n = sum(len(f) for _, _, f in _os.walk(_image_root()))
    print(f"OK   {'crosspoint img':16s} files={n}")
except Exception as e:
    ok = False
    print(f"FAIL {'crosspoint img':16s} {type(e).__name__}: {str(e)[:120]}")
sys.exit(0 if ok else 1)
"""


def verify_offline() -> bool:
    """在强制离线的子进程里把 8 个 benchmark 重载一遍。"""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "HF_ENDPOINT": MIRROR, "HF_HUB_DISABLE_XET": "1"}
    proc = subprocess.run([sys.executable, "-c", VERIFY_SNIPPET],
                          env=env, cwd=HERE, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith(("OK ", "FAIL")):
            print("   " + line)
    return proc.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="C 档数据集预取（走 hf-mirror）")
    ap.add_argument("--tasks", nargs="*", choices=list(BENCHMARKS),
                    help="只预取指定 benchmark（默认全部）")
    ap.add_argument("--verify-only", action="store_true",
                    help="不下载，只在离线模式下校验缓存是否完整")
    args = ap.parse_args()

    names = args.tasks or list(BENCHMARKS)

    if not args.verify_only:
        print(f"=== C 档预取（镜像 {MIRROR}）===")
        t0 = time.time()
        for name in names:
            spec = BENCHMARKS[name]
            print(f"  {name:16s} {spec['repo']}", flush=True)
            started = time.time()
            try:
                n = prefetch(name, spec)
            except Exception as e:
                print(f"    ✗ {type(e).__name__}: {str(e)[:160]}", flush=True)
                continue
            print(f"    ✓ {n} 行  {time.time()-started:.0f}s"
                  + ("" if n == spec.get("rows") else f"  (预期 {spec.get('rows')} 行，请核对)"),
                  flush=True)
            if name == "robospatial":
                _link_alias()
        print(f"  合计 {time.time()-t0:.0f}s\n")

    print("=== 离线校验（HF_HUB_OFFLINE=1）===")
    ok = verify_offline()
    print()
    if ok:
        print("全部离线可用 ✓ —— 之后 C 档评测不再需要网络。")
    else:
        print("仍有 benchmark 离线不可用 ✗ —— 见上方 FAIL。")
        sys.exit(1)


if __name__ == "__main__":
    main()
