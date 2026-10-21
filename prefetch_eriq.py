#!/usr/bin/env python3
"""ERIQ 图片预取 —— 把评测实际要用的图片一次性拉到本地。

为什么必须有这一步
------------------
ERIQ 的图片是**按需逐张下载**的（见 `lmms_eval/tasks/eriq/utils.py` 的
`_fetch_one`）。而 lmms-eval 的 openai 后端在**主循环里串行**构造 payload
（`simple/openai.py` 的 `build_payload_for_index`），所以下载是串行的。

2026-10-12 实测：

    串行     10.7 s/张   → 3255 张 ≈ 9.7 小时
    并发 16   0.38 s/张  → 3255 张 ≈ 21 分钟

9.7 小时的空跑大部分会撞上网关空闲超时，等于跑不完。所以跑之前先预取。

预取哪些图
----------
**只取采样后实际会跑的那些**，不是整棵 images/ 树：
    整棵树 4.27 GB  vs  采样 800 题约 3253 张 ≈ 763 MB
采样规则与 `lmms_eval/tasks/eriq/sampling.py` **完全一致**（同种子），
所以这里收集的就是评测时会读的那批。

用法
----
    python prefetch_eriq.py                # 预取全部 8 个切片
    python prefetch_eriq.py --verify-only  # 只校验，不下载
    python prefetch_eriq.py --jobs 32      # 调并发（默认 16）
    python prefetch_eriq.py --tasks eriq_task_progress

图片落在 `lmms_eval/tasks/eriq/media/`（已被 .gitignore 覆盖）。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MIRROR = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
REPO_ID = "KineMind/ERIQ"
MEDIA_DIR = HERE / "lmms_eval" / "tasks" / "eriq" / "media"

# 子任务 -> 对应的 QA 文件（与 tasks/eriq/*.yaml 一致）
SLICES: dict[str, str] = {
    "eriq_human_intention": "QA_HUMAN_INTENTION",
    "eriq_mistake_classify": "QA_MISTAKE_CLASSIFY",
    "eriq_action_understanding": "QA_ACTION_UNDERSTANDING",
    "eriq_subtask_planning": "QA_SUBTASK_PLANNING",
    "eriq_task_progress": "QA_TASK_PROGRESS",
    "eriq_mistake_existence": "QA_MISTAKE_EXISTENCE",
    "eriq_success_detection": "QA_SUCCESS_DETECTION",
    "eriq_mistake_recovery": "QA_MISTAKE_RECOVERY",
}


def _collect_refs(tasks: list[str]) -> tuple[list[str], dict[str, int]]:
    """按 sampling.py 的同一套采样规则，收集实际会跑的图片相对路径。"""
    os.environ.setdefault("HF_ENDPOINT", MIRROR)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from datasets import load_dataset
    from lmms_eval.tasks.eriq import sampling as S

    refs: list[str] = []
    counts: dict[str, int] = {}
    for name in tasks:
        qa = SLICES[name]
        ds = load_dataset(REPO_ID, data_files={"test": f"{qa}.json"}, split="test")
        sampled = S.sample_to(ds, S.SAMPLE_SIZES[name])
        n_img = 0
        for doc in sampled:
            imgs = doc.get("image")
            if isinstance(imgs, str):
                imgs = [imgs]
            for rel in imgs or []:
                refs.append(str(rel).strip().lstrip("/"))
                n_img += 1
        counts[name] = len(sampled)
        print(f"  {name:26s} {len(sampled):4d} 题  {n_img:6d} 张图", flush=True)
    return refs, counts


def _local_path(rel: str) -> pathlib.Path:
    """与 utils._fetch_one 的落盘位置一致：media/images/<rel>。"""
    return MEDIA_DIR / "images" / rel


def _fetch(rel: str) -> tuple[str, bool, str]:
    from huggingface_hub import hf_hub_download

    dst = _local_path(rel)
    if dst.exists():
        return rel, True, "cached"
    try:
        hf_hub_download(
            REPO_ID, f"images/{rel}", repo_type="dataset", local_dir=str(MEDIA_DIR),
        )
        return rel, True, "ok"
    except Exception as exc:  # noqa: BLE001 - 单张失败不该中断整批
        return rel, False, f"{type(exc).__name__}: {str(exc)[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="ERIQ 图片预取（走 hf-mirror）")
    ap.add_argument("--tasks", nargs="*", choices=list(SLICES),
                    help="只预取指定切片（默认全部 8 个）")
    ap.add_argument("--jobs", type=int, default=16, help="并发数（默认 16）")
    ap.add_argument("--verify-only", action="store_true",
                    help="不下载，只校验本地是否齐全")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", MIRROR)
    names = args.tasks or list(SLICES)

    print(f"=== 收集要跑的题（镜像 {MIRROR}）===")
    t0 = time.time()
    refs, counts = _collect_refs(names)
    uniq = sorted(set(refs))
    print(f"  合计 {sum(counts.values())} 题 / {len(refs)} 次引用 / {len(uniq)} 张去重图"
          f"  （{time.time()-t0:.0f}s）")

    if args.verify_only:
        print()
        print("=== 离线校验 ===")
        missing = [r for r in uniq if not _local_path(r).exists()]
        if missing:
            print(f"  ✗ 缺 {len(missing)} / {len(uniq)} 张，先跑一次不带 --verify-only 的预取")
            for m in missing[:5]:
                print(f"      {m}")
            sys.exit(1)
        print(f"  ✓ {len(uniq)} 张图全部就位")
        sys.exit(0)

    print()
    print(f"=== 并发 {args.jobs} 下载 ===")
    t0 = time.time()
    ok = cached = 0
    fails: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_fetch, r): r for r in uniq}
        for i, fut in enumerate(as_completed(futs), 1):
            rel, good, how = fut.result()
            if good:
                ok += 1
                if how == "cached":
                    cached += 1
            else:
                fails.append((rel, how))
            if i % 200 == 0 or i == len(uniq):
                el = time.time() - t0
                print(f"  {i:5d}/{len(uniq)}  已用 {el:5.0f}s  "
                      f"（{el/max(i,1):.2f}s/张，命中缓存 {cached}）", flush=True)

    el = time.time() - t0
    print()
    print(f"完成：{ok}/{len(uniq)} 张（缓存命中 {cached}），耗时 {el:.0f}s")
    if fails:
        print(f"失败 {len(fails)} 张：")
        for rel, why in fails[:10]:
            print(f"   {rel}  {why}")
        sys.exit(1)

    # 落盘体积
    total = sum(f.stat().st_size for f in MEDIA_DIR.rglob("*") if f.is_file())
    print(f"本地占用：{total/1048576:.0f} MB")
    print()
    print("下一步：./run_model.sh a <模型> eriq")


if __name__ == "__main__":
    main()
