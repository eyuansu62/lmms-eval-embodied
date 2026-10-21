#!/usr/bin/env python3
"""Build JSONL slices for RoboBench from HF dataset.

Usage:
    conda activate lmms-eval
    python build_robobench_jsonl.py               # 规划题 + 视觉属性 + system prompt
    python build_robobench_jsonl.py --prefetch    # 同上，并把这些题的图片预取到 media/
    python build_robobench_jsonl.py --mcq         # 重建 data/_full/ 下 7 个选择题全量

选择题的 data/<name>.jsonl 是从 data/_full/ 按子任务取前 N 得到的采样版（A/B 档
每个 benchmark 控制在 800 题左右），本脚本**不会**覆盖它们。早先的版本直接把全量
写进 data/，跑一次就会把采样版冲掉。

按能力表（能力体系（CAPABILITIES.md））对应的切片：
    #2  客体可供性        static_affordance / dynamic_affordance（选择题，已有）
    #4  客体视觉属性      static_attribute 中 material / transparency → robobench_visual_attribute
    #8  相对位置          spatial_relation（已有）
    #12 时间指代定位      timestamp_analysis（已有）
    #14 指令具化          implicit_demand_goal 的 Q1 → robobench_implicit_goal_q1
                          （robobench_implicit_goal 是 Q1/Q2/Q3 全量，Q2/Q3 不测具化，不再挂 #14）
    #25 失败定位          high_level_planning_error / low_level_execution_error（已有）
    #28 子任务分解        3_generalized_planning 下 unique_id 以 _Q1 结尾的题，
                          逐原生子任务一个 task → robobench_plan_<子任务>（50 题）/
                          robobench_plan20_<子任务>（20 题，#28 挂这个）；排除 navigation
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import random
import shutil

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FULL_DIR = DATA_DIR / "_full"
MEDIA_DIR = HERE / "media"
HF_LOCAL = "/tmp/robobench_hf"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ID = "LeoFan01/RoboBench"

# Each slice: (output_name, hf_path, capability_tag)
# MCQ slices (have 'options' field):
MCQ_SLICES = [
    ("robobench_static_affordance",
     "4_affordance_reasoning/static_affordance/questions.json",
     "#2 客体可供性感知"),
    ("robobench_dynamic_affordance",
     "4_affordance_reasoning/dynamic_affordance/questions.json",
     "#2 客体可供性感知（动态）"),
    ("robobench_static_attribute",
     "2_perception_reasoning/object_centric/static_attribute/questions.json",
     "#4 客体视觉属性"),
    ("robobench_spatial_relation",
     "2_perception_reasoning/scene_centric/spatial_centric/spatial_relation/questions.json",
     "#8 相对位置"),
    ("robobench_timestamp_analysis",
     "2_perception_reasoning/scene_centric/temporal_centric/timestamp_analysis/questions.json",
     "#12 时间指代定位"),
    ("robobench_high_level_error",
     "5_error_analysis/high_level_planning_error/questions.json",
     "#25 失败定位"),
    ("robobench_low_level_error",
     "5_error_analysis/low_level_execution_error/questions.json",
     "#25 失败定位（执行层）"),
]

# #4：static_attribute 里只有这两类是「视觉属性」。其余（deformability / fragility /
# mass / is_sealed / can_contain_liquid / contents）是功能或物理属性，占 85%，
# 能力表明确写了「不能整组代表」。
VISUAL_ATTRIBUTES = ("material", "transparency")

# 开放式（规划）切片：(task 名, HF 目录, 只取 Q1)
# 前三个是旧切片，保持全量、题序不变（3B 基线的 doc_id 依赖题序）。
OPENENDED_SLICES = [
    ("robobench_implicit_goal", "1_instruction_comprehension/implicit_demand_goal", False),
    ("robobench_explicit_goal", "1_instruction_comprehension/explicit_object_goal", False),
    ("robobench_number", "3_generalized_planning/cross_object/physical_attribute/number", False),
    # #14 指令具化只用 Q1（隐含需求 → 多步规划）。Q2 是看执行视频预测下一步、Q3 是判断
    # 某步是否完成，测的是执行跟踪而非具化。2026-10-20 起 #14 挂这个切片。
    ("robobench_implicit_goal_q1", "1_instruction_comprehension/implicit_demand_goal", True),
]

# #28 子任务分解：上游 3_generalized_planning 的 17 个原生子任务，去掉 navigation
# （没有 _Q1 后缀、也没有 DAG）。task 名取目录末段。
PLAN_SUBTASKS = [
    "cross_embodiment/single_arm",
    "cross_embodiment/dual_arm",
    "cross_embodiment/human",
    "cross_embodiment/mobile_manipulation",
    "cross_object/material_affordance/articulated",
    "cross_object/material_affordance/deformable",
    "cross_object/material_affordance/multi_object",
    "cross_object/material_affordance/rigid",
    "cross_object/material_affordance/special",
    "cross_object/physical_attribute/color",
    "cross_object/physical_attribute/number",
    "cross_object/physical_attribute/shape",
    "cross_object/physical_attribute/size",
    "cross_object/world_knowledge",
    "cross_view/multi_view",
    "cross_view/single_view",
]
# 每个子任务取 50 题 Q1，16 个子任务约 791 题（world_knowledge 只有 41 题，全取），
# 与 A/B 档每个 benchmark 约 800 题的规模一致。题目在子任务内按数据来源成块排列
# （ARIO / DSL / RDT / rtx ...），直接取前 N 会只切到一两个来源，所以先打乱再取，
# 保留原始相对顺序。
PLAN_PER_SUBTASK = 50
PLAN_SEED = 42
# 2026-10-20 起 #28 挂每子任务 20 题的版本（16 × 20 = 320 题，95% 半宽约 ±3.2 分；50 题版 ±2.0）。
# 取同一打乱顺序的前 20 个，所以是 50 题版的**子集**：已有的 50 题回复可以直接复用。
PLAN20_PER_SUBTASK = 20


def plan_task_name(subdir: str) -> str:
    return "robobench_plan_" + subdir.rsplit("/", 1)[-1]


def plan20_task_name(subdir: str) -> str:
    return "robobench_plan20_" + subdir.rsplit("/", 1)[-1]


def _download(filename: str) -> pathlib.Path:
    return pathlib.Path(hf_hub_download(
        REPO_ID, repo_type="dataset", filename=filename, local_dir=HF_LOCAL,
    ))


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_mcq_slice(name: str, hf_path: str) -> int:
    data = json.loads(_download(hf_path).read_text())

    rows = []
    for item in data:
        if "options" not in item:
            continue
        opts = item["options"]
        # Map positional list to A/B/C/D/E
        letter_opts = {chr(65 + i): o for i, o in enumerate(opts)}
        rows.append({
            "id": item["unique_id"],
            "question": item["question"],
            "options": letter_opts,
            "answer": item["gt_answer"],
            "image_paths": item.get("image_urls", []),
            "hf_path_prefix": str(pathlib.Path(item["unique_id"]).parent),
            "input_type": item.get("input_type", "image"),
        })
    _write_jsonl(FULL_DIR / f"{name}.jsonl", rows)
    print(f"  {name}: {len(rows)} rows → _full/{name}.jsonl")
    return len(rows)


def build_visual_attribute() -> list[dict]:
    """#4：static_attribute 全量里筛 material / transparency，共 31 题，不采样。"""
    src = FULL_DIR / "robobench_static_attribute.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r["id"].split("/images/", 1)[1].split("/", 1)[0] in VISUAL_ATTRIBUTES]
    _write_jsonl(DATA_DIR / "robobench_visual_attribute.jsonl", keep)
    print(f"  robobench_visual_attribute: {len(keep)} rows（{'/'.join(VISUAL_ATTRIBUTES)}）")
    return keep


def _qtype(unique_id: str) -> str:
    """Q1 多步规划 / Q2 下一步 / Q3 状态判断。上游评测器也按后缀分类。"""
    for q in ("Q1", "Q2", "Q3"):
        if unique_id.endswith("_" + q):
            return q
    raise ValueError(f"unique_id 没有 _Q1/_Q2/_Q3 后缀: {unique_id}")


_ANSWER_TYPE = {"Q1": "multistep", "Q2": "singlestep", "Q3": "yesno"}


def _openended_rows(subdir: str, q1_only: bool) -> list[dict]:
    questions = json.loads(_download(f"{subdir}/questions.json").read_text())
    dags = {d["id"]: d["gt_dag"] for d in json.loads(_download(f"{subdir}/dag.json").read_text())}
    rows = []
    for item in questions:
        uid = item["unique_id"]
        q = _qtype(uid)
        if q1_only and q != "Q1":
            continue
        row = {
            "id": uid,
            "question": item["question"],
            "answer": item["gt_answer"],
            "qtype": q,
            # 旧字段，保留给读旧样本的脚本
            "answer_type": _ANSWER_TYPE[q],
            "image_paths": item["image_urls"],
            "input_type": item.get("input_type", "image"),
            # 官方 prompt 需要这两个字段：本体类型决定 "You are a ... robot."，
            # 子任务决定是否追加 color/number/shape/size 的说明、multi_view 的视角名
            "robotic_type": item["robotic_type"],
            "subtask": item["task_type"][-1],
        }
        if q == "Q1":
            # Q1 的 judge 要用 DAG。直接嵌进题目，judge 不必再去拉 dag.json。
            row["dag"] = dags[uid[: -len("_Q1")]]
        rows.append(row)
    return rows


def build_openended() -> list[dict]:
    all_rows = []
    for name, subdir, q1_only in OPENENDED_SLICES:
        rows = _openended_rows(subdir, q1_only)
        old = DATA_DIR / f"{name}.jsonl"
        if old.exists():  # 题序必须与旧文件一致，否则旧样本的 doc_id 全部错位
            old_ids = [json.loads(l)["id"] for l in old.read_text().splitlines() if l.strip()]
            if old_ids != [r["id"] for r in rows]:
                raise SystemExit(f"{name}: 新生成的题序与 {old.name} 不一致，拒绝覆盖")
        _write_jsonl(old, rows)
        print(f"  {name}: {len(rows)} rows")
        all_rows += rows

    for subdir in PLAN_SUBTASKS:
        name = plan_task_name(subdir)
        rows = _openended_rows(f"3_generalized_planning/{subdir}", q1_only=True)
        _write_jsonl(FULL_DIR / f"{name}.jsonl", rows)
        idx = list(range(len(rows)))
        random.Random(PLAN_SEED).shuffle(idx)
        keep = sorted(idx[:PLAN_PER_SUBTASK])
        sampled = [rows[i] for i in keep]
        _write_jsonl(DATA_DIR / f"{name}.jsonl", sampled)
        print(f"  {name}: {len(sampled)} / {len(rows)} Q1")
        all_rows += sampled
        sampled20 = [rows[i] for i in sorted(idx[:PLAN20_PER_SUBTASK])]
        _write_jsonl(DATA_DIR / f"{plan20_task_name(subdir)}.jsonl", sampled20)
        print(f"  {plan20_task_name(subdir)}: {len(sampled20)} / {len(rows)} Q1（{name} 的子集）")
    return all_rows


def prefetch(rows: list[dict], workers: int = 16) -> None:
    """并发预取图片。串行按需下载太慢（ERIQ 实测串行 9.7h vs 并发 6.5min）。"""
    from utils import _get_image  # noqa: E402  同目录

    jobs = [(r["id"], f) for r in rows for f in r["image_paths"]]
    failed = []

    def one(job):
        try:
            _get_image(*job)
        except Exception as exc:
            return job, exc
        return None

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(one, jobs), 1):
            if res:
                failed.append(res)
            if i % 500 == 0:
                print(f"    prefetch {i}/{len(jobs)}")
    print(f"  prefetch: {len(jobs) - len(failed)}/{len(jobs)} ok")
    for job, exc in failed[:20]:
        print(f"    FAIL {job}: {exc}")
    if failed:
        raise SystemExit(f"{len(failed)} 张图片下载失败")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq", action="store_true", help="重建 data/_full/ 下 7 个选择题全量")
    ap.add_argument("--prefetch", action="store_true", help="预取规划题与视觉属性题的图片")
    args = ap.parse_args()

    print(f"Building RoboBench JSONL slices → {DATA_DIR}")
    if args.mcq:
        for name, hf_path, _ in MCQ_SLICES:
            build_mcq_slice(name, hf_path)

    # 官方 prompt（skill list、各子任务说明、MCQ 后缀）
    shutil.copy(_download("system_prompt.json"), DATA_DIR / "system_prompt.json")
    print("  system_prompt.json")

    rows = build_visual_attribute() + build_openended()
    if args.prefetch:
        prefetch(rows)


if __name__ == "__main__":
    main()
