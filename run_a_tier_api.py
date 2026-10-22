#!/usr/bin/env python3
"""
A/B/C 档全量评测脚本 —— 任意 OpenAI 兼容 API 模型版。

从 .env 读取 API 配置，可通过命令行参数完整覆盖，无需修改文件。
结果写入 eval_A_<bench>_<suffix>/ 、eval_B_<bench>_<suffix>/ 或 eval_C_<bench>_<suffix>/。

用法：
    # 用 .env 里的默认配置跑全部 A 档
    python run_a_tier_api.py --skip-existing

    # 换模型和 key，结果目录自动带上 suffix 以示区分
    python run_a_tier_api.py \\
        --model gpt-4o \\
        --base-url https://api.openai.com/v1 \\
        --api-key sk-xxx \\
        --output-dir-suffix gpt4o

    # 只跑部分 benchmark，调整并发
    python run_a_tier_api.py --tasks causalphys vsibench --num-concurrent 4

    # C 档（8 个，约 27.9k 题；题量是 A 档 6 倍以上，建议单跑）
    python run_a_tier_api.py --tasks omni_spatial embspatial --skip-existing
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


FAIL_MARK = "[LMMS_EVAL_REQUEST_FAILED"


def is_empty_resp(r: dict) -> bool:
    """模型回复为空（不是失败标记）。见 retry_empty 的说明。"""
    return str(r.get("filtered_resps", "")).strip() in ("", "[]", "['']", '[""]')


def _latest_samples(out_dir: pathlib.Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    for f in sorted(glob.glob(str(out_dir / "**" / "*_samples_*.jsonl"), recursive=True)):
        latest[f.split("_samples_")[-1][: -len(".jsonl")]] = f  # sorted -> newest timestamp wins
    return latest


def collect_failed(out_dir: pathlib.Path, include_empty: bool = False) -> dict[str, list[int]]:
    """{task: [doc_id, ...]} of requests that exhausted retries in the latest samples per task.

    include_empty=True 时把空回复也算进来（--retry-empty 用）。
    """
    failed: dict[str, list[int]] = {}
    for task, f in _latest_samples(out_dir).items():
        with open(f, errors="replace") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        ids = [r["doc_id"] for r in rows
               if FAIL_MARK in str(r.get("filtered_resps", "")) or (include_empty and is_empty_resp(r))]
        if ids:
            failed[task] = sorted(ids)
    return failed


def merge_fix(src: pathlib.Path, fix: pathlib.Path, merged: pathlib.Path) -> None:
    """把 fix 里补到的题合并进 src，写到 merged（src 可以就是 merged，会先备份）。

    补跑结果仍为空时保留原行；补跑成了失败标记时用失败标记替换（剔失败口径下不计分，
    比当作 0 分更接近真实）。两边同一 doc_id 的题面必须一致，否则整份 task 放弃合并 ——
    防止 doc_id 口径变了（如 C 档采样 / 全量）时把别的题的答案拼进来。
    """
    import shutil
    import time as _time
    src_files, fix_files = _latest_samples(src), _latest_samples(fix)
    if src.resolve() == merged.resolve():
        bak = merged.with_name(f"{merged.name}.bak-{_time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copytree(merged, bak)
        print(f"  旧的 {merged.name} 已备份到 {bak.name}")
    for task, f in sorted(src_files.items()):
        with open(f, errors="replace") as fh:
            rows = {r["doc_id"]: r for r in (json.loads(l) for l in fh if l.strip())}
        n_fixed = n_still = 0
        if task in fix_files:
            with open(fix_files[task], errors="replace") as fh:
                redo = {r["doc_id"]: r for r in (json.loads(l) for l in fh if l.strip())}
            key = "doc_hash" if all("doc_hash" in r for r in redo.values()) else "input"
            bad = [d for d, r in redo.items() if d in rows and str(rows[d].get(key)) != str(r.get(key))]
            if bad:
                print(f"  ✗ {task}: {len(bad)} 题的题面与原结果不一致（按 {key} 比），不合并该 task")
                redo = {}
            for d, r in redo.items():
                if is_empty_resp(r):
                    n_still += 1
                    continue
                rows[d] = r
                n_fixed += 1
        dest = merged / pathlib.Path(f).parent.name / pathlib.Path(f).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as fh:
            for d in sorted(rows):
                fh.write(json.dumps(rows[d], ensure_ascii=False) + "\n")
        left = sum(is_empty_resp(r) for r in rows.values())
        print(f"  {task:40s} 补回 {n_fixed:4d}  仍为空 {n_still:4d}  合并后空回复 {left}")


def load_env() -> dict[str, str]:
    env_file = HERE / ".env"
    if not env_file.exists():
        raise SystemExit(f"找不到 {env_file}，请先创建并填入 API key")
    cfg: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip()
    return cfg


BENCHMARKS = {
    "causalphys": {
        "tasks": ",".join([
            "causalphys_action_substitution",
            "causalphys_collision_prediction",
            "causalphys_containability",
            "causalphys_deformation",
            "causalphys_fluid_flow",
            "causalphys_mechanics_reasoning",
            "causalphys_multi_hop",
            "causalphys_spatial_manipulation",
            "causalphys_temporal_shifting",
        ]),
        "out": "eval_A_causalphys_qwen3max",
        "log": "/tmp/eval_cp_qwen3max.log",
    },
    "physbench": {
        "tasks": ",".join([
            "physbench_property_object",
            "physbench_single_body_dynamics",
            "physbench_multi_body_collision",
            "physbench_fluid_prediction",
            "physbench_fluid_perception",
        ]),
        "out": "eval_A_physbench_qwen3max",
        "log": "/tmp/eval_pb_qwen3max.log",
    },
    "vsibench": {
        "tasks": "vsibench_rel_direction,vsibench_rel_distance,vsibench_size_estimation",
        "out": "eval_A_vsibench_qwen3max",
        "log": "/tmp/eval_vsi_qwen3max.log",
        # 帧数必须在这里传，yaml 传不进来。
        # `_default_template_yaml` 里的 `num_frames: 32` 挂在
        # `lmms_eval_specific_kwargs.neo_ov` 段下，而本套件用的模型名是 `openai`，
        # 匹配不上任何一段，那份配置从没被读到（死配置）。
        # chat/simple 后端实际用的是 `simple/openai.py` 的 `max_frames_num` 默认值 10。
        # 实测 10→32：rel_direction 0.5437→0.7282，McNemar p<0.0001；32 是上游
        # 官方默认值，我们一直在跑偏离 3.2 倍的帧数。egotaskqa 加帧无收益（p=1.000）、
        # physbench 视频中位仅 1s，所以**只给 vsibench 加，别推广**。
        "model_args_extra": "max_frames_num=32",
    },
    "momagraph": {
        "tasks": "momagraph_effect_prediction,momagraph_action_sequence,momagraph_spatial_reasoning",
        "out": "eval_A_momagraph_qwen3max",
        "log": "/tmp/eval_mg_qwen3max.log",
    },
    "negvqa": {
        "tasks": "negvqa_natural_scene_original,negvqa_natural_scene_negated",
        "out": "eval_A_negvqa_qwen3max",
        "log": "/tmp/eval_nv_qwen3max.log",
    },
    "refspatial": {
        "tasks": "refspatial_attribute_reference,refspatial_ordinal_reference",
        "out": "eval_A_refspatial_qwen3max",
        "log": "/tmp/eval_rs_qwen3max.log",
    },
    "egotaskqa": {
        "tasks": "egotaskqa_event_order,egotaskqa_counterfactual,egotaskqa_explanatory",
        "out": "eval_A_egotaskqa_qwen3max",
        "log": "/tmp/eval_ego_qwen3max.log",
    },
    # #27 单体/多体状态理解的主测（能力体系：EgoTaskQA Descriptive 对象状态/状态变化题），
    # 2026-10-20 加。200 题，全是 yes/no。输出目录与 egotaskqa 相同 —— build_report.load()
    # 按任务名取最新时间戳，补跑不会覆盖另外三个切片。
    "egotaskqa_state": {
        "tasks": "egotaskqa_state_change",
        "out": "eval_A_egotaskqa_qwen3max",
        "log": "/tmp/eval_egostate_qwen3max.log",
    },
    # ERIQ 一个 benchmark 占 能力体系 的 7 个能力行（#14/#25/#27/#28/#32/#33/#34），
    # 其中 #32 进度估计**只有它一个来源**（无兜底）。8 个子任务按 A 档
    # 「切法 A」等比分配到 800 题（各 50~197），分派表见 lmms_eval/tasks/eriq/sampling.py。
    "eriq": {
        "tasks": ",".join([
            "eriq_human_intention",      # #14 指令具化
            "eriq_mistake_classify",     # #25 失败定位与原因分析
            "eriq_action_understanding", # #27 单体/多体状态理解
            "eriq_subtask_planning",     # #28 子任务分解
            "eriq_task_progress",        # #32 进度估计
            "eriq_mistake_existence",    # #33 失败检测（Yes/No 题）
            "eriq_success_detection",    # #33 失败检测
            "eriq_mistake_recovery",     # #34 恢复策略
        ]),
        "out": "eval_A_eriq_qwen3max",
        "log": "/tmp/eval_eriq_qwen3max.log",
    },
    # B 档
    # RoboBench 选择题：按能力表取子集。2026-10-14 起：
    #   * static_attribute 换成 visual_attribute（只留 material/transparency，#4）——
    #     整组里 85% 是可变形性/易碎性/质量，不是视觉属性；
    #   * prompt 改成官方前缀（utils.doc_to_text），yaml version 0.1 → 0.2。
    # 之前的 eval_B_robobench_* 是旧口径（含 static_attribute、裸题干），不能与新结果混比。
    "robobench": {
        "tasks": ",".join([
            "robobench_static_affordance",   # #2
            "robobench_dynamic_affordance",  # #2
            "robobench_visual_attribute",    # #4
            "robobench_spatial_relation",    # #8
            "robobench_timestamp_analysis",  # #12
            "robobench_high_level_error",    # #25
            "robobench_low_level_error",     # #25
        ]),
        "out": "eval_B_robobench_qwen3max",
        "log": "/tmp/eval_rb_qwen3max.log",
    },
    # #4 客体视觉属性的单独入口：visual_attribute 切片 2026-10-14 才建出来，那之前跑过的
    # 9 个模型都没有这份任务（它们只有 static_attribute 228 题，与这 31 题只重合 17 题）。
    # 输出目录与 robobench 相同 —— build_report.load() 按任务名取**最新时间戳**，所以补出来的
    # visual_attribute 会和历史的那几个任务并存，不会覆盖。3B 基线的 .response_cache 不共享。
    "robobench_attr": {
        "tasks": "robobench_visual_attribute",
        "out": "eval_B_robobench_qwen3max",
        "log": "/tmp/eval_rbattr_qwen3max.log",
    },
    # RoboBench 规划题（开放式）：#14 指令具化 + #28 子任务分解（16 个原生子任务的 Q1）。
    # Q1/Q2 跑完是 -1（待判），必须再跑 judge：./run_model.sh judge <模型>
    "robobench_plan": {
        "tasks": ",".join(
            ["robobench_implicit_goal"]
            + [f"robobench_plan_{s}" for s in (
                "single_arm", "dual_arm", "human", "mobile_manipulation",
                "articulated", "deformable", "multi_object", "rigid", "special",
                "color", "number", "shape", "size", "world_knowledge",
                "multi_view", "single_view",
            )]
        ),
        "out": "eval_B_robobench_plan_qwen3max",
        "log": "/tmp/eval_rbplan_qwen3max.log",
    },
    # #14 指令具化的单独入口：只跑 implicit_demand_goal 的 Q1（100 题）。2026-10-20 起 #14 只挂
    # 这一份（原挂的 ERIQ human_intention 有选项捷径，Q2/Q3 不测具化）。输出目录与 robobench_plan
    # 相同 —— build_report.load() 按任务名取最新时间戳，补跑不会覆盖历史的规划题。
    # 跑完必须判分：./run_model.sh judge <模型> robobench_implicit_goal_q1
    "robobench_q1": {
        "tasks": "robobench_implicit_goal_q1",
        "out": "eval_B_robobench_plan_qwen3max",
        "log": "/tmp/eval_rbq1_qwen3max.log",
    },
    # #28 子任务分解的单独入口：16 个规划子任务各 20 题（共 320 题），是 robobench_plan_* 50 题版的
    # 子集。2026-10-20 起 #28 只挂这一份（原挂的 ERIQ subtask_planning 是单步判断，不是分解，
    # 且选择题 67% 正确项是最长项）。输出目录与 robobench_plan 相同。
    # 跑完必须判分：./run_model.sh judge <模型> robobench_plan20_<子任务> ...
    "robobench_plan20": {
        "tasks": ",".join(f"robobench_plan20_{s}" for s in (
            "single_arm", "dual_arm", "human", "mobile_manipulation",
            "articulated", "deformable", "multi_object", "rigid", "special",
            "color", "number", "shape", "size", "world_knowledge",
            "multi_view", "single_view",
        )),
        "out": "eval_B_robobench_plan_qwen3max",
        "log": "/tmp/eval_rbplan20_qwen3max.log",
    },
    # ASIMOV 安全监控的单独入口（#36）：safety_tool_call 2026-10-19 才接入，
    # 之前的 asimov 跑没有这个子任务。输出目录与 asimov 相同 —— build_report.load()
    # 按任务名取最新时间戳，补跑不会覆盖历史的那 5 个任务。
    "asimov_safety": {
        "tasks": "asimov_safety",
        "out": "eval_B_asimov_qwen3max",
        "log": "/tmp/eval_asafety_qwen3max.log",
    },
    # #36 安全监控的视频帧主测：ASIMOV human_safety_monitoring 400 题（每题 8 张双目图），2026-10-20 加。
    # 输出目录与 asimov 相同；数据 gated，先跑 lmms_eval/tasks/asimov/build_asimov_hsm.py。
    "asimov_hsm": {
        "tasks": "asimov_hsm",
        "out": "eval_B_asimov_qwen3max",
        "log": "/tmp/eval_ahsm_qwen3max.log",
    },
    "mvrobobench": {
        "tasks": "mvrb_affordance,mvrb_distance,mvrb_trajectory,mvrb_cross_view",
        "out": "eval_B_mvrobobench_qwen3max",
        "log": "/tmp/eval_mvrb_qwen3max.log",
    },
    "pio": {
        "tasks": "pio_affordance,pio_attribute_reference,pio_safety",
        "out": "eval_B_pio_qwen3max",
        "log": "/tmp/eval_pio_qwen3max.log",
    },
    # ASIMOV（google/asimov_agentic，gated）：4 个能力行，全量 1599 题、每题 1 张小图。
    # 先跑 lmms_eval/tasks/asimov/build_asimov_jsonl.py 生成本地数据（不进 git）。
    # 决策题是文本化口径（官方为 tool-call），不能与论文分数直接比较。
    "asimov": {
        "tasks": ",".join([
            "asimov_vla_feasibility",   # #1 本体能力可供性感知
            "asimov_constraints",       # #3 本体客体交互关系
            "asimov_uncertainty",       # #15 指令歧义识别
            "asimov_gauge_readable",    # #46 低视觉质量判断（与下一项等权平均）
            "asimov_gauge_unreadable",  # #46
            "asimov_safety",            # #36 安全监控（与 pio_safety 两源）
        ]),
        "out": "eval_B_asimov_qwen3max",
        "log": "/tmp/eval_asimov_qwen3max.log",
    },
    # EAI（Embodied Agent Interface）目标解释：#16 目标理解，BEHAVIOR 100 + VirtualHome 338 题，纯文本（官方原协议）。
    # 主分 eai_f1 是官方 micro F1；报表里是逐题平均（eai_f1_macro）。
    "eai": {
        "tasks": "eai_behavior_goal,eai_virtualhome_goal",
        "out": "eval_B_eai_qwen3max",
        "log": "/tmp/eval_eai_qwen3max.log",
    },
    # #31 约束满足检查：EAI VirtualHome 动作序列 305 题，纯文本，判分现场跑官方模拟器
    # （lmms-eval 环境需装 eai-eval 1.0.5）。2026-10-20 加；输出目录与 eai 相同。
    "eai_as": {
        "tasks": "eai_action_seq",
        "out": "eval_B_eai_qwen3max",
        "log": "/tmp/eval_eaias_qwen3max.log",
    },
    # MM-UPD（#47 自我检视）：50 道原题 × 正常/无解两版 = 100 次请求，只问主版本（非循环评测）。
    # 主分是双重准确率（两版都对才算对），由 build_report.DUAL_TASKS 按 pair_id 配对。
    "mmupd": {
        "tasks": "mmupd_aad,mmupd_iasd,mmupd_ivqd",
        "out": "eval_B_mmupd_qwen3max",
        "log": "/tmp/eval_mmupd_qwen3max.log",
    },
    # AQuA（#47 询问必要信息）：150 题，按歧义等级 38/38/37/37 拆成 4 个子任务。
    # 判分在 process_results 里调内部路由的 qwen3.8-max 当 judge（结果缓存在 .response_cache/aqua_judge.sqlite）。
    "aqua": {
        "tasks": "aqua_level0,aqua_level1,aqua_level2,aqua_level3",
        "out": "eval_B_aqua_qwen3max",
        "log": "/tmp/eval_aqua_qwen3max.log",
    },
    # OpenEQA（#6 场景语义）：两类共 150 题，每题 16 帧图。判分在 process_results 里调 qwen3.8-max。
    "openeqa": {
        "tasks": "openeqa_world_knowledge,openeqa_functional_reasoning",
        "out": "eval_B_openeqa_qwen3max",
        "log": "/tmp/eval_openeqa_qwen3max.log",
    },
    # IntPhys 2（#35 物体恒常性 / #38 运动连续性）：各 240 个视频，每个 11–15 帧（1 帧/秒）。
    # 官方 prompt1 + Yes/No 判分；纯本地判分，无 judge。
    "intphys2": {
        "tasks": "intphys2_permanence,intphys2_continuity",
        "out": "eval_B_intphys2_qwen3max",
        "log": "/tmp/eval_intphys2_qwen3max.log",
    },
    # CoreCognition（#35 的守恒部分 + 恒常性第二来源），2026-10-20 加。每题一张图，官方 prompt。
    "corecognition": {
        "tasks": "corecognition_conservation,corecognition_permanence",
        "out": "eval_B_corecognition_qwen3max",
        "log": "/tmp/eval_corecog_qwen3max.log",
    },
    # TVBench（#13 事件顺序理解），2026-10-21 加，替换 EgoTaskQA event_order（那个切片里没有顺序题）。
    # action_sequence 200（按 start/end 预裁片段，二选一谁先）+ egocentric_sequence 200（四选一动作排列）。
    # 16 帧 = TVBench 论文的评测帧数；egocentric 片段 15–45s、7 步左右，默认 10 帧会漏步骤。
    "tvbench_cap13": {
        "tasks": "tvbench_cap13_action_sequence,tvbench_cap13_egocentric_sequence",
        "out": "eval_B_tvbench_cap13_qwen3max",
        "log": "/tmp/eval_tvbench13_qwen3max.log",
        "model_args_extra": "max_frames_num=16",
    },
    # VLM4D（#37 轨迹理解与预测的「当前运动方向」部分），2026-10-21 加。真实视频 300 题，分 ego / exo 两切片。
    # 视频多为 2–10s，帧数照 TVBench 取 16。
    "vlm4d_cap37": {
        "tasks": "vlm4d_cap37_ego,vlm4d_cap37_exo",
        "out": "eval_B_vlm4d_cap37_qwen3max",
        "log": "/tmp/eval_vlm4d37_qwen3max.log",
        "model_args_extra": "max_frames_num=16",
    },
    # SpatialUncertain（实验性，未进能力表；对应 #45 场景完整判断与处置）。判断：清晰 / 完全遮挡配对 150 对；
    # 处置：五个视角选一 200 题（每题 5 张图）。
    "spatialuncertain": {
        "tasks": "spatialuncertain_occ_answerable,spatialuncertain_occ_abstain,spatialuncertain_view_select",
        "out": "eval_B_spatialuncertain_qwen3max",
        "log": "/tmp/eval_su_qwen3max.log",
    },
    # ActiView（主动感知多轮，实验性，未进入能力表），2026-10-22 加。全图对照 + 放大两轮 + 三档移动多轮。
    "actiview": {
        "tasks": "actiview_full,actiview_zoom,actiview_shift_easy,actiview_shift_medium,actiview_shift_hard",
        "out": "eval_B_actiview_qwen3max",
        "log": "/tmp/eval_av_qwen3max.log",
    },
    # C 档 —— 键名刻意与 C 档结果目录的词干一致
    # （eval_C_<键>_<后缀>），这样 build_report.py 拼目录名时不用额外做映射。
    # 注意 omnispatial / robospatial 是连写的，和任务名 omni_spatial / robo_spatial_all 不同。
    "crosspoint": {
        "tasks": "crosspoint_bench",
        "out": "eval_C_crosspoint_qwen3max",
        "log": "/tmp/eval_Cx_crosspoint_qwen3max.log",
    },
    "multihopspatial": {
        "tasks": "multihopspatial",
        "out": "eval_C_multihopspatial_qwen3max",
        "log": "/tmp/eval_Cx_multihopspatial_qwen3max.log",
    },
    "omnispatial": {
        "tasks": "omni_spatial",
        "out": "eval_C_omnispatial_qwen3max",
        "log": "/tmp/eval_Cx_omnispatial_qwen3max.log",
    },
    "robospatial": {
        "tasks": "robo_spatial_all",
        "out": "eval_C_robospatial_qwen3max",
        "log": "/tmp/eval_Cx_robospatial_qwen3max.log",
    },
    "sparbench": {
        "tasks": "sparbench",
        "out": "eval_C_sparbench_qwen3max",
        "log": "/tmp/eval_Cx_sparbench_qwen3max.log",
    },
    "embspatial": {
        "tasks": "embspatial",
        "out": "eval_C_embspatial_qwen3max",
        "log": "/tmp/eval_Cx_embspatial_qwen3max.log",
    },
    "phyx": {
        # 只跑 MCQ：phyx_oe 是开放式物理推导，评分走 LLM judge，与 3B 基线口径不同
        "tasks": "phyx_mc",
        "out": "eval_C_phyx_qwen3max",
        "log": "/tmp/eval_Cx_phyx_qwen3max.log",
    },
    "epic": {
        "tasks": "epic_bench",
        "out": "eval_C_epic_qwen3max",
        "log": "/tmp/eval_Cx_epic_qwen3max.log",
    },
    # EPIC 朝向题补跑（#10 朝向和姿态）。C 档采样按 EPIC 的 3 个大类各取 267 题，朝向题所在的
    # TargetLocalization 有 4410 题，只抽到 20 道朝向题。这里关掉采样（doc_id = 原始下标），
    # 只跑没抽到的 356 道，与采样集那 20 道合并成全部 376 道（合并在 c_tier_subscores.py）。
    # 不进 build_report 的 C_BENCH：它只是 #10 的补充，不改 EPIC 整体分数的均衡口径。
    "epic_orient": {
        "tasks": "epic_bench",
        "out": "eval_C_epicorient_qwen3max",
        "log": "/tmp/eval_Cx_epicorient_qwen3max.log",
        "env": {
            "LMMS_C_TIER_SAMPLE_FULL": "1",
            "LMMS_DOC_ID_FILTER": str(HERE / "lmms_eval/tasks/epic_bench/orientation_supplement_ids.json"),
        },
        # doc_id 换成了原始下标，与采样 run 的紧凑下标同名不同题；不碰缓存最稳
        "no_cache": True,
    },
}


def _is_done(out_dir: pathlib.Path) -> bool:
    """Return True if a completed results file already exists in out_dir."""
    if not out_dir.exists():
        return False
    # lmms-eval writes <timestamp>_results.json under a model subdir
    return bool(list(out_dir.rglob("*results.json")))


def run_benchmark(name: str, cfg: dict[str, str], bench: dict[str, str], dry: bool, skip_existing: bool = False, cache_dir: str | None = None, retry_failed: bool = False, limit: int | None = None, retry_empty: bool = False) -> int:
    model = cfg.get("EVAL_ROUTER_MODEL", "qwen/qwen3.8-max-0902")
    base_url = cfg.get("EVAL_ROUTER_BASE_URL")
    if not base_url:
        raise SystemExit("EVAL_ROUTER_BASE_URL is not set (see .env.example)")
    api_key = cfg.get("EVAL_ROUTER_API_KEY", "")
    timeout = cfg.get("_TIMEOUT", "600")
    num_concurrent = cfg.get("_NUM_CONCURRENT", "8")
    max_size_in_mb = cfg.get("_MAX_SIZE_IN_MB", "2")

    model_args = (
        f"model={model},"
        f"base_url={base_url},"
        f"api_key={api_key},"
        f"timeout={timeout},"
        f"max_size_in_mb={max_size_in_mb},"
        f"num_concurrent={num_concurrent},"
        # 流式：网关空闲 60s 即 504，thinking 模型非流式会被截断
        f"stream=true"
    )
    # 每个 benchmark 可追加自己的 model_args（如 vsibench 的 max_frames_num=32）。
    # 放在最后，确保能覆盖上面的通用项。
    if bench.get("model_args_extra"):
        model_args += f",{bench['model_args_extra']}"

    cmd = [
        sys.executable, "-m", "lmms_eval",
        "--model", "openai",
        "--model_args", model_args,
        "--tasks", bench["tasks"],
        "--batch_size", "1",
        "--output_path", str(HERE / bench["out"]),
        "--log_samples",
    ]
    if cache_dir and not bench.get("no_cache"):
        cmd += ["--use_cache", str(pathlib.Path(cache_dir) / name)]
    if limit:
        cmd += ["--limit", str(limit)]

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = base_url
    # 网关对大请求体返回 400（实测失败的最小为 6.42MB）；超过 6MB 的单个请求改用 JPEG，其余保持 PNG
    env.setdefault("LMMS_MAX_REQUEST_MB", cfg.get("_MAX_REQUEST_MB", "6"))
    # A/B 档的媒体已本地化，但 C 档是运行时从 HF 拉的（CrossPoint 的 image/ 目录、
    # 各 benchmark 的 parquet 首次都要下载）。本机连不上 huggingface.co，必须走镜像，
    # 否则会卡在 HEAD 重试上把整个任务拖死。已设过的环境变量优先。
    if cfg.get("HF_ENDPOINT"):
        env.setdefault("HF_ENDPOINT", cfg["HF_ENDPOINT"])
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    # 个别 benchmark 自带环境变量（如 epic_orient 关采样 + 按下标过滤）
    env.update(bench.get("env", {}))

    if retry_failed:
        failed = collect_failed(HERE / bench["out"])
        n = sum(len(v) for v in failed.values())
        print(f"\n[{name}] 补跑失败题：{n} 题  {dict((k, len(v)) for k, v in failed.items())}")
        if not n:
            return 0
        filter_path = HERE / f"{bench['out']}_retry_filter.json"
        filter_path.write_text(json.dumps(failed, indent=1))
        env["LMMS_DOC_ID_FILTER"] = str(filter_path)
        cmd[cmd.index("--tasks") + 1] = ",".join(failed)
        cmd[cmd.index("--output_path") + 1] = str(HERE / f"{bench['out']}_retry")
        if "--use_cache" in cmd:  # 不复用缓存，确保真的重新请求
            i = cmd.index("--use_cache"); del cmd[i:i + 2]
        bench = {**bench, "out": f"{bench['out']}_retry", "log": bench["log"].replace(".log", "_retry.log")}
        skip_existing = False

    # --retry-empty：只重新请求空回复（及失败题）。挑题来源是报表实际读的目录（有 _merged 用
    # _merged），结果写 <out>_emptyfix/，跑完合并回 <out>_merged/。不走缓存 —— 缓存键含完整
    # model_args（连 num_concurrent）与任务配置哈希，任一项和当初不同就整体失效，
    # 2026-10-17 用缓存重跑来补空回复，结果 deepseek 的 vsibench / eai 被整份重新请求。
    # ⚠️ 不区分空回复的成因。同日排查：OpenRouter 模型 1853 条空回复里约 95% 是推理撞上 token
    # 上限或推理后没给答案，属于模型行为；对它们重新请求等于只给这个模型重掷骰子（deepseek
    # vsibench 34.4 → 42.9），已撤销。只在空回复的 output_tokens 为 0 时才该用它。
    merge_after = None
    if retry_empty:
        base = HERE / bench["out"]
        src = pathlib.Path(f"{base}_merged") if pathlib.Path(f"{base}_merged").exists() else base
        failed = collect_failed(src, include_empty=True)
        n = sum(len(v) for v in failed.values())
        print(f"\n[{name}] 补跑空回复/失败题（来源 {src.name}）：{n} 题  {dict((k, len(v)) for k, v in failed.items())}")
        if not n:
            return 0
        fix_dir = pathlib.Path(f"{base}_emptyfix")
        if fix_dir.exists():
            import shutil, time as _time
            shutil.move(str(fix_dir), f"{fix_dir}.old-{_time.strftime('%Y%m%d_%H%M%S')}")
        filter_path = HERE / f"{bench['out']}_emptyfix_filter.json"
        filter_path.write_text(json.dumps(failed, indent=1))
        env["LMMS_DOC_ID_FILTER"] = str(filter_path)
        cmd[cmd.index("--tasks") + 1] = ",".join(failed)
        cmd[cmd.index("--output_path") + 1] = str(fix_dir)
        if "--use_cache" in cmd:
            i = cmd.index("--use_cache"); del cmd[i:i + 2]
        merge_after = (src, fix_dir, pathlib.Path(f"{base}_merged"))
        bench = {**bench, "out": fix_dir.name, "log": bench["log"].replace(".log", "_emptyfix.log")}
        skip_existing = False

    log_path = bench["log"]
    out_dir = HERE / bench["out"]
    print(f"\n{'='*60}")
    print(f"[{name}] 开始评测")
    print(f"  output → {bench['out']}")
    print(f"  log    → {log_path}")

    if skip_existing and _is_done(out_dir):
        print(f"  ↩ 已有结果（{out_dir}），跳过")
        return 0

    if dry:
        print("  [DRY RUN] 不执行")
        return 0

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)

    with open(log_path, "a") as logf:
        logf.write(f"\nEXIT={proc.returncode}\n")

    if proc.returncode == 0:
        print(f"  ✓ 完成（exit 0）")
        if merge_after:
            merge_fix(*merge_after)
    else:
        print(f"  ✗ 失败（exit {proc.returncode}），查 {log_path}")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS),
                    help="要跑的 benchmark，默认全部")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--skip-existing", action="store_true",
                    help="跳过 output 目录已有 results*.json 的 benchmark（中断续跑用）")
    ap.add_argument("--cache-dir", default=str(HERE / ".response_cache"),
                    help="response-level 缓存目录，用于 task 内断点续跑（默认 .response_cache/）")
    ap.add_argument("--no-cache", action="store_true", help="禁用 response 缓存")
    ap.add_argument("--retry-failed", action="store_true",
                    help="只补跑已有结果里 API 失败（LMMS_EVAL_REQUEST_FAILED）的题，输出到 <out>_retry/，再用 merge_retry.py 合并")
    ap.add_argument("--retry-empty", action="store_true",
                    help="只补跑报表所读目录（有 _merged 用 _merged）里的空回复与失败题，输出 <out>_emptyfix/ 并合并回 <out>_merged/。"
                         "⚠️ 推理撞上限的空回复不该补（等于重掷骰子），先看 token_counts")
    # API 参数覆盖（优先级高于 .env）
    ap.add_argument("--model", help="模型名，覆盖 .env 的 EVAL_ROUTER_MODEL")
    ap.add_argument("--base-url", help="API base URL，覆盖 .env 的 EVAL_ROUTER_BASE_URL")
    ap.add_argument("--api-key", help="API key，覆盖 .env 的 EVAL_ROUTER_API_KEY")
    ap.add_argument("--timeout", type=int, help="单请求超时秒数，覆盖默认的 600")
    ap.add_argument("--num-concurrent", type=int, help="并发请求数，覆盖默认的 8")
    ap.add_argument("--max-size-in-mb", type=float, help="图片压缩上限 MB，覆盖默认的 2")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个任务只跑前 N 题（调试用；留空跑全量）")
    ap.add_argument("--output-dir-suffix", default="",
                    help="输出目录后缀，换模型时隔离结果目录（如 gpt4o 则写入 eval_A_causalphys_gpt4o/）。"
                         "不填则用模型名自动生成（取最后一个 / 后面的部分，空格和特殊字符替换为 _）")
    args = ap.parse_args()

    cfg = load_env()

    # CLI 覆盖 .env
    if args.model:
        cfg["EVAL_ROUTER_MODEL"] = args.model
    if args.base_url:
        cfg["EVAL_ROUTER_BASE_URL"] = args.base_url
    if args.api_key:
        cfg["EVAL_ROUTER_API_KEY"] = args.api_key
    if args.timeout:
        cfg["_TIMEOUT"] = str(args.timeout)
    if args.num_concurrent:
        cfg["_NUM_CONCURRENT"] = str(args.num_concurrent)
    if args.max_size_in_mb:
        cfg["_MAX_SIZE_IN_MB"] = str(args.max_size_in_mb)

    if not cfg.get("EVAL_ROUTER_API_KEY"):
        raise SystemExit("EVAL_ROUTER_API_KEY 未设置，请填入 .env 或用 --api-key 传入")

    # 确定输出目录后缀
    suffix = args.output_dir_suffix
    if not suffix:
        # 从模型名自动生成：取最后一段，替换非字母数字字符为 _
        import re as _re
        model_name = cfg.get("EVAL_ROUTER_MODEL", "model")
        suffix = _re.sub(r"[^a-zA-Z0-9]", "_", model_name.split("/")[-1]).strip("_")

    cache_dir = None if args.no_cache else args.cache_dir
    print(f"模型: {cfg.get('EVAL_ROUTER_MODEL')}")
    print(f"端点: {cfg.get('EVAL_ROUTER_BASE_URL')}")
    print(f"输出后缀: {suffix}")
    print(f"任务: {args.tasks}")
    print(f"缓存: {cache_dir or '禁用'}")
    if args.skip_existing:
        print("续跑模式：已有结果的 benchmark 将跳过")

    failures = []
    for name in args.tasks:
        bench = dict(BENCHMARKS[name])
        # 将输出目录里的旧后缀替换为当前 suffix
        for old in ("_qwen3max",):
            bench["out"] = bench["out"].replace(old, f"_{suffix}")
            bench["log"] = bench["log"].replace(old, f"_{suffix}")
        rc = run_benchmark(name, cfg, bench, args.dry_run,
                           skip_existing=args.skip_existing, cache_dir=cache_dir,
                           retry_failed=args.retry_failed, limit=args.limit,
                           retry_empty=args.retry_empty)
        if rc != 0:
            failures.append(name)

    print("\n" + "="*60)
    if failures:
        print(f"失败的 benchmark: {failures}")
        sys.exit(1)
    else:
        print("全部完成 ✓")


if __name__ == "__main__":
    main()
