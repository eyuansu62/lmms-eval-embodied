#!/usr/bin/env python3
"""ASIMOV（google/asimov_agentic）→ 本地 jsonl + 图片。

    python lmms_eval/tasks/asimov/build_asimov_jsonl.py

产物（都不进 git —— 原数据是 gated，见 .gitignore 的 data/ 与 media/ 规则）：
    lmms_eval/tasks/asimov/data/asimov_constraints.jsonl        641 题  #3
    lmms_eval/tasks/asimov/data/asimov_vla_feasibility.jsonl    400 题  #1
    lmms_eval/tasks/asimov/data/asimov_uncertainty.jsonl        396 题  #15
    lmms_eval/tasks/asimov/data/asimov_gauge_readable.jsonl      31 题  #46
    lmms_eval/tasks/asimov/data/asimov_gauge_unreadable.jsonl   131 题  #46
    lmms_eval/tasks/asimov/media/<task>/<example_index>.jpg

数据源
------
HF `google/asimov_agentic`，**gated（人工审批）**。本机 ~/.cache/huggingface/token
对应的账号 2026-10-16 已获批。必须走 `hf_hub_download`：裸 curl 打 resolve/raw
即使带 token 也拿到 403/401 的说明文字（172 字节），而且**不报错**，parquet
解析时才炸。仓库里的官方 harness（*.py）经 hf-mirror 下下来是 0 字节，至今没拿到。

每题原生结构（逐题核对过）：
  决策题（constraints / vla_feasibility / uncertainty）
    messages = [system: [text], user: [text, image(JPEG bytes)]]
    ground_truth = JSON 字符串，其中 ground_truth 字段只有两种取值：
        "should call run_instruction_for_duration[ function]"      -> execute
        "should not call run_instruction_for_duration[ function]"  -> refuse / clarify
    （uncertainty 那份不带 " function" 后缀，所以按前缀判。）
    tools = [robot_stop, run_instruction_for_duration]
  读表题（gauge_reading 两份）
    messages = [user: [text, image]]，无 system、无 tools
    ground_truth = "[lo, hi]"（可读，读数落在区间内算对）或 "unreadable"

这里只做无损搬运：system / user 原文照存，判分口径见 utils.py。
"""

from __future__ import annotations

import io
import json
import pathlib
import re

REPO_ID = "google/asimov_agentic"
HERE = pathlib.Path(__file__).resolve().parent

# task 名 -> parquet 在仓库里的路径
SOURCES = {
    "asimov_constraints": "constraints/physical_constraints.parquet",
    "asimov_vla_feasibility": "vla_feasibility/vla_feasibility_extended_di_vla_protocol.parquet",
    "asimov_uncertainty": "uncertainty/agent_uq_with_di.parquet",
    "asimov_gauge_readable": "gauge_reading/gauge_can_read_adversarial.parquet",
    "asimov_gauge_unreadable": "gauge_reading/gauge_cannot_read_express_uncertainty.parquet",
}
GAUGE_TASKS = {"asimov_gauge_readable", "asimov_gauge_unreadable"}

# 负类的标签名按题意取：有歧义时正确做法是「先澄清」，不是「拒绝」。
# 判分只看「执行 / 不执行」，标签名只影响 prompt 里给的选项（见 utils.py）。
NEGATIVE = {"asimov_uncertainty": "clarify"}

_GAUGE_RE = re.compile(r"Read the (.+?) in the image", re.IGNORECASE)


def _label(task: str, gt: str) -> str:
    if gt.startswith("should call run_instruction_for_duration"):
        return "execute"
    if gt.startswith("should not call run_instruction_for_duration"):
        return NEGATIVE.get(task, "refuse")
    raise ValueError(f"{task}: 未知 ground_truth {gt!r}")


def _norm_type(x: str | None) -> str:
    """uncertainty_type 的「无歧义」有 None / 'None' / 'N/A' / 'NONE' / 'none' 五种写法。"""
    if not x or str(x).strip().upper() in ("NONE", "N/A"):
        return "NONE"
    return str(x).strip().upper()


def _norm_attack(x: str | None) -> str:
    """attack_class 原生有拼写漂移（'CONDITIONAL / SENSORY' 与 'CONDITIONAL_SENSORY'
    各占 3 / 28 题），统一成下划线写法；可执行题为 None -> 'FEASIBLE'。"""
    if not x:
        return "FEASIBLE"
    return "_".join(p.strip() for p in x.replace("/", " ").split()).upper()


def build(task: str, rel: str) -> int:
    # 必须在 import huggingface_hub 之前设：它在导入时就把端点读成常量，
    # 之后再改环境变量不生效，会直连 huggingface.co（本机不通，Network unreachable）。
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
    df = pd.read_parquet(local)

    media = HERE / "media" / task
    media.mkdir(parents=True, exist_ok=True)
    out = HERE / "data" / f"{task}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            msgs = row["messages"]
            roles = [m["role"] for m in msgs]
            if task in GAUGE_TASKS:
                assert roles == ["user"], roles
                system = ""
            else:
                assert roles == ["system", "user"], roles
                system = msgs[0]["content"][0]["text"]
            user = msgs[-1]["content"]
            instruction = user[0]["text"]
            img_bytes = user[1]["image"]
            assert len(user) == 2 and user[0]["image"] is None and img_bytes is not None

            idx = int(row["example_index"])
            img_rel = f"{task}/{idx}.jpg"
            # 原图就是 JPEG，直接落盘，不重编码
            with Image.open(io.BytesIO(img_bytes)) as im:
                assert im.format == "JPEG", im.format
            (HERE / "media" / img_rel).write_bytes(img_bytes)

            if task in GAUGE_TASKS:
                raw = str(row["ground_truth"]).strip()
                answer = "unreadable" if raw == "unreadable" else [float(x) for x in json.loads(raw)]
                assert (task == "asimov_gauge_unreadable") == (answer == "unreadable"), raw
                m = _GAUGE_RE.search(instruction)
                rec = {
                    "id": f"{task}_{idx}",
                    "example_index": idx,
                    "instruction": instruction,
                    "image": img_rel,
                    "answer": answer,
                    "gauge_type": m.group(1) if m else "",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue

            gt = json.loads(row["ground_truth"])
            rec = {
                "id": f"{task}_{idx}",
                "example_index": idx,
                "system": system,
                "instruction": instruction,
                "image": img_rel,
                "label": _label(task, gt["ground_truth"]),
                "issue_type": gt.get("issue_type", ""),
                "rationale": gt.get("rationale", ""),
            }
            if task == "asimov_constraints":
                rec["physical_constraints"] = gt.get("physical_constraints", "")
            elif task == "asimov_uncertainty":
                rec["uncertainty_type"] = _norm_type(gt.get("uncertainty_type"))
                rec["correct_action"] = gt.get("correct_action", "")
            else:
                rec["attack_class"] = _norm_attack(gt.get("attack_class"))
                rec["feasible_substeps"] = gt.get("feasible_substeps", "")
                rec["infeasible_substeps"] = gt.get("infeasible_substeps", "")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"{task}: {n} 题 -> {out.relative_to(HERE.parents[2])}")
    return n


def main() -> None:
    for task, rel in SOURCES.items():
        build(task, rel)


if __name__ == "__main__":
    main()
