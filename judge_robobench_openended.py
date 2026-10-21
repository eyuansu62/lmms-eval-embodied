#!/usr/bin/env python3
"""Post-hoc judge for RoboBench open-ended (planning) subtasks — official protocol.

判分与官方 yulin-luo/RoboBench 的 src/robobench/evaluation/planning.py 一致：

  Q1 多步规划  judge 看首帧图 + GT 动作序列 + GT DAG + 模型计划（官方 v3 prompt），
               给 node_correctness / task_completion 两个 0-10 整数，
               分数 = (node + completion) / 20（官方是 ×100，这里保持 0-1 与报表口径一致）
  Q2 下一步    judge 先从回复里抽出一步，再按 技能(0/1) / 对象(0/0.5/1) / 参数(0/0.5/1)
               三维比较；技能或对象为 0 时参数强制为 0；分数 = 三项之和 / 3
  Q3 状态判断  yes/no 精确匹配（评测时 utils 已判；这里用同一函数复算）

与官方的差异（有意为之）：
  * judge 模型：官方 config/benchmark.example.yaml 的 evaluation.planning.eval_model 是
    浮动名 "gpt-4o"。2026-10-20 起默认走 OpenRouter 的 openai/gpt-4o-2024-08-06（钉日期，
    防止浮动名被改指）。OpenRouter 对本地 IP 返回 403 地区限制，须经 .env 的
    OPENROUTER_PROXY 代理。之前用本地路由器的 qwen3.8-max：它本身是被测模型，有自评偏差，
    10-20 之前的 judged 文件都是它判的。--judge-model / --judge-profile 可改。
    **所有模型必须用同一个 judge**，换 judge 之后的分数不能和之前的横比；judged 文件里
    每题的 rb_judge.judge_model 记了是谁判的。
  * Q3 用正则取最后一个 yes/no，官方是让 judge 抽取。
  * judge 解析失败按官方做法不计入均值（会在汇总里报 parse_fail 数），请求失败
    （LMMS_EVAL_REQUEST_FAILED）的题记 0 分，走报表的「含失败」口径。

输出：
  * 每个任务在源 samples 文件旁写 ``<ts>_judged_samples_<task>.jsonl``：逐题覆写
    rb_openended_score，并附 rb_judge 明细。build_report 按时间戳取最新文件，
    ``<ts>_judged`` 排在 ``<ts>`` 之后，所以会自动读到判过分的版本；而之后若重跑
    评测生成更新的 samples，又会回到「待判」状态，提醒重判。
  * ``--out``（默认 ``<eval_dir>_judge.json``）：各任务汇总。
  * judge 调用缓存在 ``<eval_dir>/robobench_judge_cache.jsonl``，中断后重跑不会重复请求。

Usage:
    python judge_robobench_openended.py --eval-dir eval_B_robobench_plan_qwen3max
        [--judge-model openai/gpt-4o-2024-08-06] [--judge-profile OPENROUTER]
        [--workers 8] [--tasks robobench_plan_color ...]
"""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures as cf
import glob
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
TASK_DIR = HERE / "lmms_eval" / "tasks" / "robobench"
sys.path.insert(0, str(TASK_DIR))
import utils as rb  # noqa: E402  lmms_eval/tasks/robobench/utils.py

FAIL_MARK = "[LMMS_EVAL_REQUEST_FAILED"
DEFAULT_JUDGE = "openai/gpt-4o-2024-08-06"
# .env 里的端点前缀：<PROFILE>_BASE_URL / <PROFILE>_API_KEY，可选 <PROFILE>_PROXY
DEFAULT_PROFILE = "OPENROUTER"
# 单张图超过这个大小就转 JPEG。网关对 >=10MB 的请求体会报误导性的 "unexpected EOF"。
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def is_openended_task(task: str) -> bool:
    return task in ("robobench_implicit_goal", "robobench_implicit_goal_q1",
                    "robobench_explicit_goal", "robobench_number") \
        or task.startswith(("robobench_plan_", "robobench_plan20_"))


# ── 官方 prompt（逐字照搬 evaluation/planning.py）──────────────────────────────

PROMPT_V3 = """You are a world-simulator judge for robot task plans. Given a scene image, a ground-truth (GT) action sequence, a GT DAG of dependencies, and a model's predicted plan, output two integer scores in [0, 10].

## Inputs
- **Scene image** `I_0`: first frame of the task — use it to identify the objects, their initial spatial relations, and physical constraints.
- **GT action list** `A*`: e.g. `1-grasp(cup), 2-pick_up(cup), 3-place(cup, table)` (single-arm) or `1-left:move_to(none, towel), right:move_to(none, towel), 2-left:grasp(edge), right:grasp(edge), ...` (dual-arm).
- **GT DAG** `G`: dependency graph over GT actions; defines which actions enable which milestones.
- **Model plan**: predicted action list in the same syntax.

Each action is a node `(skill, object, parameter?)`.

## Score 1: Node Correctness

**Step 1 — count GT nodes** `N_GT` exactly:
- A line `i-action(args)` (single arm / human / mobile) contributes **1 node**.
- A line `i-left:actionL, right:actionR` (dual-arm) contributes **2 nodes**, one per arm. `no_ops` is a node and only matches another `no_ops`.
- Skip nothing. Do not collapse, merge, or drop nodes.

**Step 2 — greedy one-to-one match**: walk the model plan left-to-right. For each predicted node, find at most one yet-unmatched GT node where ALL three hold:
- **Skill**: identical token. `grasp ≠ pick_up`, `push ≠ pull`, `place ≠ insert`.
- **Object**: same physical referent. Allow alias only when the two names point to the same entity in the scene (`faucet ≈ tap`, `cup ≈ mug`, `dish_rack ≈ rack`). **Different colors / sides / IDs never match** (`red_button ≠ green_button`, `left_door ≠ right_door`, `object_3 ≠ object_8`, `purple_cylinder ≠ pink_cylinder_block`). For dual-arm, the predicted node's arm side must match the GT side.
- **Parameter**: directionally or functionally equivalent (`clockwise ≈ cw`, `open ≈ on` for binary state, `under_table ≈ under_of_table`). Specific named locations must match (`bottom_left_corner ≠ point_5`).

A node is fully correct or 0 — **no partial credit per node**.

**Step 3** — `NodeCorrectness = floor(matched / N_GT × 10)`.

## Score 2: Task Completion (world-simulation rollout)

**IMPORTANT — handle trivial / empty `S*` before scoring:**
- If after applying the exclusion list `S*` is **empty** (the goal contains only excluded robot motions — `move_to` / `grasp` / `pick_up` / `release` / `no_ops` / `observation` etc. — and no genuine object-level state change), output `Completion=10` and stop. Both an empty model plan and a non-empty model plan are scored 10 in this case.
- For **dual-arm tasks**: if one arm's full sequence consists of `no_ops` (or excluded motions only), that arm contributes nothing to `S*`. Evaluate `S*` from the OTHER arm only. Do not penalize the model for matching the `no_ops` arm with `no_ops`.
- For **single-step or one-action tasks** (e.g. human embodiment `1-grasp(X), 2-wipe(...)` where `S*={surface cleaned}`), make the `S*` membership decision based on whether the final scene state changes — not on whether intermediate `grasp` / `move_to` nodes are present.

Follow the 4-step procedure:

1. **Initial world** `W_0`: from `I_0`, list the objects mentioned in `A*`, their starting locations / states / containment.
2. **Critical states** `S*`: from `A*` and `G`, extract the set of **object-level state changes** required by the goal:
   - Object moved to a new container/surface (`apple: table → bowl`).
   - Container open/closed (`drawer: closed → open`).
   - Object activated/deactivated (`light: off → on`).
   - Object orientation/assembly meaningfully changed.
   - **Excluded** (these are robot motions, not critical states): `move_to, grasp, hold, approach, align, release, retract, observation, look, scan, plan, think`.
3. **Order** via `G`: enforce precedence (a state can only be marked achieved after its prerequisites).
4. **Rollout**: simulate the model plan step-by-step against `W_0`. For each predicted action: check preconditions are met, then update `W_t → W_{t+1}` with its effect, and mark any `s ∈ S*` it accomplishes. Aggregate achieved set `Ŝ ⊆ S*`.

Counting rules during rollout:
- **Implicit succession**: a later action can satisfy an earlier prerequisite implicitly. `pick_up` implies `grasp`+`move_to`; `place` implies `move_to`. If the goal state is reached, missing intermediate motion nodes do not block it.
- **Wrong-direction action**: physically infeasible step achieves nothing (`push(drawer, inward)` does not open).
- **Wrong object/target**: an action on the wrong referent yields no state change toward `S*` (model places `object_3` at `cloth_corner` when target is `point_5` → that critical state is not in `Ŝ`).
- **Surrogate skill**: a skill that physically achieves the same end state still counts (e.g. `pour` from a container instead of `scoop`+`pour` with a spoon, when the goal is "contents in bowl"). But a skill that cannot reach the goal does not (e.g. `push` along a flat table when the goal requires lifting onto a shelf).

`TaskCompletion = floor(|Ŝ| / |S*| × 10)`.

## Examples

**A. Single-arm, skip pick_up** — GT: `1-move_to(none,apple), 2-grasp(apple), 3-pick_up(apple), 4-place(apple, bowl)` (`N_GT=4`, `S*={apple in bowl}`, `|S*|=1`). Model: `[move_to, grasp, place(apple, bowl)]`. Match: 3/4 → Node=7. Critical state achieved → Completion=10.

**B. Dual-arm towel-fold** — GT: `1-left:move_to(none,L), right:move_to(none,R), 2-left:grasp(L), right:grasp(R), 3-left:pick_up(L), right:pick_up(R), 4-left:unfold(L,t), right:unfold(R,t)` (`N_GT=8`, `S*={towel unfolded}`, `|S*|=1`). Model omits both `pick_up`s, has 6 matched nodes → Node=floor(6/8·10)=7. Towel still ends unfolded → Completion=10.

**C. Drawer wrong direction** — GT: `1-grasp(drawer), 2-pull(drawer, outward)` (`N_GT=2`, `S*={drawer open}`). Model: `[grasp(drawer), push(drawer, inward)]`. Skills don't match on step 2 → Node=floor(1/2·10)=5. Drawer not opened → Completion=0.

**D. Wrong object ID** — GT: `1-grasp(3), 2-pick_up(3), 3-place(3, target)` (`N_GT=3`, `S*={object 3 at target}`). Model uses object `8` throughout. Object IDs differ → 0/3 nodes match, Node=0. Critical state not achieved → Completion=0.

**E. Functional surrogate** — GT: `1-grasp(spoon), 2-scoop(spoon, sauce, jar), 3-pour(spoon, sauce, bowl)` (`S*={sauce in bowl}`). Model: `[grasp(jar), pick_up(jar), pour(jar, sauce, bowl)]`. 0 nodes match (different objects/skills) → Node=0. But the goal state (sauce in bowl) is reached via `pour(jar, ...)` → Completion=10.

**F. Empty `S*` (dual-arm pre-task observation)** — GT: `1-left:move_to(none, book), right:move_to(none, book), 2-left:no_ops, right:no_ops`. After excluding `move_to` and `no_ops`, `S*` is empty. Both judges should output **Completion=10** regardless of what the model produces (no goal state to fail). Node score is computed normally.

**G. Dual-arm one-side no_ops** — GT: `1-left:no_ops, right:move_to(none, lid), 2-left:no_ops, right:grasp(lid), 3-left:no_ops, right:pull(lid, up)`, `S*={lid open}` (from right arm). Evaluate Completion only against the right-arm sub-plan. If model's right arm achieves `lid=open`, Completion=10. The left-arm `no_ops` lines are not penalized.

## Output (JSON only)

```json
{
  "node_correctness": {"result": <int 0-10>, "reason": "matched=X / N_GT=Y → floor(X/Y*10)=Z. <list any aliases used>"},
  "task_completion":  {"result": <int 0-10>, "reason": "achieved=A / |S*|=B → floor(A/B*10)=C. <name S* and which were achieved>"}
}
```

Both scores must be integers in [0, 10]. The `reason` must show the explicit counts. Output JSON only, no surrounding prose."""

Q1_USER = """Task Instruction:
{question}

Ground Truth Action List:
{gt}

GT DAG Dependencies:
{dag}

Model Plan Action List:
{pred}

Evaluate the model plan. Output JSON only."""

PROMPT_COMPARE_STEPS = """
You are evaluating the similarity between two robot action steps. Your task is to evaluate the extracted step against the ground truth step using specific criteria.

### Input data format
{{
    "extracted_step": "The step extracted from model response",
    "gt_step": "The ground truth step"
}}

### Evaluation criteria

1. **Skill usage accuracy (0, 1 point)**:
   - Consider only the skill/action part of both steps (ignore the objects/parameters)
   - Score based on whether the skills are completely identical.
   - 1: Perfect skill match (exactly the same action word)
   - 0: Different or not completely identical skills

2. **Operation object reasonableness (0, 0.5, or 1 point)**:
   - Consider only the operation object(s) part of both steps
   - Check if objects refer to similar or related things
   - 1: Objects are identical or clearly refer to the same thing (e.g., "door_handle" vs "door handle", "microwave_door" vs "microwave_handle")
   - 0.5: Objects are similar or related (e.g., "table" vs "table_leg", "cup" vs "mug", objects referring to the same category)
   - 0: Objects are completely different or unrelated

3. **Parameter accuracy (0, 0.5, or 1 point)**:
   - Consider the function parameters and their quality
   - If skill is 0 or object is 0, this parameter score is automatically 0
   - If skill and object both have scores, evaluate the parameter quality:
   - 1: Parameters are completely correct and high quality
   - 0.5: Parameters are partially correct or medium quality
   - 0: Parameters are incorrect or low quality

### Evaluation Guidelines
- Skill evaluation must be strict - only award 1 point when skills are exactly identical
- Object evaluation can be more flexible - consider semantic similarity
- Parameter evaluation depends on skill and object scores being non-zero
- Give detailed reasons explaining your scoring decision
- Consider the context and precision required for robot task execution

### Output format
Please output your evaluation results strictly according to the following JSON structure:
{{
    "skill_usage_accuracy": {{"result": x, "reason": "brief explanation of skill evaluation"}},
    "operation_object_reasonableness": {{"result": y, "reason": "brief explanation of object evaluation"}},
    "parameter_accuracy": {{"result": z, "reason": "brief explanation of parameter evaluation"}}
}}

Extracted step: {extracted_step}
Ground truth step: {gt_step}

Please output your results as required.
"""

PROMPT_EXTRACT_STEP = """
Extract the next step from this response. The step should be in the format: skill(element1, element2, ...)

For example:
- "grasp(microwave_handle)"
- "push(microwave_handle, close)"
- "move_to(none, drawer)"

Response: {response}

Return ONLY the step in the correct format, nothing else.
"""


# ── 解析（逻辑同官方 _parse_q1_score / _parse_q2_score）────────────────────────

def _json_obj(text: str) -> dict | None:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def parse_q1(text: str) -> tuple[int, int] | None:
    data = _json_obj(text)
    try:
        vals = [data["node_correctness"]["result"], data["task_completion"]["result"]]
    except (KeyError, TypeError):
        return None
    out = []
    for v in vals:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            s = int(v)
        else:
            m = re.search(r"-?\d+(?:\.\d+)?", str(v))
            if not m:
                return None
            s = int(float(m.group()))
        if not 0 <= s <= 10:
            return None
        out.append(s)
    return out[0], out[1]


def parse_q2(text: str) -> tuple[float, float, float] | None:
    data = _json_obj(text)
    try:
        vals = [data["skill_usage_accuracy"]["result"],
                data["operation_object_reasonableness"]["result"],
                data["parameter_accuracy"]["result"]]
        vals = [float(v) for v in vals]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(0 <= v <= 1 for v in vals):
        return None
    return vals[0], vals[1], vals[2]


# ── judge 调用 ────────────────────────────────────────────────────────────────

class Judge:
    def __init__(self, model: str, api_key: str, base_url: str, cache_path: pathlib.Path,
                 proxy: str = ""):
        import httpx
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=600,
                             http_client=httpx.Client(proxy=proxy or None, timeout=600))
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            for line in cache_path.read_text(errors="replace").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["text"]
        self.calls = 0

    def ask(self, messages: list[dict], parse, attempts: int = 3):
        """返回 (解析结果或 None, 原始回复)。解析失败会重问（不同 attempt 不共用缓存）。"""
        text = ""
        for attempt in range(attempts):
            payload = json.dumps([self.model, messages, attempt], ensure_ascii=False)
            key = hashlib.sha1(payload.encode()).hexdigest()
            text = self.cache.get(key)
            if text is None:
                text = self._call(messages)
                if not text:
                    # _call 重试耗尽返回 ""：不能缓存，否则之后每次重跑都命中这条空回复，
                    # 该题永远判不出来（2026-10-20 OpenRouter 403 地区限制时就写进了 8 条）
                    continue
                with self.lock:
                    self.cache[key] = text
                    with self.cache_path.open("a") as f:
                        f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
            parsed = parse(text)
            if parsed is not None:
                return parsed, text
        return None, text

    def _call(self, messages: list[dict]) -> str:
        # 流式：网关空闲 60s 即 504，thinking 模型非流式会被截断
        for attempt in range(6):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0,
                    max_tokens=4096, stream=True,
                )
                parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                with self.lock:
                    self.calls += 1
                return "".join(parts)
            except Exception as exc:  # noqa: BLE001
                wait = min(60, 2 ** attempt)
                print(f"  [judge retry {attempt + 1}] {type(exc).__name__}: {str(exc)[:120]}; sleep {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
        return ""


def first_frame_data_url(doc: dict) -> str:
    """Q1 judge 用首帧（官方取 image_urls[0]）。"""
    fname = doc["image_paths"][0]
    rb._get_image(doc["id"], fname)  # 确保已下载
    path = next(rb.CACHE_DIR / p for p in rb._candidate_paths(doc["id"], fname)
                if (rb.CACHE_DIR / p).exists())
    data = path.read_bytes()
    media = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    if len(data) > MAX_IMAGE_BYTES:
        from PIL import Image

        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, format="JPEG", quality=90)
        data, media = buf.getvalue(), "image/jpeg"
    return f"data:{media};base64,{base64.b64encode(data).decode()}"


def judge_one(judge: Judge, doc: dict, pred: str) -> dict:
    """返回 {"score": float | None, ...明细}。score=None 表示 judge 解析失败（不计入均值）。"""
    q = doc["qtype"]
    if q == "Q3":
        guess = rb.extract_yes_no(pred)
        return {"qtype": q, "score": float(guess == doc["answer"].strip().lower()) if guess else 0.0,
                "extracted": guess}

    if q == "Q1":
        user = Q1_USER.format(question=doc["question"], gt=doc["answer"],
                              dag=json.dumps(doc["dag"], indent=2), pred=pred)
        messages = [
            {"role": "system", "content": PROMPT_V3},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": first_frame_data_url(doc)}},
                {"type": "text", "text": user},
            ]},
        ]
        parsed, raw = judge.ask(messages, parse_q1)
        if parsed is None:
            return {"qtype": q, "score": None, "parse_error": True, "raw": raw[:2000]}
        node, comp = parsed
        return {"qtype": q, "score": (node + comp) / 20.0,
                "node_correctness": node, "task_completion": comp}

    # Q2：先抽取下一步，再三维比较
    extracted, _ = judge.ask(
        [{"role": "user", "content": PROMPT_EXTRACT_STEP.format(response=pred)}],
        lambda t: t.strip() or None,
    )
    if not extracted:
        return {"qtype": q, "score": None, "parse_error": True, "stage": "extract"}
    # 官方取法：gt_answer.split("-")[-1]。有 2 道 Q2 的 GT 是多步序列，这样取到的是
    # 最后一步 —— 是上游数据的问题，照搬以保持与官方分数可比。
    gt_step = doc["answer"].split("-")[-1]
    parsed, raw = judge.ask(
        [{"role": "user", "content": PROMPT_COMPARE_STEPS.format(extracted_step=extracted, gt_step=gt_step)}],
        parse_q2,
    )
    if parsed is None:
        return {"qtype": q, "score": None, "parse_error": True, "stage": "compare", "raw": raw[:2000]}
    skill, obj, param = parsed
    if skill == 0 or obj == 0:
        param = 0.0
    return {"qtype": q, "score": (skill + obj + param) / 3.0, "extracted": extracted,
            "skill": skill, "object": obj, "parameter": param}


# ── 文件处理 ──────────────────────────────────────────────────────────────────

def latest_source_samples(eval_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """{task: 最新的原始 samples 文件}；跳过本脚本写出的 *_judged_samples_*。"""
    best: dict[str, pathlib.Path] = {}
    for f in glob.glob(str(eval_dir / "**" / "*_samples_*.jsonl"), recursive=True):
        p = pathlib.Path(f)
        if "_judged_samples_" in p.name:
            continue
        task = p.name.split("_samples_")[-1][: -len(".jsonl")]
        if not is_openended_task(task):
            continue
        if task not in best or p.name > best[task].name:
            best[task] = p
    return best


def response_of(item: dict) -> str:
    raw = item.get("filtered_resps") or item.get("resps") or ""
    while isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw)


def load_task(task: str, src: pathlib.Path) -> tuple[list[dict], list[dict]]:
    docs = [json.loads(l) for l in (TASK_DIR / "data" / f"{task}.jsonl").read_text().splitlines() if l.strip()]
    if "qtype" not in docs[0]:
        raise SystemExit(f"{task}: 数据缺 qtype 字段，先跑 build_robobench_jsonl.py 重建")
    items = [json.loads(l) for l in src.read_text(errors="replace").splitlines() if l.strip()]
    return docs, items


def judge_item(judge: Judge, doc: dict, item: dict) -> dict:
    pred = response_of(item)
    if FAIL_MARK in pred:
        return {"qtype": doc["qtype"], "score": 0.0, "request_failed": True}
    try:
        return judge_one(judge, doc, pred)
    except Exception as exc:  # noqa: BLE001  单题出错不拖垮整个任务
        return {"qtype": doc["qtype"], "score": None, "parse_error": True,
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def write_task(task: str, src: pathlib.Path, items: list[dict], details: list[dict], judge_model: str) -> dict:
    out_path = src.with_name(src.name.replace("_samples_", "_judged_samples_", 1))
    with out_path.open("w") as f:
        for item, det in zip(items, details):
            item = dict(item)
            item["rb_openended_score"] = det["score"] if det["score"] is not None else -1.0
            item["rb_judge"] = {**det, "judge_model": judge_model}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_q: dict[str, list[float]] = collections.defaultdict(list)
    q1_parts = {"node_correctness": [], "task_completion": []}
    for det in details:
        if det["score"] is not None:
            by_q[det["qtype"]].append(det["score"])
        for k in q1_parts:
            if k in det:
                q1_parts[k].append(det[k])
    scored = [d["score"] for d in details if d["score"] is not None]
    return {
        "task": task,
        "source": str(src),
        "judged_file": str(out_path),
        "judge_model": judge_model,
        "n": len(details),
        "request_failed": sum(1 for d in details if d.get("request_failed")),
        "parse_fail": sum(1 for d in details if d["score"] is None),
        "score": sum(scored) / len(scored) if scored else None,
        "by_qtype": {q: {"n": len(v), "score": sum(v) / len(v)} for q, v in sorted(by_q.items())},
        "q1_mean_0_10": {k: sum(v) / len(v) for k, v in q1_parts.items() if v},
    }


def load_env() -> dict[str, str]:
    cfg: dict[str, str] = {}
    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip().strip('"')
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="评测输出目录（有补跑时用 *_merged）")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE,
                    help=f"judge 模型（默认 {DEFAULT_JUDGE}；所有被测模型必须用同一个）")
    ap.add_argument("--judge-profile", default=DEFAULT_PROFILE,
                    help=f"取 .env 的 <PROFILE>_BASE_URL / <PROFILE>_API_KEY（默认 {DEFAULT_PROFILE}；"
                         "本地路由器是 EVAL_ROUTER）")
    ap.add_argument("--base-url", help="judge 端点，覆盖 --judge-profile")
    ap.add_argument("--api-key", help="judge key，覆盖 --judge-profile")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", help="只判这些任务（默认目录里所有开放式任务）")
    ap.add_argument("--out", help="汇总 JSON 路径（默认 <eval_dir>_judge.json）")
    args = ap.parse_args()

    eval_dir = pathlib.Path(args.eval_dir)
    sources = latest_source_samples(eval_dir)
    if args.tasks:
        sources = {t: p for t, p in sources.items() if t in args.tasks}
    if not sources:
        sys.exit(f"{eval_dir} 下没有 RoboBench 开放式任务的 samples 文件")

    cfg = load_env()
    prof = args.judge_profile
    api_key = args.api_key or cfg.get(f"{prof}_API_KEY") or os.environ.get(f"{prof}_API_KEY", "")
    base_url = args.base_url or cfg.get(f"{prof}_BASE_URL") or os.environ.get(f"{prof}_BASE_URL", "")
    if not api_key or not base_url:
        sys.exit(f"judge 需要端点和 key：填 .env 的 {prof}_BASE_URL / {prof}_API_KEY，或传 --base-url / --api-key")

    proxy = cfg.get(f"{prof}_PROXY") or os.environ.get(f"{prof}_PROXY", "")
    judge = Judge(args.judge_model, api_key, base_url, eval_dir / "robobench_judge_cache.jsonl", proxy)
    print(f"judge: {judge.model} @ {base_url}{'（经代理）' if proxy else ''}  缓存 {len(judge.cache)} 条")

    # 所有任务的题放进同一个线程池：逐任务判的话，每个任务尾部都会空转等最慢的那几题
    loaded = {task: load_task(task, src) for task, src in sorted(sources.items())}
    jobs = [(task, i) for task, (_, items) in loaded.items() for i in range(len(items))]
    details: dict[str, list] = {task: [None] * len(items) for task, (_, items) in loaded.items()}
    print(f"共 {len(jobs)} 题，{len(loaded)} 个任务，并发 {args.workers}")

    def work(job):
        task, i = job
        docs, items = loaded[task]
        return job, judge_item(judge, docs[items[i]["doc_id"]], items[i])

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, ((task, i), det) in enumerate(ex.map(work, jobs), 1):
            details[task][i] = det
            if k % 50 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  judge 调用 {judge.calls} 次  [{time.time() - t0:.0f}s]", flush=True)

    results = []
    for task, (_, items) in loaded.items():
        r = write_task(task, sources[task], items, details[task], judge.model)
        results.append(r)
        sc = f"{r['score'] * 100:6.2f}" if r["score"] is not None else "   -  "
        qs = "  ".join(f"{q}={v['score'] * 100:.1f}(n={v['n']})" for q, v in r["by_qtype"].items())
        print(f"  {task:36s} {sc}  n={r['n']:4d} 请求失败={r['request_failed']} "
              f"判分失败={r['parse_fail']}  {qs}")

    out = pathlib.Path(args.out) if args.out else eval_dir.with_name(eval_dir.name + "_judge.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\njudge 调用 {judge.calls} 次（其余命中缓存） → {out}")


if __name__ == "__main__":
    main()
