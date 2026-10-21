"""ERIQ · Embodied Reasoning Intelligence Quotient（lmms-eval task）。

具身推理基准，6,052 题 / 15 个子任务 / 4 个推理维度。数据来自真实机器人
试次的第一视角（Robo View），全部是确定性选择题（MC / Yes-No），**不需要
LLM judge**。

本 task 只接 能力体系（CAPABILITIES.md）里挂到能力行的 7 个（对应 8 个子任务）：
    #14 指令具化          Human Intention Comprehension
    #25 失败定位与原因分析   Mistake Classification
    #27 单体/多体状态理解    Action Understanding
    #28 子任务分解         Subtask Planning
    #32 进度估计          Task Progress
    #33 失败检测          Mistake Existence + Success Detection
    #34 恢复策略          Mistake Recovery

数据：https://huggingface.co/datasets/KineMind/ERIQ
      （官方 README 写的是 agibot-research/ERIQ，该路径现 307 重定向到
        KineMind/ERIQ；**别写死旧名字**。）
代码：https://github.com/GenieReasoner/ERIQ
论文：https://arxiv.org/abs/2512.24125

⚠️ 三个坑，都实测过
-----------------------------------------------------------------------
1. **`gt_answer` 字段全是空的。** 16 个文件、3061 题，`gt_answer` 一律为 `""`。
   真实答案在 `conversations[1].value`（gpt 轮）。官方 README 的字段说明是错的，
   照它写会全部拿到空标签、分数无从谈起。

2. **图片路径缺前缀。** JSON 里 `image` 是不带前缀的相对路径
   （`1832/25953/.../head_color.jpg`），实际文件在仓库的 `images/` 下。
   直接拼会 404，必须先试 `images/<rel>`。这里做了容错：先试带前缀的，
   再回落不带前缀的（仿 robobench 的 `_candidate_paths`），两种布局都能对上。

3. **答案有两种格式。** 14 个文件是字母 A–D；`QA_MISTAKE_EXISTENCE` 是
   Yes/No（167/167）。判分要按 ground truth 的形态分流。
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List

from PIL import Image

REPO_ID = "KineMind/ERIQ"

# 图片树在仓库里的目录名（JSON 里的相对路径不含这一层）
IMAGE_SUBDIR = "images"


# ── 图片解析（按需逐张下载） ─────────────────────────────────────────────
#
# **不要用 snapshot_download 拉整棵树。** 两个理由：
#   1. 整棵 images/ 是 4.27 GB，而采样 800 题只需约 3250 张 ≈ 763 MB。
#   2. 更糟的是它**不校验完整性**：`snapshot_download(local_files_only=True)`
#      只要发现快照目录存在就返回成功，哪怕一张图都没有 —— 2026-10-12 冒烟
#      测试就撞上这个：快照目录存在（只有几个 QA_*.json 的符号链接、共 4 KB），
#      于是判分时 Image.open 抛 FileNotFoundError。
#
# 改成 hf_hub_download 逐张取，与 mvrobobench / robobench 同一套做法。
# 先试 local_files_only=True（命中时 0.000s），未命中才联网（首次约 2.9s/张）。

_DL_LOCK = threading.Lock()
_DL_MEMO: Dict[str, str] = {}  # key -> 本地路径，进程内记忆化


def _cache_dir() -> str:
    """图片落盘目录（HF 缓存不会自动带 local_dir，这里显式指定）。"""
    override = os.environ.get("ERIQ_IMAGE_DIR")
    if override:
        return override
    d = Path(__file__).resolve().parent / "media"
    return str(d)


def _fetch_one(rel_norm: str) -> str:
    """取一张图，返回本地路径。

    `rel_norm` 是**去掉前导斜杠、不含 `images/`** 的相对路径。
    仓库里图片在 `images/<rel>` 下，但 JSON 里的 `image` 字段不含该前缀，
    所以先试带前缀的 key，缺失再试不带前缀的（防止上游哪天把前缀补进 JSON）。
    """
    if rel_norm in _DL_MEMO:
        return _DL_MEMO[rel_norm]

    cache_dir = _cache_dir()

    # 快路径：先看文件在不在本地。必须放在 hf_hub_download 之前 ——
    # 后者即使命中缓存也要发一次网络 HEAD（实测 0.4s/张），而图片解析是在
    # 主循环里**串行**做的，3255 次引用就是 ~22 分钟纯等待。
    # 预取（prefetch_eriq.py）会把图落到 <media>/images/<rel>，所以这里
    # 先查该路径，命中就完全不走网络（0 次 I/O 往返）。
    fast = Path(cache_dir) / IMAGE_SUBDIR / rel_norm
    if fast.exists():
        with _DL_LOCK:
            _DL_MEMO[rel_norm] = str(fast)
        return str(fast)

    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    last_exc: Exception | None = None

    # 两种布局都试：带 images/ 前缀的在前（实测就是这种）
    for key in (f"{IMAGE_SUBDIR}/{rel_norm}", rel_norm):
        try:
            path = hf_hub_download(
                REPO_ID, key, repo_type="dataset", local_dir=cache_dir,
            )
        except Exception as exc:  # 本地没有 / 该 key 不存在 → 试下一个
            last_exc = exc
            continue
        with _DL_LOCK:
            _DL_MEMO[rel_norm] = path
        return path

    raise FileNotFoundError(
        f"ERIQ: 取不到图片 {rel_norm!r}（试过 {IMAGE_SUBDIR}/<rel> 与 <rel>）"
    ) from last_exc


def eriq_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """返回该题的全部图片（PIL）。ERIQ 题均 1~10 张，按时序排列。"""
    refs = doc.get("image")
    if refs is None:
        return []
    if isinstance(refs, str):
        refs = [refs]

    out: List[Image.Image] = []
    for rel in refs:
        path = _fetch_one(str(rel).strip().lstrip("/"))
        with Image.open(path) as im:
            out.append(im.convert("RGB"))
    return out


# ── 题干 / 标签 ───────────────────────────────────────────────────────────

_IMAGE_TAG_RE = re.compile(r"<image>\s*", re.IGNORECASE)


def _first_turn(doc: Dict[str, Any]) -> str:
    """取 human 轮（题干）。"""
    for turn in doc.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "human":
            return str(turn.get("value", ""))
    return ""


def eriq_doc_to_target(doc: Dict[str, Any]) -> str:
    """取 gpt 轮（真实答案）。

    **不要用 `doc["gt_answer"]`** —— 该字段在全部 16 个文件里都是空的。
    """
    convo = doc.get("conversations") or []
    for turn in convo:
        if isinstance(turn, dict) and turn.get("from") == "gpt":
            return str(turn.get("value", "")).strip()
    return ""


def eriq_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    """题干 = human 轮去掉 <image> 占位符。

    选项本身就是行首的 `A. / B. / C. / D.`（或 `A.Yes / B.No`），不需要重新
    排版 —— 实测 `MISTAKE_EXISTENCE` 是 `(Yes/No)` 结尾、没有选项行，那类题
    靠 post_prompt 告诉模型只回 Yes/No。
    """
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")

    question = _IMAGE_TAG_RE.sub("", _first_turn(doc)).strip()
    return f"{pre}{question}{post}"


# ── 判分 ──────────────────────────────────────────────────────────────────

# 官方 eval_code/eval_hf.py 的 preprocess_answer，逐字照抄。
#
# ⚠️ 实测这个正则**不是**「必须以字母打头」那么简单：
#   `^([A-F])[\W\s_]*` 里的 `[\W\s_]*` 能匹配**零个**字符，所以它实际是
#   「首字符落在 A–F 就取它」。于是 ——
#       'Based on the image, the answer is A'  ->  'B'   （取了 Based 的 B！）
#       'Answer: C'                            ->  'A'   （取了 Answer 的 A！）
#       '(C)' / 'The answer is B' / '**B**'    ->  None
#   这是**取错字母**而非「判 0」，比单纯严格更糟。默认口径仍用官方实现以保证
#   与 README 的 6 个 baseline 可比，但**必须**同时看宽松口径（见下）来区分
#   「模型不会」与「格式不合规」。
_STRICT_MC_RE = re.compile(r"^([A-F])[\W\s_]*")
_STRICT_YN_RE = re.compile(r"^(yes|no)[\W\s_]*", re.IGNORECASE)


def extract_official(text: str, is_yes_no: bool) -> str | None:
    """官方口径，**逐字对齐** eval_code/eval_hf.py 的 preprocess_answer。

    刻意**不做 strip**：官方直接对原始回复跑 `^([A-F])`，所以
    `' A'` / `'\\nA'` 这类**前导空白**会让官方返回 None。加了 strip 就不再
    与官方一致、也就失去了跟 README baseline 对读的意义（实测差异 2/49 例）。
    宽松口径 `extract_lenient` 会把这些捞回来。

    取不到返回 None（与官方一致）。
    """
    if not isinstance(text, str):
        return None
    if is_yes_no:
        m = _STRICT_YN_RE.search(text.lower())
        return m.group(1) if m else None
    m = _STRICT_MC_RE.search(text.upper())
    return m.group(1).upper() if m else None


def extract_lenient(text: str, is_yes_no: bool, letters: str = "ABCDEF") -> str | None:
    """宽松口径：复用仓库共享的 `_task_utils/mcq_extract`，Yes/No 走词位判定。

    与官方口径并列落盘，用来分辨「模型答错」与「模型答对但格式不合规」。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    body = _strip_think(text)
    if is_yes_no:
        # 末行优先，其次整段：要求出现 yes/no 且不能两个都出现
        last = body.strip().split("\n")[-1].lower()
        for seg in (last, body.lower()):
            has_yes, has_no = "yes" in seg, "no" in seg
            if has_yes ^ has_no:
                return "yes" if has_yes else "no"
        return None

    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    got = extract_mcq_answer(body, choices=list(letters))
    return got.upper() if got else None


def _strip_think(text: str) -> str:
    """去掉推理标签；取 `</think>` / `</thought>` 之后的内容。"""
    for tag in ("</think>", "</thought>", "<answer>"):
        if tag == "<answer>":
            m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
        elif tag in text:
            text = text.split(tag, 1)[1]
    return text.strip()


def _normalise(gt: str) -> str:
    """ground truth 归一：去空白、大写。"""
    return str(gt).strip()


# ── Yes/No → 字母 归一（第三个口径） ─────────────────────────────────────
#
# 背景：`success_detection` / `action_understanding` / `subtask_planning` 这三个
# 子任务的选项**字面就是 Yes/No**，但 ground truth 是字母：
#
#     ... has the robot completed the action: "..."?
#     A.Yes
#     B.No
#
# 而官方判分器只认 `^([A-F])`，模型若回 `Yes` / `No` 就判 0。
# 中文注释：erIQ 官方 README 的 6 个 baseline 恰好都规矩地回了字母，所以这个坑
# 一直没暴露；2026-10-13 跑 deepseek-v4.1-flash 时它坚持回 Yes/No，
# 24.9% 的题（199/800）因此被静默判 0，success_detection 单项从 60.4% 塌到 17.3%。
#
# 这不是「宽松抽取」能救的：宽松口径同样只找字母。必须做**语义换算**。
#
# 设计取舍：
#   - **不改官方口径**（eriq_accuracy 仍与 README baseline 逐字可比）；
#   - 归一映射（Yes->A / No->B）**从题干原文解析**，不靠 ground truth 分布反推 ——
#     后者会把「模型答对」当成推断依据，是循环论证；
#   - 只在题干的选项文本**确实**是 {yes,no} 两项时才换算，其余子任务
#     （数字选项如 task_progress 的 `A. 7`、自由文本选项）天然不触发。

_OPT_LINE_RE = re.compile(r"^([A-F])[\.\、\)]\s*(\S+)", re.MULTILINE)


def yes_no_option_map(question: str) -> Dict[str, str] | None:
    """题干若给出 `A.Yes / B.No` 形式，返回 ``{"yes": "A", "no": "B"}``；否则 None。

    只认「恰好两个选项、且选项文本是 yes/no」的情形，避免误伤数字或文本选项。
    """
    if not isinstance(question, str):
        return None
    opts = _OPT_LINE_RE.findall(question)
    if len(opts) != 2:
        return None
    low = {letter: text.strip().lower().rstrip(".") for letter, text in opts}
    if sorted(low.values()) != ["no", "yes"]:
        return None
    return {text: letter for letter, text in low.items()}


def extract_yn_normalized(
    text: str, gt: str, question: str
) -> tuple[str | None, bool]:
    """把 Yes/No 回复换算成字母。

    返回 ``(换算后的字母, 是否真的做了换算)``。不适用时返回 ``(None, False)``，
    调用方应回落到官方口径。
    """
    if not isinstance(text, str):
        return None, False
    gt_u = str(gt).strip().upper()
    if gt_u not in list("ABCDEF"):
        return None, False
    m = yes_no_option_map(question)
    if m is None:
        return None, False
    first = text.strip()[:4].strip().lower().rstrip(".")
    if first in m:
        return m[first], True
    return None, False


def eriq_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """判分。同时落严格（官方）、宽松、Yes/No 归一三个口径。

    三个口径都进同一个 entry 字典，由各自的聚合函数取用。
    """
    response = results[0] if results else ""
    gt = _normalise(eriq_doc_to_target(doc))
    is_yes_no = gt.lower() in ("yes", "no")

    pred_strict = extract_official(response, is_yes_no)
    pred_lenient = extract_lenient(response, is_yes_no)

    # 比较时统一大小写：Yes/No 题比小写，字母题比大写。
    gt_cmp = gt.lower() if is_yes_no else gt.upper()

    def _cmp(pred: str | None) -> bool:
        if pred is None:
            return False
        return (pred.lower() if is_yes_no else pred.upper()) == gt_cmp

    # 第三口径：题干是 A.Yes/B.No 而模型回了 Yes/No 时，换算成字母再判。
    question = _IMAGE_TAG_RE.sub("", _first_turn(doc))
    pred_yn, converted = extract_yn_normalized(response, gt, question)
    yn_correct = _cmp(pred_yn) if converted else _cmp(pred_strict)

    entry = {
        "id": doc.get("id", ""),
        "ground_truth": gt,
        "pred_strict": pred_strict or "",
        "pred_lenient": pred_lenient or "",
        "pred_yn": pred_yn or "",
        "is_correct": _cmp(pred_strict),
        "lenient_correct": _cmp(pred_lenient),
        "yn_correct": yn_correct,
        "yn_converted": converted,
        "is_yes_no": is_yes_no,
    }
    return {
        "eriq_accuracy": entry,
        "eriq_accuracy_lenient": entry,
        "eriq_accuracy_yn": entry,
    }


# ── 聚合 ──────────────────────────────────────────────────────────────────


def eriq_aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    """严格口径（官方）准确率。"""
    if not results:
        return 0.0
    return sum(1 for r in results if r["is_correct"]) / len(results)


def eriq_aggregate_accuracy_lenient(results: List[Dict[str, Any]]) -> float:
    """宽松口径准确率，用于分辨「不会」与「格式不合规」。"""
    if not results:
        return 0.0
    return sum(1 for r in results if r["lenient_correct"]) / len(results)


def eriq_aggregate_accuracy_yn(results: List[Dict[str, Any]]) -> float:
    """Yes/No 归一后的准确率（第三个口径）。

    只在题干确实给出 `A.Yes / B.No`、且模型回了 Yes/No 时才偏离官方口径；
    其余情形与 `eriq_accuracy` 逐题相同。因此：

        eriq_accuracy_yn - eriq_accuracy = 「答案对但格式与官方判分器不兼容」的量

    对 9 个规矩答字母的模型，这个差值是 0~1.3 个百分点；对偏好回 Yes/No 的
    模型（如 deepseek-v4.1-flash）可以到 9 个百分点以上。
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["yn_correct"]) / len(results)
