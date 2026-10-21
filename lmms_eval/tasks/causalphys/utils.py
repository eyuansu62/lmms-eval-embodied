"""CausalPhys 任务的媒体解析与评测逻辑。

覆盖**文档范围内的 12 个子类**（见 ``_SUBSET_DIR`` 的说明）。
原本接入 16 个，其中 Goal_orientation 的 4 个不在 能力体系（CAPABILITIES.md） 范围内，已移除。

数据来源
--------
``haorentang/causalphys``（HF dataset，未 gated）。
镜像由 ``HF_ENDPOINT`` 指定（本机指向 hf-mirror.com），``hf_hub_download`` 自动走它。

媒体解析（实测全量 annotation 的规则，详见 DATA_SHAPE.md）
--------------------------------------------------------------------------
官方 loader 依次尝试 ``paths`` → ``path`` 字段。实测这一步是必须的：
46 个 ``optics`` 样本只写了 ``paths``，漏掉就会当媒体缺失。

四种媒体形态：

1. **单图**         ``path`` 直接是一个图片文件名（大多数子类）
2. **多图列表**     ``path`` 是文件名列表（``viewpoint_transformation`` 等）
3. **目录**         ``path`` 是目录名，媒体在目录内（``deformation`` 的 94/200 条）
4. **帧序列目录**   ``<id>_frames/``，配合 ``path`` 指向的原始视频
                   （``action_substitution`` 等）

路径归一化
----------
上游有 2 个样本（``spatial_manipulation`` / ``temporal_shifting`` 的 ``109.json``）
把全角右括号 ``）``(U+FF09) 写进了文件名，仓库里实际是半角 ``)``。
用 NFKC 归一化还原。注意这 2 个样本上游**本身**就缺第 3 张图（5 张只有 4 张），
归一化后仍差 1 张，属上游数据问题，这里按"缺就跳过该文件"处理。

判分
----
单选题，选项内嵌在 ``question`` 字符串里（`` A: ... B: ... ``），
没有独立 options 字段，正确答案是 ``ground_truth_answer`` 的选项字母。
``graph`` 是因果图，只在需要 judge 解释性评分时使用，**不喂给模型**。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final

from huggingface_hub import hf_hub_download

_REPO_ID: Final = "haorentang/causalphys"
_DATASET_TYPE: Final = "dataset"

# 本地 media 目录（离线部署）：把 HF cache 复制到这里就能完全离线运行
_LOCAL_MEDIA: Final = pathlib.Path(__file__).resolve().parent / "media"

# 子集 -> 仓库内的目录前缀。**只含在文档范围内的 12 个子类。**
#
# 文档 能力体系（CAPABILITIES.md）里，各条 P0 条目显式点名的有 8 个：
#   Anticipation/{Collision_Prediction,Fluid_Flow,deformation}
#   Intervention/{action_substitution,spatial_manipulation,temporal_shifting}
#   Perception/{Mechanics_Reasoning,containability}
# 另有 #23 用通配前缀覆盖了 Perception/** + Anticipation/** + Intervention/**，
# 于是又纳入 4 个：intention_speculation、viewpoint_transformation、
# scene_reconstruction、optics。
#
# **Goal_orientation 的 4 个子类不在文档范围内，已移除**：
#   Object_Relocation / behaviour_selection / tool_selection / trajectory
# 见 README 的「与文档的对应关系」一节。
_SUBSET_DIR: Final = {
    # Anticipation
    "collision_prediction": "Anticipation/Collision_Prediction",
    "fluid_flow": "Anticipation/Fluid_Flow",
    "intention_speculation": "Anticipation/Intention_Speculation",
    "deformation": "Anticipation/deformation",
    # Intervention
    "action_substitution": "Intervention/action_substitution",
    "spatial_manipulation": "Intervention/spatial_manipulation",
    "temporal_shifting": "Intervention/temporal_shifting",
    "viewpoint_transformation": "Intervention/viewpoint_transformation",
    # Perception
    "mechanics_reasoning": "Perception/Mechanics_Reasoning",
    "scene_reconstruction": "Perception/Scene_Reconstruction",
    "containability": "Perception/containability",
    "optics": "Perception/optics",
}

# 帧序列的每样本帧数。实测所有帧目录固定 5 帧。
# 注意 ``action_substitution`` **不是**帧序列 —— 它的 path 直接就是图片列表
# （帧目录只有 9 个，而图片有 495 张），走"全是图片"分支。
_NUM_FRAMES: Final = 5

_IMAGE_EXT: Final = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _normalize(name: str) -> str:
    """NFKC 归一化，修上游把全角括号写进文件名的问题。"""
    return unicodedata.normalize("NFKC", name)


def _raw_paths(doc: Mapping[str, Any]) -> list[str]:
    """取媒体路径，统一成列表。

    JSONL 里是 ``media_paths``（build 脚本已把上游的 ``paths``/``path``
    统一合并成这个字段，并做过 NFKC 归一化）。
    同时兼容直接读原始 annotation 的情形（那里字段名是 ``paths``/``path``），
    便于未归并的数据也能跑。
    """
    raw = doc.get("media_paths")
    if raw is None:
        raw = doc.get("paths")
    if raw is None:
        raw = doc.get("path")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x)]
    if isinstance(raw, str):
        # JSONL 里 media_paths 是 json.dumps 过的字符串，需再解析一次
        s = raw.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except Exception:  # noqa: BLE001 - 不是合法 json 就按单路径处理
                parsed = None
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x)]
        return [s] if s else []
    return []


def _is_image(name: str) -> bool:
    return name.lower().endswith(_IMAGE_EXT)


def _resolve(subset: str, doc: Mapping[str, Any]) -> list[str]:
    """解析出该样本的媒体文件列表（相对仓库根路径）。

    判定顺序（按各子类的实测形态，见模块 docstring）：

      1. ``media_paths`` 全是图片文件  -> 直接用（单图 / 多图列表）
      2. 其余含视频或目录的情形        -> ``<id>_frames/`` 帧序列
      3. 兜底                          -> 原始视频/文件本身
    """
    subset_dir = _SUBSET_DIR[subset]
    sample_id = str(doc.get("id") or "").strip()
    raws = _raw_paths(doc)

    if not raws:
        return []

    # --- 1: 全是图片 -> 直接用。---
    # 覆盖两类子集：
    #   * 单图       object_relocation / optics / containability / ...
    #   * 多图列表   viewpoint_transformation / action_substitution /
    #                temporal_shifting（path 就是 5 张图）
    if all(_is_image(_normalize(p)) for p in raws):
        return [f"{subset_dir}/data/{_normalize(p)}" for p in raws]

    # --- 2: 帧序列。---
    # 有 ``<id>_frames/`` 的子集：collision_prediction / fluid_flow /
    # intention_speculation / mechanics_reasoning / trajectory /
    # spatial_manipulation / deformation（部分样本）。
    #
    # 注意 ``deformation`` 是**混合形态**，同一子类下：
    #   id=004 -> data/004/0028.png（目录，无 _frames）
    #   id=117 -> data/117.mp4 + data/117_frames/*.jpg（帧序列）
    # 所以这里不按扩展名判断，而是**查索引**看 ``<id>_frames/`` 是否真的存在，
    # 存在才走帧序列，否则落到下面的目录分支。
    if sample_id and _has_dir(f"{subset_dir}/data/{sample_id}_frames/"):
        stem = os.path.splitext(_normalize(raws[0]))[0]
        if not stem:
            stem = sample_id
        return [
            f"{subset_dir}/data/{sample_id}_frames/{stem}_f{i:03d}.jpg"
            for i in range(_NUM_FRAMES)
        ]

    # --- 3: 目录（deformation 的 94/200 条：path 是目录名，无扩展名）。---
    if len(raws) == 1:
        dirname = _normalize(raws[0]).rstrip("/")
        if _has_dir(f"{subset_dir}/data/{dirname}/"):
            return [f"{subset_dir}/data/{dirname}/"]

    # --- 4: 兜底 —— 直接把 path 当文件用。---
    return [f"{subset_dir}/data/{_normalize(p)}" for p in raws]


def _load_media_index() -> set[str] | None:
    """加载本地媒体索引（仓库内所有 ``data/`` 下的文件路径）。

    为什么需要它：``deformation`` 有 94/200 条样本的 ``path`` 指向**目录**
    （``data/117/`` 里的若干图片），要展开目录就得先知道目录里有什么。

    用 HF 的 ``HfApi.list_repo_files`` 能拿到，但要翻 27k 文件的分页，
    每个样本都等一次不现实；而且 ``HfApi`` 不读 ``HF_ENDPOINT``，
    在只有镜像可达的机器上会直接 ``Network is unreachable``。

    所以改成读随任务一起分发的本地索引（``data/media_index.txt``，
    由 build 脚本从仓库文件清单生成）。索引缺失时返回 ``None``，
    调用方降级为不展开。
    """
    index_path = pathlib.Path(__file__).resolve().parent / "data" / "media_index.txt"
    if not index_path.is_file():
        return None
    try:
        with index_path.open(encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}
    except Exception:  # noqa: BLE001
        return None


_MEDIA_INDEX: set[str] | None = None
_MEDIA_INDEX_LOADED: bool = False


def _media_index() -> set[str] | None:
    """惰性加载并缓存索引。"""
    global _MEDIA_INDEX, _MEDIA_INDEX_LOADED
    if not _MEDIA_INDEX_LOADED:
        _MEDIA_INDEX = _load_media_index()
        _MEDIA_INDEX_LOADED = True
    return _MEDIA_INDEX


def _has_dir(rel_dir: str) -> bool:
    """索引里是否存在这个目录（有任意文件以它开头）。

    用于区分 ``deformation`` 的两种形态：有 ``<id>_frames/`` 就走帧序列，
    否则 ``path`` 是目录名。索引缺失时保守返回 False。
    """
    index = _media_index()
    if not index:
        return False
    prefix = rel_dir.rstrip("/") + "/"
    return any(f.startswith(prefix) for f in index)


def _list_dir_files(rel_dir: str) -> list[str]:
    """展开一个目录下的图片（用于 ``deformation`` 这类 path 指向目录的子类）。

    结果按文件名排序，保证可复现。索引缺失时返回空列表。
    """
    index = _media_index()
    if not index:
        return []
    prefix = f"{rel_dir.rstrip('/')}/"
    return sorted(
        f for f in index if f.startswith(prefix) and _is_image(f) and "/" not in f[len(prefix) :]
    )


def _download(paths: list[str]) -> list[str]:
    """把仓库内相对路径批量下载到本地缓存，返回本地路径列表。

    优先检查项目内 media 目录（离线部署），再走 hf_hub_download。
    ``hf_hub_download`` 自带缓存，重复调用不会重复下载。
    单个文件下载失败会被跳过（上游有个别缺失文件，见模块 docstring）。
    """
    local: list[str] = []
    for p in paths:
        # 优先本地 media 目录
        local_path = _LOCAL_MEDIA / p
        if local_path.exists():
            local.append(str(local_path))
            continue
        # 也检查 HF cache 内的 snapshots（hf_hub_download 的输出结构）。
        # 必须按完整相对路径匹配：帧文件名（如 000026.jpg）在不同目录间大量重名，
        # 只按文件名 glob 会载入别的题的图片。
        hf_cache = _LOCAL_MEDIA / "hf_cache"
        if hf_cache.exists():
            hits = [s / p for s in (hf_cache / "snapshots").glob("*") if (s / p).exists()]
            if hits:
                local.append(str(hits[0]))
                continue
        # 回退：从 HF 下载
        try:
            local.append(
                hf_hub_download(repo_id=_REPO_ID, repo_type=_DATASET_TYPE, filename=p)
            )
        except Exception:  # noqa: BLE001 - 上游缺文件时跳过该张
            continue
    return local


def doc_to_visual(subset: str, doc: Mapping[str, Any]) -> list[str]:
    """返回该样本的视觉输入（本地文件路径列表）。"""
    resolved = _resolve(subset, doc)
    if not resolved:
        return []

    # 目录形态：用本地索引展开后再下载
    if len(resolved) == 1 and resolved[0].endswith("/"):
        files = _list_dir_files(resolved[0].rstrip("/"))
        return _download(files)

    return _download(resolved)


def make_doc_to_visual(subset: str):
    """按子集生成 ``doc_to_visual``，供 yaml 的 ``!function`` 引用。"""

    def _fn(doc: Mapping[str, Any]) -> list[str]:
        return doc_to_visual(subset, doc)

    return _fn


def causalphys_multi_hop_doc_to_visual(doc: Mapping[str, Any]) -> list[str]:
    """多跳池（P0 #23）的 ``doc_to_visual``。

    池子由多个子类的题混合而成（``build_multihop.py`` 从
    spatial_manipulation / temporal_shifting / action_substitution 汇总），
    所以不能绑死单一子集 —— 要按每行的 ``source_subset`` 路由到对应解析规则。
    """
    subset = str(doc.get("source_subset") or "").strip()
    if subset not in _SUBSET_DIR:
        return []
    return doc_to_visual(subset, doc)


def doc_to_text(doc: Mapping[str, Any], lmms_eval_specific_kwargs: Mapping[str, str] | None = None) -> str:
    """题干 + 选项 + 作答指令。选项已内嵌在 ``question`` 里。"""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get(
        "post_prompt", "\nAnswer with the option's letter from the given choices directly."
    )
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def doc_to_target(doc: Mapping[str, Any]) -> str:
    return str(doc["answer"]).strip()


def _extract_choice(text: str, valid: set[str]) -> str | None:
    """从模型输出里抽出选项字母。

    按优先级依次尝试，避免把模型回显的整个选项
    （``"B: The upward force applied by the hand."``）误判成 A。
    """
    if not text:
        return None
    s = text.strip()

    # 1) 严格：整段就是单个字母（可带括号/点/冒号）
    m = re.fullmatch(r"[\(\[]?\s*([A-Za-z])\s*[\)\]]?[\.\:,]?\s*", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 2) "answer is X" / "答案是 X"
    m = re.search(
        r"(?:answer|答案|choice|选项)\s*(?:is|为|是|:)?\s*[\(\[]?\s*([A-Za-z])\b",
        s,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 3) 开头就是 "X:" / "X." / "(X)"
    m = re.match(r"^\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*[\.\:]\s", s)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 4) 兜底：取最后一个孤立的大写字母
    letters = re.findall(r"\b([A-Z])\b", s)
    for ch in reversed(letters):
        if ch in valid:
            return ch
    return None


def _valid_choices(doc: Mapping[str, Any]) -> set[str]:
    """从 ``options`` 字段推断合法选项集合。"""
    found = set(re.findall(r"\b([A-Z])\s*:", str(doc.get("options") or "")))
    return found or {"A", "B", "C", "D"}


def process_results(doc: Mapping[str, Any], results: Sequence[str]) -> dict[str, float]:
    prediction = _extract_choice(results[0] if results else "", _valid_choices(doc))
    target = str(doc["answer"]).strip().upper()
    return {"causalphys_accuracy": 1.0 if prediction == target else 0.0}


def aggregate_results(results: Sequence[float]) -> float:
    return sum(results) / len(results) if results else 0.0


# --- 各子集的 doc_to_visual 绑定（yaml 的 !function 需要具名函数） ---
causalphys_collision_prediction_doc_to_visual = make_doc_to_visual("collision_prediction")
causalphys_fluid_flow_doc_to_visual = make_doc_to_visual("fluid_flow")
causalphys_intention_speculation_doc_to_visual = make_doc_to_visual("intention_speculation")
causalphys_deformation_doc_to_visual = make_doc_to_visual("deformation")
causalphys_action_substitution_doc_to_visual = make_doc_to_visual("action_substitution")
causalphys_spatial_manipulation_doc_to_visual = make_doc_to_visual("spatial_manipulation")
causalphys_temporal_shifting_doc_to_visual = make_doc_to_visual("temporal_shifting")
causalphys_viewpoint_transformation_doc_to_visual = make_doc_to_visual("viewpoint_transformation")
causalphys_mechanics_reasoning_doc_to_visual = make_doc_to_visual("mechanics_reasoning")
causalphys_scene_reconstruction_doc_to_visual = make_doc_to_visual("scene_reconstruction")
causalphys_containability_doc_to_visual = make_doc_to_visual("containability")
causalphys_optics_doc_to_visual = make_doc_to_visual("optics")
