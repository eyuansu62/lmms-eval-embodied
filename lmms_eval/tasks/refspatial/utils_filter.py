"""RefSpatial-Expand-Bench 切片的 ``process_docs`` 过滤。

为什么单独一个文件
------------------
lmms-eval 的 ``!function`` 按「文件名 + 函数名」解析。本目录原有的 ``utils.py``
是给内置 task（``BAAI/RefSpatial-Bench``）用的，不能改动；这里的过滤逻辑
另开一个模块，与它并存。

为什么用索引文件而不是本地 jsonl 作 dataset_path
------------------------------------------------
RefSpatial 的每条样本带 ``image``（PIL）与 ``mask``（ndarray）两个二进制字段，
不宜落成 jsonl。所以 ``dataset_path`` 仍指向 HF 上的
``JingkunAn/RefSpatial-Expand-Bench``，由这里按 ``id`` 白名单过滤。

白名单来自 ``build_refspatial_slices.py`` 产出的
``data/refspatial_<name>_indices.json``。

范围说明
--------
文档「子集定位」#19 / #20 明确写了 ``id IN {...}`` 白名单，并附注
"其余 location 需审标后加入"。故这里只取文档指定的 7 条，
**样本量极小，分数仅作定性参考**。
"""

from __future__ import annotations

import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str) -> set[int]:
    p = _HERE / "data" / f"refspatial_{name}_indices.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"找不到 {p}。请先跑：\n"
            "  python build_refspatial_slices.py"
        )
    return set(json.loads(p.read_text(encoding="utf-8")))


# 模块级缓存，避免每个 doc 都读一次文件
_CACHE: dict[str, set[int]] = {}


def _ids(name: str) -> set[int]:
    if name not in _CACHE:
        _CACHE[name] = _load(name)
    return _CACHE[name]


def process_docs_attribute_reference(dataset):
    """#19 属性指代：id IN {1,3,9}（文档指定）。"""
    keep = _ids("attribute_reference")
    return dataset.filter(lambda ex: int(ex["id"]) in keep)


def process_docs_ordinal_reference(dataset):
    """#20 序数指代：id IN {0,2,8,13}（文档指定）。"""
    keep = _ids("ordinal_reference")
    return dataset.filter(lambda ex: int(ex["id"]) in keep)
