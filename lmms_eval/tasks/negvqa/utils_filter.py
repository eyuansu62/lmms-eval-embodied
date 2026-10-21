"""NegVQA 筛选切片的 ``process_docs``。

把 parquet 里的 1000/7379 例过滤成严格筛出的切片。

为什么单独一个文件
------------------
lmms-eval 的 ``!function`` 是按**文件名 + 函数名**解析的，同一个 yaml 里
``doc_to_text`` 要用 ``utils``、``process_docs`` 要用这个，所以拆成两个模块，
避免把一个文件写得过长。

筛选索引从哪来
--------------
``data/negvqa_natural_scene_<split>_indices.json``，由 ``screen_negvqa.py``
生成（跑一次筛选脚本即可）。yaml 的 ``dataset_path`` 仍指向**原始 parquet**
（图片内嵌其中，不方便单独落盘），靠这里按索引过滤。
"""

from __future__ import annotations

import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent

# 候选索引文件（按优先级）。dev 与 test 是分开筛的。
_INDEX_FILES = [
    _HERE / "data" / "negvqa_natural_scene_indices.json",
    _HERE / "data" / "negvqa_natural_scene_test_indices.json",
    _HERE / "data" / "negvqa_natural_scene_dev_indices.json",
]


def _load_indices() -> set[int]:
    for p in _INDEX_FILES:
        if p.is_file():
            try:
                return set(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
    raise FileNotFoundError(
        "找不到筛选索引。请先跑：\n"
        "  python screen_negvqa.py --input <parquet> --output data/negvqa_natural_scene_<split>.jsonl\n"
        f"已查找：{[str(p) for p in _INDEX_FILES]}"
    )


def process_docs_screened(dataset):
    """只保留 ``screen_negvqa.py`` 筛出的样本。"""
    keep = _load_indices()
    return dataset.filter(lambda ex: int(ex["index"]) in keep)
