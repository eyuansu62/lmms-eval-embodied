#!/usr/bin/env python3
"""Build JSONL slices for Point-It-Out (PIO) from the official JSON.

s1s2.json must already be at /tmp/pio_s1s2.json, or set PIO_JSON env var.
Images live in data/images_s1s2/ relative to the PIO repo root;
we skip image download here and handle it in utils.py at eval time.

Usage:
    conda activate lmms-eval
    python build_pio_jsonl.py
"""

from __future__ import annotations

import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

PIO_JSON = pathlib.Path(os.environ.get("PIO_JSON", "/tmp/pio_s1s2.json"))

# Each slice: (output_name, filter_fn, capability_tag)
SLICES = [
    ("pio_affordance",
     lambda d: d["s1_or_s2"] == "s2" and d["subclasses"] in ("afd", "contact"),
     "#2 客体可供性感知（s2/afd+contact）"),
    ("pio_attribute_reference",
     lambda d: d["s1_or_s2"] == "s1" and d["subclasses"] in ("restriction", "object part"),
     "#19 属性指代（s1/restriction+object_part）"),
    ("pio_safety",
     lambda d: d["s1_or_s2"] == "s2" and d["subclasses"] == "safety",
     "#36 安全监控（s2/safety）"),
]


def build_slice(name: str, filter_fn, raw: list) -> int:
    items = [d for d in raw if filter_fn(d)]
    out = DATA_DIR / f"{name}.jsonl"
    with out.open("w") as f:
        for idx, item in enumerate(items):
            row = {
                "id": f"{name}/{idx}",
                "image_path": item["image_path"],    # relative to images_s1s2/
                "height": item["height"],
                "width": item["width"],
                "lang": item["lang"],
                "polygon": item["polygon"],           # list of [x,y,...] flat coords
                "s1_or_s2": item["s1_or_s2"],
                "subclasses": item["subclasses"],
                "dataset_name": item.get("dataset_name", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(items)} rows → {out.name}")
    return len(items)


def main() -> None:
    if not PIO_JSON.exists():
        raise SystemExit(f"PIO JSON not found at {PIO_JSON}; download s1s2.json first")
    raw = json.loads(PIO_JSON.read_text())
    print(f"Loaded {len(raw)} PIO items")
    total = 0
    for name, filt, tag in SLICES:
        total += build_slice(name, filt, raw)
    print(f"\nTotal: {total} rows across {len(SLICES)} slices")


if __name__ == "__main__":
    main()
