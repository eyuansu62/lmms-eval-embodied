# SpatialUncertain（实验性，未进入能力表）

数据：[Yuezhangjoslin/spatialuncertain](https://huggingface.co/datasets/Yuezhangjoslin/spatialuncertain)（CC BY 4.0，gated，需先在 HF 页面同意条款）。
场景是 Holodeck 合成的室内渲染图。

```
python lmms_eval/tasks/spatialuncertain/build_spatialuncertain.py
```

| 任务 | 题数 | 内容 |
|---|---|---|
| `spatialuncertain_occ_answerable` | 150 | 清晰视角，应正常作答 |
| `spatialuncertain_occ_abstain` | 150 | 同一问题、同样选项，目标被完全遮挡，应选 “Cannot determine” |
| `spatialuncertain_view_select` | 200 | 五个视角里选出最能判断两个物体真实大小的那个 |

两个 occ 切片按 `pair_id` 配对计分（两份都对才算对），见 `dump_reports._spatialuncertain_paired_scores`。

## 已知问题（2026-10-22 实测，读分前必看）

**view_select 有捷径，不测它想测的能力。** 正确项恒为正对、居中的那张图（原数据的 `view_000`），其余四张是斜拍。

| 条件 | 准确率 |
|---|---|
| 像素左右对称基线：不看题、不用模型，选最对称的那张 | **89.5%** |
| qwen3.8-flash，原题 | 85.0% |
| qwen3.8-flash，题目换成“哪张最正、最左右对称”（不提大小） | 85.5% |
| qwen3.8-flash，题中物体换成场景里没有的东西 | 51.5% |
| 四个模型正式评测 | 64.0–82.5% |
| 随机 | 20% |

问“最对称”与问“判断大小”得分相同，模型选对的题靠的是认构图。引用该分数时请并列对称基线。

**遮挡配对分数贴地板，且标签有噪声。** 四个模型配对得分 3.3–7.3%：遮挡时只有 6–13% 选拒答，54–63% 直接给出清晰图上的答案（清晰图误拒答 1–3%）。“模型几乎不拒答”这个结论成立，但：

- 标为 full_occlusion 的图里，目标有时仍露出一半；
- `size_comparison` 可靠常识作答（如“灶台比垃圾桶大还是小”），不看图也能答对。

更干净的口径是只用 `relative_position` 与 `depth_ordering`，并人工核一遍遮挡图。
