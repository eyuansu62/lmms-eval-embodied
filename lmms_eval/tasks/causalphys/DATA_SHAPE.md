# CausalPhys 数据形态实测记录

来源：`haorentang/causalphys`（HF，未 gated，可匿名下载）。
镜像地址 `https://huggingface.co/datasets/haorentang/causalphys/resolve/main/<path>`。
**注意：`curl` 必须带 `-L`**，否则只拿到 307 重定向（正文是 "Temporary Redirect"）。

> 27,142 个文件。全量清单可用 git metadata 一次取到，不必逐个探：
> ```
> git clone --depth 1 --no-checkout https://huggingface.co/datasets/haorentang/causalphys
> cd causalphys && git ls-tree -r --name-only HEAD > files.txt
> ```
> 只拉某几个子集的 annotation（省流量）：
> ```
> git ls-tree -r --name-only HEAD | grep -E "(Sub1|Sub2)/annotations/.*\.json$" > want.txt
> git checkout HEAD -- $(cat want.txt | tr '\n' ' ')
> ```

## 目录结构

```
<Category>/<Subcategory>/annotations/*.json     ← 题目
<Category>/<Subcategory>/data/                  ← 媒体
```

顶层 4 个 Category，16 个 Subcategory：

| Category | Subcategory | 文件数 |
|---|---|---|
| Anticipation | Collision_Prediction | 2100 |
| | Fluid_Flow | 1400 |
| | Intention_Speculation | 1400 |
| | deformation | 1106 |
| Goal_orientation | Object_Relocation | 381 |
| | behaviour_selection | 687 |
| | tool_selection | 282 |
| | trajectory | 927 |
| Intervention | action_substitution | 603 |
| | spatial_manipulation | 992 |
| | temporal_shifting | 984 |
| | viewpoint_transformation | 4829 |
| Perception | Mechanics_Reasoning | 2186 |
| | Scene_Reconstruction | 430 |
| | containability | 2201 |
| | optics | 6630 |

## annotation JSON 字段（所有子类一致）

```json
{
  "id": "100",
  "question": "What is the primary cause ...? A: ... B: ... C: ...",
  "ground_truth_answer": "B",
  "category": "Perception",
  "sub_category": "Mechanics Reasoning",
  "path": "MechanicsReasoning100.webm",
  "graph": { "nodes": [...], "edges": [...] }
}
```

- 单选题**内嵌在 `question` 字符串里**（` A: ... B: ... `），没有独立 options 字段
- `ground_truth_answer` 是选项字母
- `graph` 是因果图（nodes/edges），**不可喂给模型**；仅在 judge 解释性评分时用

## ⚠️ 关键：文件名 ≠ 内部 id 字段

这是最容易踩的坑。`annotations/` 下的**文件名**和 JSON 里的 `id` 字段**不一致**：

| 文件名 | 内部 `id` | `path` | 真实存在 |
|---|---|---|---|
| `100.json` | `"100"` | `MechanicsReasoning100.webm` | ✓ |
| `195.json` | `"195"` | `MechanicsReasoning195.webm` | ✓ |
| `396.json` | `"096"` | `MechanicsReasoning096.webm` | ✓ |
| `201.json` | `"0001"` | `001.mp4` | ✓ |

**必须读 `id` 字段，不能拿文件名当 id。** 官方 loader 也是用 `ann["id"]` 拼帧目录。

## 媒体解析：两种方式，实测 100% 可解析

对 `Perception/Mechanics_Reasoning`(298) 和 `Anticipation/Collision_Prediction`(300)
共 598 个 annotation 做了全量校验：

```
path 字段存在        : 598 / 598   (100%)
<id>_frames 目录存在 : 598 / 598   (100%)
两者都无             : 0
```

所以两种取媒体方式都可靠，任选：

1. **`data/<path>`** —— 原始视频（`.webm` / `.mp4`），由 `path` 字段直接给出
2. **`data/<id>_frames/`** —— 预抽好的 5 帧 jpg，`<id>` 取 JSON 的 `id` 字段

帧目录命名有三种补零形式，按 `id` 依次尝试即可覆盖全部：
`{id}_frames` → `{int(id):04d}_frames` → `{int(id):03d}_frames`

每样本固定 5 帧，命名 `<path词干>_f000.jpg` … `_f004.jpg`：

```
Perception/Mechanics_Reasoning/data/0001_frames/001_f000.jpg        (约 61 KB)
Anticipation/Collision_Prediction/data/000_frames/CollisionPrediction000_f000.jpg  (约 178 KB)
```

`data/` 下还有 `*_mermaid.mmd`，是因果图的可视化版本（内容即 `graph` 字段），**不是题目素材**。

## 官方 loader 的行为

`dataset/causal_vl.py`（GitHub `haorentang/CausalPhys`）：

- 遍历 `os.path.join(root, "*", "*", "annotations", "*.json")`
- `_resolve_media` 依次尝试 `paths` → `path` 字段
- 路径不存在则**抛异常并 `continue` 跳过该样本**，不中断
- 视频优先找 `<data_dir>/<id>_frames/`，有帧直接用，没有才用 cv2/imageio 抽帧
- 默认 `num_video_frames=5`、`max_images=12`

## 官方评测脚本

`evaluation/evaluate_responses.py`、`generate_responses.py`、`analyze_results.py`；
`evaluation/api_evaluators.py`、`qwen_evaluator.py`、`llama_evaluator.py`。
`--use_judge` 为可选解释性评分（需单独配置模型/API）。
多选题答案比对是主要指标，ACC 可本地复现。

## P0 映射

| P0 条目 | 能力 | 对应子集 |
|---|---|---|
| #15 | 因果链推理 | `Intervention/*`、`Anticipation/*` |
| #16 | 归因理解 | `Perception/Mechanics_Reasoning` |
| #27 | 轨迹理解与预测 | `Anticipation/Collision_Prediction` |
| #28 | 单体力学效果 | `Anticipation/deformation`、`Fluid_Flow` |

## 已完成

- [x] 全量文件清单
- [x] annotation 字段结构
- [x] 文件名 ≠ 内部 id 的坑
- [x] 媒体解析双路径（path / `<id>_frames`）100% 验证

## 待办

- [ ] 其余 14 个子类的媒体形态（`containability` 是单图+检测框，与帧序列不同）
- [ ] 决定媒体策略：全量下载 vs 运行时按需拉取
- [ ] 写 lmms-eval task（先做 Mechanics_Reasoning 单子集 + 小样本）
