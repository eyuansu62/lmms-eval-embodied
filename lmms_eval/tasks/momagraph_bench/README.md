# MomaGraph-Bench

具身智能 VLM 评测 —— 任务规划与图推理。来自 能力体系（CAPABILITIES.md） 的 P0 优先级条目。

- 论文/代码：https://github.com/HybridRobotics/MomaGraph
- 数据：https://huggingface.co/datasets/cheryyunl/MomaGraph-Bench

**4 个 task：3 个按 `type` 拆分的条目 + 1 个全量基线。**

## 已接入的 task

| task | 筛选 | 题数 | 对应 P0 |
|---|---|---|---|
| `momagraph_effect_prediction` | `type == "Effect Prediction"` | 168 | #18 #31 |
| `momagraph_spatial_reasoning` | `type == "Spatial Reasoning"` | 306 | #30 |
| `momagraph_action_sequence` | `type` 以 `Action Reasoning` 开头，或 `== "Sequence Planning"` | 343 | #29 |
| `momagraph_bench` | 全量 | 1315 | 参考基线 |

同一份 parquet，靠 `process_docs` 过滤出不同子集，**不重复存数据**。

## 用法

```bash
conda activate lmms-eval
cd <repo-root>

CUDA_VISIBLE_DEVICES=4 python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks momagraph_effect_prediction,momagraph_spatial_reasoning,momagraph_action_sequence \
  --batch_size 1 --output_path ./results
```

跑之前先确认有空闲 GPU —— 这台机器上 0/1/2/3/6/7 常年被占满，
4 和 5 通常空闲。

## 目录内容

```
momagraph_bench/
├── momagraph_bench.yaml                  # 全量基线
├── momagraph_effect_prediction.yaml      # P0 #18 #31
├── momagraph_spatial_reasoning.yaml      # P0 #30
├── momagraph_action_sequence.yaml        # P0 #29
├── utils.py
├── README.md
└── data/
    └── train-*.parquet                   # 3 个文件, 1.1 GB
```

## 数据形态

和 CausalPhys 最大的不同：**图片以 PIL 对象内嵌在 parquet 里**，
不需要任何网络请求或额外媒体下载。parquet 拷进仓库后完全离线可跑。

字段：

| 字段 | 说明 |
|---|---|
| `image` | PIL 图片对象 |
| `question` | 题干（**不含**选项） |
| `options` | `list[str]`，选项自带 `"A) ..."` 前缀 |
| `correct_answer` | `"(C)"` 这种带括号的字母 |
| `idx` | 题号 |
| `scene` | 场景标识，如 `lamp_collection/421267/lamp` |
| `task_instruction` | 任务描述，如 `Power on the lamp.` |
| `type` | 题型，**拆分的依据** |

### `type` 的 14 个取值

| type | 题数 | 归属 task |
|---|---|---|
| Spatial Reasoning | 306 | `spatial_reasoning` |
| Object Selection | 273 | 仅全量 |
| Action Reasoning | 265 | `action_sequence` |
| Affordance Reasoning | 218 | 仅全量 |
| Effect Prediction | 168 | `effect_prediction` |
| Sequence Planning | 75 | `action_sequence` |
| Dynamic Verification | 3 | 仅全量 |
| Action Reasoning (overall plan) | 1 | `action_sequence` |
| Action Reasoning (Scene 1 → Scene 2) | 1 | `action_sequence` |
| Action Reasoning (Scene 1 → Scene 3) | 1 | `action_sequence` |
| Affordance Reasoning (power) | 1 | 仅全量 |
| Object Selection (cold items) | 1 | 仅全量 |
| Object Selection (Goal state planning) | 1 | 仅全量 |
| Graph Inference | 1 | 仅全量 |

## 拆分依据与两处偏差

### #30 的 reachability 子筛 —— 做不到

文档要求"`type==Spatial Reasoning` 候选；**再筛 question 涉及 reachability 的题**"，
并自注"已查看的示例多为相对方位；**尚未锁定可达性 ID**"。

本轮实测确认数据里基本没有这类题：

```
306 题中 301 条是 "What is the spatial relationship between X and Y?" 模板
其余 5 条是普通相对方位（如 "Where is the microwave relative to the refrigerator?"）

关键词全量检索（题干）：reach=0  reachab=0  arm=0  grasp=0
                       able to=0  can the=0  within=0  accessible=0  feasib=0
关键词检索（选项）：    reach=2  其余全 0
```

所以只按 `type` 筛，不做子筛。若后续拿到可达性 ID 清单，在
`process_docs_spatial_reasoning` 里加一层过滤即可。

### #29 的映射 —— 按已核验口径取

文档说"发布 `type` 与 Action Sequence Reasoning 的对应关系**待核验**；
**不把** Action Reasoning / Object Selection 整体等同"。

本轮按已核验的口径取 `Action Reasoning`（含 3 个 `(xxx)` 变体）
+ `Sequence Planning` = 343 题，**不含** Object Selection（274 题留在全量）。

用前缀匹配而非精确相等，是为了带上 `Action Reasoning (overall plan)`
这类变体；若后续核验结果不同，改 `utils._ACTION_SEQUENCE_PREFIX` 即可。

## 判分

`utils._extract_choice` 按优先级抽取答案字母，顺序与 CausalPhys 的实现一致 ——
严格匹配优先于兜底，避免把模型回显的整个选项（`"C) Insert the plug"`）
误判成 A。

已用 16 组构造样例验证，全部通过（含 `"C)"`、`"(C)"`、`"答案是 C"`、
`"I think it's D"`、空输出、无选项等边界）。

**答案分布偏斜**：全量 `B` 占 642/1315（49%），`D` 只有 60。
各 `type` 内部也不同（如 `Affordance Reasoning` 只有 A/B/C，没有 D）。
解读准确率时要留意基线偏移。好消息是**答案没有按序排列**
（1315 题里切换 891 次），不需要像 CausalPhys 那样打散。

## 验证状态

- ✅ 4/4 task 注册
- ✅ 拆分题数精确匹配（168 / 306 / 343 / 1315）
- ✅ 判分逻辑 16/16 构造样例
- ✅ 三个拆分 task 端到端冒烟

冒烟成绩（Qwen2.5-VL-3B-Instruct，每 task 6 样本，**限样本仅供参考**）：

| task | acc |
|---|---|
| `momagraph_effect_prediction` | 1.0000 |
| `momagraph_action_sequence` | 0.6667 |
| `momagraph_spatial_reasoning` | 0.5000 |

每 task 仅 6 条，误差极大，只用于验证管线通畅。

## 待办

- [ ] 全量评测（不用 `--limit`）
- [ ] 按 `type` 或 `scene` 分解报告，便于定位薄弱环节
- [ ] 核验 #29 的 type↔能力映射、#30 的可达性 ID（都需要外部信息）
