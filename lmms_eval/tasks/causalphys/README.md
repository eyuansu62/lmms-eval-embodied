# CausalPhys

具身智能 VLM 评测 —— 因果与物理推理。来自 能力体系（CAPABILITIES.md） 的 P0 优先级条目。

- 论文：https://arxiv.org/html/2606.05966v1
- 代码：https://github.com/haorentang/CausalPhys
- 数据：https://huggingface.co/datasets/haorentang/causalphys

**9 个 task：8 个子类（文档显式点名）+ 1 个按因果图筛出的多跳子集。**

> **范围说明**：只接入 能力体系（CAPABILITIES.md）**显式点名**的子类。
> 接入过程经过两轮收窄，最终从最初的 16 个子类/3062 题降到 8 个/977 题：
> 1. 移除 `Goal_orientation` 全部 4 个（文档任何条目都未提该目录）
> 2. 移除 4 个只出现在 #21 通配池里的子类（不作为独立 task，只经筛选进入多跳）
> 详见下文「与文档的对应关系」。

## 已接入的 task

| task | 子集 | 题数 | 对应 P0 能力 |
|---|---|---|---|
| `causalphys_mechanics_reasoning` | Perception/Mechanics_Reasoning | **153** | #24 归因理解 |
| `causalphys_collision_prediction` | Anticipation/Collision_Prediction | 300 | #37 轨迹理解与预测 |
| `causalphys_fluid_flow` | Anticipation/Fluid_Flow | 200 | #41 刚体/流体/柔体 |
| `causalphys_deformation` | Anticipation/deformation | 200 | #41 刚体/流体/柔体 |
| `causalphys_action_substitution` | Intervention/action_substitution | 99 | #23 干预推理 |
| `causalphys_spatial_manipulation` | Intervention/spatial_manipulation | 151 | #23 干预推理 |
| `causalphys_temporal_shifting` | Intervention/temporal_shifting | 149 | #23 干预推理 |
| `causalphys_containability` | Perception/containability | 201 | #11 拓扑关系 |
| | **小计（8 个子类）** | **1453** | |
| `causalphys_multi_hop` | 因果图链长 ≥3 条边 | 168 | #21 因果链推理 |
| | **合计** | **1621** | |

> 注：`mechanics_reasoning` 的 153 是**筛选后**的题数（整目录 298 题）。文档预期
> "约 100 题可作归因种子"，实测筛出 153 —— 差异见下文。

## 与文档的对应关系

能力体系（CAPABILITIES.md）里，CausalPhys 共 6 条条目，分两类写法：

### ① 显式点名的目录（5 条，取整目录）

| 条目 | 能力 | 点名目录 | 本目录的 task |
|---|---|---|---|
| #11 | 拓扑关系 | `Perception/containability` | `containability` |
| #23 | 干预推理 | `Intervention/spatial_manipulation` + `temporal_shifting` + `action_substitution` | 3 个 task |
| #24 | 归因理解 | `Perception/Mechanics_Reasoning` | `mechanics_reasoning`（**需筛选**） |
| #37 | 轨迹理解与预测 | `Anticipation/Collision_Prediction` | `collision_prediction` |
| #41 | 刚体/流体/柔体 | `Anticipation/deformation` + `Fluid_Flow` | 2 个 task |

这 5 条覆盖 8 个子类。

### ② #21 的通配池（1 条，需筛选）

| 条目 | 能力 | 选取条件 |
|---|---|---|
| #21 | 因果链推理 | 候选池 `Perception/**` + `Anticipation/**` + `Intervention/**`；**保留需跨 ≥2 个因果环节才能作答的题** |

通配池覆盖全部 12 个子类（8 个点名 + 4 个未点名）。筛选后产出 168 题，
即 `causalphys_multi_hop`。

**那 4 个未点名的子类不作为独立 task** —— 它们唯一的入口是 #21 的通配池，
而 #21 要求筛选。实测它们筛完只剩 5 题（全在 `scene_reconstruction`），
把它们当全量独立接入才是越界。数据保留在 `data/` 下供 `build_multihop.py` 重建池子。

| 未点名子类 | 题数 | #21 筛选后 |
|---|---|---|
| Anticipation/Intention_Speculation | 200 | 0 |
| Intervention/viewpoint_transformation | 174 | 0 |
| Perception/Scene_Reconstruction | 200 | 5 |
| Perception/optics | 252 | 0 |

### 两处未按文档预期实现

**#24 筛出 153 题，文档预期约 100。** 判据（`build_attribution.py`）是
"题干明确询问原因"，命中 `What is the primary cause`（131）、
`What is the direct cause`（14）、`What caused`（3）、`Why did/does`（3）等句式。
153 与 100 的差额不是判据松 —— 单是 `What is the primary cause` 模板就占 131 题，
且 153 题对应 153 个不同媒体文件，无重复可去。文档那个数字标注为"约"、
未给判据，故保留可复跑的 153。

**#11 / #23 / #37 / #41 未做进一步筛选。** 这四条的"尚需处理"都写了
"只承担原能力中的部分"之类，但**未给可机械执行的判据**，故按"取上述目录"整目录接入。


## 与文档的对应关系

能力体系（CAPABILITIES.md）用的是两种写法，本目录两者都遵守：

**① 显式点名的 8 个目录**（各条 P0 条目直接写全路径）：

| P0 条目 | 点名目录 |
|---|---|
| #24 | `Perception/Mechanics_Reasoning` |
| #37 | `Anticipation/Collision_Prediction` |
| #41 | `Anticipation/deformation` + `Anticipation/Fluid_Flow` |
| #23 | `Intervention/spatial_manipulation` + `temporal_shifting` + `action_substitution` |
| #11 | `Perception/containability` |

**② #23 的通配前缀**：

```
Perception/**/annotations/*.json
Anticipation/**/annotations/*.json
Intervention/**/annotations/*.json
```

按此纳入 `Perception` 全部 4 个、`Anticipation` 全部 4 个、`Intervention` 全部 4 个。

**结果：12 个子类。** `Goal_orientation` 的 4 个目录在两种写法里都没出现，
故不在范围内。

> 注：#23 的选取条件除了通配前缀，还要求"保留需跨 ≥2 个因果环节才能作答的题"。
> 所以严格讲 `intention_speculation` / `viewpoint_transformation` /
> `scene_reconstruction` / `optics` 这 4 个应该**经多跳筛选后**才计入。
> 实测它们在 ≥3 条边阈值下只剩 5 题（全在 `scene_reconstruction`，
> 详见「多跳子集」一节），所以这里按**全量**接入并在计数中标明 ——
> 若要严格贴合 #23，可只保留那 5 题。


## 多跳子集（P0 #21）

文档 #23 要求从 `Perception/**` + `Anticipation/**` + `Intervention/**` 中
"审查 question 与 graph，**保留需跨 ≥2 个因果环节才能作答的题**"。

**阈值取 ≥3 条因果边**，而不是 ≥2 条。理由见下方「为什么是 3 条边」。

| 来源子集 | 总数 | 入选 | 占比 |
|---|---|---|---|
| Intervention/spatial_manipulation | 151 | 80 | 53.0% |
| Intervention/temporal_shifting | 149 | 83 | 55.7% |
| Perception/scene_reconstruction | 200 | 5 | 2.5% |
| Anticipation/intention_speculation | 200 | 0 | 0.0% |
| Intervention/action_substitution | 99 | 0 | 0.0% |
| Intervention/viewpoint_transformation | 174 | 0 | 0.0% |
| Perception/optics | 252 | 0 | 0.0% |
| **合计** | **1225** | **168** | 13.7% |

入选题的链长分布：`3 条边` 139 题、`4 条边` 27 题、`5 条边` 2 题。

**7 个候选子类里 5 个筛到 0。** 原因一致：它们的 graph 最长因果链只有 1–2 条边
（`intention_speculation`、`optics` 全为 1 条；`viewpoint_transformation` 全为 2 条；
`action_substitution` 最长 2 条）。只有 `spatial_manipulation`、`temporal_shifting`
和 `scene_reconstruction` 能过阈值。

生成方式：`python build_multihop.py`（输出 `data/multihop_pool.jsonl`）。

### 链长怎么算

graph 的节点有三种 `type`：

| type | 计入链长？ | 说明 |
|---|---|---|
| `event` | ✅ | 动作/事件 |
| `attribute` | ✅ | 状态/属性 |
| `object` | ❌ | 场景参与者。会出现在链条上，但不是因果**环节** |

只保留 `event`/`attribute` 之间的边，在这个 DAG 上求最长路径的边数。
示例（`spatial_manipulation` id=048）：

```
obj   → e1     e1 → a1     a1 → a2      ← 因果链：e1→a1→a2，2 条边
cloth → a1                              ← cloth 是 object，不计
```

### 为什么是 3 条边

文档自己提醒：

> 仅 graph 有长度 2 路径不足以证明问题要求多跳。

"长度 2 路径"正是 `A→B→C`，也就是 **≥2 条边** 的口径。若取 2 条边，
会放进大量结构上有传递、但题目并不要求多跳推理的样本（299/1225 条，
比 3 条边口径多出近一倍）。取 3 条边意味着链上存在**中间环节**，
更贴近"跨多个环节、识别间接效果和连锁反应"的原意。

## 用法

```bash
conda activate lmms-eval
cd <repo-root>

# 指定要跑的 task（此处举例两个）
CUDA_VISIBLE_DEVICES=4 python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks causalphys_mechanics_reasoning,causalphys_optics \
  --batch_size 1 --output_path ./results

# 也可以用 tag 一次性指定全部（见 yaml 的 metadata）
```

跑之前先确认有空闲 GPU —— 这台机器上 0/1/2/3/6/7 常年被占满，
4 和 5 通常空闲。

## 目录内容

```
causalphys/
├── build_causalphys_jsonl.py                  # 归并脚本（见下）
├── gen_yamls.py                               # 生成 12 个子类的 yaml
├── causalphys_<subset>.yaml  × 12
├── utils.py                                   # 媒体解析 + 判分
├── README.md
└── data/
    ├── <subset>_annotations/                  # 上游原始 annotation（文档范围内 2424 个 json）
    ├── <subset>.jsonl                         # 归并产物，yaml 实际读这个
    └── media_index.txt                        # 媒体路径索引，23726 条
```

改了上游数据或加了子集后，重跑：

```bash
python build_causalphys_jsonl.py   # 重建 jsonl + 索引
python gen_yamls.py                # 重建 yaml
```

## 为什么要有 build 脚本这一步

CausalPhys 是**文件式**数据集：每个样本一个 json。两个问题让
`datasets` 无法直接读：

1. **schema 不一致** —— 不同样本的 `graph.nodes` 有的带 `text` 字段有的不带，
   列类型推断直接崩：
   ```
   DatasetGenerationError: An error occurred while generating the dataset
   ... because column names don't match
   ```
2. **仓库太大** —— 27k 文件，让 `load_dataset` 遍历整个 repo 不现实。

所以先把 annotation 归并成单个 JSONL，只保留任务需要的字段，
`graph` 序列化成字符串（留给后续 judge 评分用）。

### 打散（重要）

上游数据**按答案排序**。以 `collision_prediction` 为例：前 150 条全 `A`、
后 150 条全 `B`，切换点唯一。若原样保留，任何 `--limit N (N≤150)` 都只取到
单一标签，小样本调试的分数毫无意义 —— 实测 `--limit 8` 得到 0/8，
既不是判分 bug 也不是模型不会，纯粹是切片伪影。

归并时用固定种子（`SHUFFLE_SEED = 20260923`）打散，保证可复现的同时
让任意前缀都覆盖多种标签。

## 数据形态要点

### 文件名 ≠ JSON 里的 `id` 字段

最容易踩的坑：

| 文件名 | 内部 `id` | `path` |
|---|---|---|
| `100.json` | `"100"` | `MechanicsReasoning100.webm` |
| `396.json` | `"096"` | `MechanicsReasoning096.webm` |
| `201.json` | `"0001"` | `001.mp4` |

必须读 `id` 字段。官方 loader 也是用 `ann["id"]` 拼帧目录。

### `path` vs `paths`

46 个 `optics` 样本只写了 `paths`（复数），其余写 `path`。
官方 loader 依次尝试 `paths` → `path`；漏掉这一步会把这 46 题当成
媒体缺失。归并脚本统一成 `media_paths` 字段，下游只读一个字段。

### 四种媒体形态

| 形态 | 子类 | 解析方式 |
|---|---|---|
| 帧序列 | collision_prediction / fluid_flow / intention_speculation / mechanics_reasoning / trajectory / spatial_manipulation / **deformation（部分）** | `data/<id>_frames/*.jpg`（固定 5 帧） |
| 单图 | object_relocation / behaviour_selection / tool_selection / containability / optics / scene_reconstruction | `data/<path>` |
| 多图列表 | viewpoint_transformation / action_substitution / temporal_shifting | `data/<path_i>`（每条 2–5 张） |
| 目录 | **deformation（94/200 条）** | `data/<path>/` 内的图片 |

`deformation` 是**混合形态**，同一子类下两种并存：
```
id=004 -> data/004/0028.png, 0033.png        （目录，无 _frames）
id=117 -> data/117.mp4 + data/117_frames/*.jpg（帧序列）
```
所以解析时不按扩展名猜，而是**查索引**判断 `<id>_frames/` 是否真实存在。

### 路径归一化

上游有 2 个样本（`spatial_manipulation` / `temporal_shifting` 的 `109.json`）
把**全角右括号** `）`(U+FF09) 写进了文件名，仓库里实际是半角 `)`。
用 NFKC 归一化还原。

这 2 个样本上游**本身**就缺第 3 张图（5 张只有 4 张），属上游数据问题。

### 为什么有 `media_index.txt`

`deformation` 的目录形态要展开目录才知道里面有什么。用 HF 的
`HfApi.list_repo_files` 能拿到，但要翻 27k 文件的分页，每个样本都等一次
不现实；而且 **`HfApi` 不读 `HF_ENDPOINT`**（那个变量只对
`hf_hub_download` 之类生效），在只有镜像可达的机器上会直接
`Network is unreachable`。

所以改成读随任务分发的本地索引。

### 媒体按需下载

`hf_hub_download` 走 `HF_ENDPOINT` 镜像并缓存，不预拉全量。
实测跑 64 个样本（16 子类 × 4）只产生约 20 MB 缓存。

完整实测记录见同目录的 `DATA_SHAPE.md`。

## 判分

单选题，选项内嵌在 `question` 字符串里（` A: ... B: ...`），
没有独立 options 字段。`utils._extract_choice` 按优先级抽取答案字母，
优先级顺序是为了避免把模型回显的整个选项（`"B: The upward force..."`）
误判成 A。

已用 15 组构造样例验证，全部通过。

## 验证状态

- ✅ 17/17 task 注册（16 子类 + 1 多跳）
- ✅ **全量媒体解析：2424/2424 个样本，零空输入**
- ✅ 多跳池路由解析：168/168（142 帧序列 + 26 图片）
- ✅ `deformation` 两种形态各自走对分支
- ✅ 判分逻辑 15/15 构造样例
- ✅ 端到端冒烟（16 子类 × 4 样本 = 64 条），判分与真值一致
- ✅ `causalphys_multi_hop` 冒烟（8 样本）

冒烟成绩（Qwen2.5-VL-3B-Instruct，每子类 4 样本，**限样本仅供参考**）：

| task | acc | | task | acc |
|---|---|---|---|---|
| spatial_manipulation | 1.00 | | trajectory | 0.50 |
| tool_selection | 1.00 | | mechanics_reasoning | 0.50 |
| deformation | 0.75 | | optics | 0.50 |
| intention_speculation | 0.75 | | temporal_shifting | 0.50 |
| scene_reconstruction | 0.75 | | action_substitution | 0.25 |
| behaviour_selection | 0.50 | | containability | 0.25 |
| collision_prediction | 0.50 | | object_relocation | 0.00 |
| fluid_flow | 0.50 | | viewpoint_transformation | 0.50 |

每子类只有 4 条，误差极大，仅用于验证管线通畅。

## 待办

- [ ] 全量评测（不用 `--limit`）
- [ ] `graph` 字段的 judge 解释性评分（可选，官方 `--use_judge`）
- [ ] 给 12 个子类加一个 task group / tag，方便一次调用
- [ ] #23 的筛题标准目前只看图结构；文档提醒"仅 graph 有长度 2 路径不足以
      证明问题要求多跳" —— 若要更严格，可在 `build_multihop.py` 里叠加
      question 文本审查
- [ ] 确认 7 个文档外子类是否需要保留（见上文说明）
