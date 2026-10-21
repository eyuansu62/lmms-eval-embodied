# 能力表读表说明

`summary_capability.csv`（明细，一个来源一行）与 `summary_capability_agg.csv`（合并，一个能力一行或按口径拆行）
由 `dump_reports.py` 重算，映射在 `map_subtask_to_capability.py`。能力编号 = 能力体系（CAPABILITIES.md）行序。
本文记录**读分时必须知道的口径与局限**，2026-10-20 整理。

## 1. 口径与合并规则

| 口径列 | 含义 | 能否与 A/B 档全量平均 |
|---|---|---|
| A/B档·全量 | 标准图像/视频问答，按 benchmark 自然分布 | — |
| C档·均衡 / C档·题型全量 | C 档按类别**均衡采样**的准确率，不是真实分布水平 | ✗（单独成行） |
| B档·文本化 | ASIMOV 决策题，官方是 tool-call，这里改成文本作答 | ✗ |
| B档·纯文本 | EAI，原协议就是符号/文本接口，不看图 | ✗ |
| B档·二分类 | IntPhys 2，全答 Yes 即 50% | ✗ |
| B档·judge / 帧序列 / 非循环 | AQuA（LLM judge）、OpenEQA（16 帧）、MM-UPD（不做官方循环评测） | ✗ |

- **合并 = 逐题口径**（2026-10-20 起）：一项能力的分数 = 该能力下所有题的逐题平均，即「答对题数 / 总题数」
  （连续分的题取逐题均值）。子任务合成来源、来源合成能力，两层都按题数加权。此前是等权，小来源占不成比例的权重
  （#19 refspatial 3 题与 PIO 95 题各占一半）；切换后 #19 #25 #16 #27 #33 #8 变化较大，与 10-20 前不可比。
- **唯一例外 #46**：readable 31 / unreadable 131 两类等权（逐题口径下全答「看不清」即 81%）。
  2026-10-21 起 #46 是**判断口径**：看得清的图答了数值即算对（不看读数准不准），看不清的图答 unreadable 算对。
- vsibench 子任务也是逐题口径：rel_direction 的 easy/medium/hard 并成一个池子算，不是官方的三档等权
  （官方口径仍在 `summary_A_B.csv` 的 vsibench 总分里）。
- **只合并覆盖全部 10 个模型的来源**：缺模型的来源只留在明细表。
- #9 的三个来源 能力体系 要求分开计分，**不合并**。

- **ERIQ 进能力表一律「宽松抽取 + Yes/No 归一」口径**（子任务名带 `_lenient` / `_yn`）：官方严格口径只认回复开头的字母，
  先分析再给字母、或照提示词回 Yes/No 的模型会被整片判错（qwen3.8-flash #32 严格 7.25 / 宽松 68.12；deepseek #33 严格 29.4）。
  ERIQ 主表（summary_A_B、summary_subtasks 的原子任务名）仍是官方严格口径，三口径并排见 `summary_eriq_calibers.csv`。
- **#35 拆两行**：恒常性（IntPhys 2 240 + CoreCognition 98）与守恒（CoreCognition 71），能力体系 要求分别计分。

## 2. 判分模型（judge）

| 来源 | judge | 备注 |
|---|---|---|
| RoboBench 规划题（#14 #28） | GPT-4o（`openai/gpt-4o-2024-08-06`，OpenRouter 经代理） | 与官方一致。2026-10-20 前是 qwen3.8-max，偏松，已全部重判 |
| AQuA（#47）、OpenEQA（#6） | qwen3.8-max | qwen3.8-max 本身是被测模型，可能有自评偏差 |
| EAI（#16 #31） | 官方判分（#31 现场跑 VirtualHome 模拟器） | 无 LLM judge |

## 3. 逐项已知局限

### 3.1 来源不对口 / 有捷径（读分最需要注意）

| # | 能力 | 来源 | 问题 |
|---|---|---|---|
| 18 | 条件指令理解 | MomaGraph effect_prediction | 是「插上插头灯会亮」式常识因果，**不看图 92.9%**（有图 91.1%），10 个模型 88–96，基本无区分度。能力体系 主测（BDDL if-then-else 派生题）无现成数据，暂无替代 |
| 29 | 指令顺序理解 | MomaGraph action_sequence | 测**常识顺序**，不是「遵守用户指定的先后」；**不看图 56.9%**（有图 62.4%，随机 25%），答案 81% 是 B/C。EAI、LexiCon、Plan-Bench 均查过，都不对口 |

### 3.2 接近随机 / 盲猜基线高（必须对照基线读）

| # | 能力 | 来源 | 基线 |
|---|---|---|---|
| 22 24 27 | EgoTaskQA 各切片 | EgoTaskQA 是非题 | 模型普遍 45–65；#27 盲测 52.5% vs 有视频 63.5%。是**题难**（抽象属性 + 多机位剪辑），不是标注坏 |
| 13 | 事件顺序理解 | TVBench | 不看视频：action_sequence 54.5（随机 50）、egocentric_sequence 59.0（随机 25；只有 25 套实验流程，常识能排除部分干扰项）。有视频高 15–40 分 |
| 35 38 | 物体恒常性 / 运动连续性 | IntPhys 2 | 全答 Yes = 50% |
| 33 | 失败检测 | ERIQ success_detection（197 题，逐题口径下占 #33 的 71%） | 全选 B = 59.4% |
| 35 | 物体恒常性 · 守恒 | CoreCognition conservation（是非题已 YES/NO 等量） | 全答 YES = 50%；qwen 系列 90–97，可能接近天花板 |
| 45 | 场景完整判断 | CrossPoint 可见性题 | 全选同一答案 ≈ 53.5% |
| 37 | 轨迹理解与预测 · CausalPhys 碰撞题部分 | CausalPhys 碰撞预测 148 题 | 是/否各半，随机 50，**58 以下与随机无差别**。只有 doubao、glm 明显高于随机；#37 合并分主要由 VLM4D 300 题拉开 |
| 1 / 3 | ASIMOV vla_feasibility / constraints | ASIMOV | 盲猜 50 / 53.2 |

### 3.3 天花板（区分不出模型）

| # | 能力 | 来源 | 10 个模型范围 |
|---|---|---|---|
| 18 | 条件指令理解 | MomaGraph | 88–96（见 3.1） |
| 26 | 物理知识 | PhyX Mechanics 接触力学 39 题 | 85–100（3B 基线 20.5）；一题 ≈ 2.6 分 |
| 3 | 本体客体交互关系 | ASIMOV constraints | 82–91 |
| 31 | 约束满足检查 | EAI 动作序列 | 76–88，失败几乎都是「缺步骤」 |

### 3.4 题量小（单题波动大）

| # | 来源 | 题数 | 一题 ≈ |
|---|---|---|---|
| 20 | PIO 18 + refspatial 4 | 22 | 一题 ≈ 4.5 分 |
| 4 | robobench visual_attribute | 31 | 3 分 |
| 32 | ERIQ task_progress | 69 | 唯一来源 |

#19 refspatial 只有 3 题，逐题口径下只占 #19 的 3%，已不影响分数。

### 3.5 替补或缺层

| # | 能力 | 能力体系 原定 | 实际 |
|---|---|---|---|
| 15 | 指令歧义识别 | Ambi3D | ASIMOV uncertainty |
| 46 | 低视觉质量判断 | RoboStressBench（非商用） | ASIMOV 读表题 |
| 22 | 反事实推理 | CausalVQA | EgoTaskQA counterfactual |
| 17 | 否定理解 | NegBench | NegVQA |
| 47 | 知识检索与求助 | 三层 | 只有自我检视（MM-UPD）与询问（AQuA），**检索层没有覆盖** |

### 3.6 个别模型的特例

| 模型 | 位置 | 情况 |
|---|---|---|
| gemma-4-26b | vsibench 数值题（#9 绝对距离 0.35、#44 0.00） | **拒答**：尺寸估计 265 题里 258 题回「看不清，无法回答」。是模型行为，不是判分问题 |
| mimo-v2.6-flash | #14（23 题）、#28（44 题） | 空回复：推理用满 65536 输出上限，按规则计 0、不重跑 |
| deepseek-v4.1-flash | ERIQ | 判断题回 `Yes`/`No` 而非字母，官方严格口径全判错；#27 的 ERIQ 判断题子集已改用 Yes/No 归一口径（27.9 → 69.1），ERIQ 主表仍是严格口径 |

### 3.7 C 档

#7 #10 #26 #43 #45（及 #4 的 epic 行）来自 C 档，类别均衡采样，**不与 A/B 档平均**。
#7 只剩 GroundDetection / AffordanceRegion 两类（FeasiblePath 已撤，见第 5 节）。

## 4. 2026-10-20 的来源替换

体检方法：对每个选择/判断题来源算「最长选项」「众数字母」「Yes/No 众数」三种不看题基线，可疑的再做无图盲测（qwen3.8-flash）。

| # | 原来源 | 问题 | 现来源 |
|---|---|---|---|
| 14 | ERIQ human_intention 56 题 | 52 题只有正确项带 possibly/likely | RoboBench implicit_goal Q1，100 题，GPT-4o 判 |
| 27 | ERIQ action_understanding 148 题 | 四选一不看视频答对 68.8% | EgoTaskQA 状态题 200 + ERIQ 判断题 68 |
| 28 | ERIQ subtask_planning 148 题 | 单步判断非分解；67% 正确项是最长项 | RoboBench 规划题 320 题，GPT-4o 判 |
| 31 | MomaGraph effect_prediction | 不看图 92.9% | EAI VirtualHome 动作序列 305 题 |
| 36 | ASIMOV safety_tool_call 168 题 + PIO 12 | 输入是结构化事件 JSON，不看图，96–100 | ASIMOV 人员安全监控 400 题（能力体系 主测 Video，4 帧双目图判断 3m 内有无他人，停/继续各 200）+ PIO 12 |

| 30 | mvrobobench trajectory + MomaGraph spatial_reasoning | MomaGraph 不看图 57.5%（有图 66.7%） | 只留 mvrobobench trajectory 198 题 |
| 41 | PhysBench fluid 两切片 + causalphys | fluid_perception 不看图 70.0%（有图 90.0%） | 撤 fluid_perception；fluid_prediction（不看视频 26.4%）与 causalphys 保留 |

这七项 10-20 前后的分数**不可比**。#36 新来源下多个模型漏报严重（deepseek、mimo-flash 该停只停了一半）。

## 5. 2026-10-21 的对口性修正

审查口径：逐项核对「来源实际在问什么」是否等于能力定义（与第 4 节的捷径体检互补）。

| # | 原来源 | 问题 | 现来源 |
|---|---|---|---|
| 13 | EgoTaskQA event_order 切片 | 切片里**没有顺序题**（描述 66% / 反事实 21% / 解释 12%） | TVBench：action_sequence 200（二选一「先做了哪个」，按标注起止裁片段，A/B 各 100）+ egocentric_sequence 200（四选一动作排列），16 帧 |
| 7 | EPIC GroundDetection / AffordanceRegion / FeasiblePath | FeasiblePath 真值是路径掩码，官方要输出 ≥3 个路径点；本地只让模型指一个点，模型去指目标物体 | 撤 FeasiblePath，只留前两类 169 题 |
| 42 | MV-RoboBench cross_view + EPIC VisualMatching | EPIC 官方是 3 图 + 框的匹配协议，本地口径测不到跨视角对应 | 只留 MV-RoboBench cross_view 198 题 |
| 26 | PhyX Mechanics 133 题 | 多是汽车 / 飞机 / 抛体 / 读图计算，能力体系 只要「具身操作相关题」 | 人工挑 39 道接触力学题（推拉、摩擦、绳与滑轮、斜面、杠杆与扳手、手指夹持、搬箱、弹簧），白名单在 `c_tier_subscores.DERIVED` |
| 17 | NegVQA 原版 91 + 否定版 91 平均 | 原版没有否定词，比否定版高 8–26 分 | 配对口径：同一题原版和否定版都答对才算对（91 对） |
| 10 | EPIC 朝向指代 376 题 | 227 题问人、81 题问动物 | 只留物体朝向 68 题（椅子、沙发、车、鞋…），一题 ≈ 1.5 分 |
| 37 | CausalPhys 碰撞预测 148 题 | 只问「车会不会撞」，8/10 模型与随机无差别 | 补 VLM4D 真实视频 300 题：ego4d 150（盲测 46.7，多为左/右二选一，按 50 读）+ 第三人称 150（盲测 30.7，随机 25）；碰撞题保留 |
| 46 | ASIMOV 读表，可读题按读数落区间计分 | 可读题各模型 0–48，丢分主要是读数不准，测的是读表精度 | 判断口径（见第 1 节），两类仍等权 |

这八项 10-21 前后的分数**不可比**。

仍不完全对口、暂无替代：
- **#6 场景语义**：OpenEQA 150 题里直接问「什么房间 / 用途」的约 10 题，其余多是「我想做某事，场景里有什么能用」。
  OpenEQA 全量 1636 题里此类题也只有约 25 道，没有现成的场景 / 功能区识别评测集。
- **#37 轨迹理解与预测**：10-21 补了 VLM4D（当前运动方向，见上表），「预测」仍只有 CausalPhys「车会不会撞」（见 3.2）。
- **#45 场景完整判断与处置**：只测判断（点在另一视角里是否可见），不测处置（换视角 / 拒答）。

以上八项之外，#18 #29 同属不对口（见 3.1），无公开对口数据、不自建，维持现状。
