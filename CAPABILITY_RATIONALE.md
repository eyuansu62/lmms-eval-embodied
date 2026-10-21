# 能力选型依据

本文逐项说明 47 项能力**为什么挂这些数据集**：原规划指定了什么、实际用了什么、换掉过什么及原因、读分时要注意什么。能力列表与计分规则见 [CAPABILITIES.md](CAPABILITIES.md)，读表口径与已知局限的汇总见 [reports/README_capability.md](reports/README_capability.md)。

**说明**

- 「原规划来源」是能力体系设计时为每项能力指定的数据集与子集；「实际来源」是现在评分代码里真正使用的。
- 「盲测」指不给图片/视频、只给题干和选项，由 qwen3.8-flash（关闭思考）作答的准确率，用来判断题目能否不看画面就答对。另有三种不看题的基线：全选最长选项、全选最常见字母、全答 Yes/No 中多数一方。
- 「分数范围」是 10 个模型在该来源上的最低–最高分（逐题准确率，%），截至 2026-10-21。
- 选型的总原则：① 题目要真的在问这项能力（对口）；② 不看画面不能轻易答对（无捷径）；③ 原规划来源能用就用，不能用时找最接近的公开数据集替补，不自建题目；④ 同一批题不在两项能力里重复计分。

## 基础智能能力 › 世界理解 › 对象理解

### #1 本体能力可供性感知（P0）

- **定义**：当前执行器可以执行什么任务?
- **原规划来源**：ASIMOV-2.0：Constraints → Gripper Geometry and Type / Kinematics and Reach / Multi-arm and Coordination
- **实际来源**：asimov（400 题，B档·文本化，分数 67.0–88.8）
- **选型依据**：沿用原规划。ASIMOV 的 vla_feasibility：给定一个机器人平台的能力说明，判断某条指令它能不能执行，正是「当前执行器能做什么」。原规划把同一份 ASIMOV Constraints 同时列在 #1/#3/#31/#36 下，并要求复用题不重复加权，所以 ASIMOV 每个子集只挂一项能力：vla_feasibility 归 #1，constraints 归 #3。
- **读分注意**：官方是 tool-call 作答，这里改成文本作答，与论文分数不可比（口径标「B档·文本化」）。是非题，盲猜 50。

### #2 客体可供性感知（P0）

- **定义**：目标可以用来做什么?目标可以如何交互?交互部位标点/框
- **原规划来源**：RoboVQA：discriminative / generative affordance / Point-It-Out：s2 → afd / contact / RoboBench：static_affordance / dynamic_affordance
- **实际来源**：pio, robobench（329 题，A/B档·全量，分数 29.2–67.8）
- **选型依据**：沿用原规划中可用的两源：RoboBench 的 static_affordance / dynamic_affordance（选择题），Point-It-Out s2 的 affordance / contact（在图上标点）。RoboVQA 本地未接入。
- **读分注意**：PIO 是坐标题，曾因坐标格式（归一化 vs 像素）没说清导致个别模型塌到 0.8%，已修正提示词并重算。

### #3 本体客体交互关系（P0）

- **定义**：执行器可对目标进行什么操作?执行器能否对目标进行某种操作?
- **原规划来源**：MV-RoboBench：robotic_affordance_recognition / ASIMOV-2.0：Constraints → Gripper Geometry and Type / Kinematics and Reach / Multi-arm and Coordination
- **实际来源**：mvrobobench（206 题，A/B档·全量，分数 23.3–74.8）；asimov（641 题，B档·文本化，分数 82.4–91.3）
- **选型依据**：沿用原规划：MV-RoboBench 的 robotic_affordance_recognition（多视角下判断夹爪能对哪个物体做什么），加 ASIMOV constraints（给定夹爪类型与几何等本体约束，判断能不能抬起某物体）。两源口径不同（ASIMOV 为文本化），分两行出分。
- **读分注意**：ASIMOV constraints 10 个模型 82–91，接近天花板，盲猜 53.2；区分度主要来自 MV-RoboBench。ASIMOV 只覆盖 Gripper Geometry 一类，测不到原规划写的 Kinematics and Reach / Multi-arm 两类。

### #4 客体视觉属性（P0）

- **定义**：颜色、形状、3D几何属性。。。
- **原规划来源**：RoboBench：static_attribute → material / transparency / EPIC-Bench：BasicAttributes（颜色 / 几何题）
- **实际来源**：robobench（31 题，A/B档·全量，分数 67.7–87.1）；epic（49 题，C档·均衡，分数 22.4–83.7）
- **选型依据**：沿用原规划两源：RoboBench 的 visual_attribute（材质、透明度等），加 EPIC-Bench 的 BasicAttributes 颜色/材质/几何题（C 档，2026-10-19 用已跑数据补上第二来源）。EPIC 这三类与 #7/#10/#42 用到的题型不重叠。
- **读分注意**：RoboBench 只有 31 题，一题约 3 分。EPIC 部分是 C 档类别均衡采样，单独成行，不与 A/B 档平均。

### #5 客体物理属性（P1）

- **定义**：这个表面摩擦力大小如何,是不是刚体
- **原规划来源**：PhysBench：property → mass / attribute
- **实际来源**：physbench（330 题，A/B档·全量，分数 48.2–69.4）
- **选型依据**：沿用原规划：PhysBench 中 task_type=property 且 sub_type 为 mass / attribute 的题（质量、摩擦、刚柔等物理属性）。
- **读分注意**：曾有一个 physbench/object_permanence 切片挂在 #35，实测与本项 330 题 100% 重合（同一批题在两项各算一次），已于 2026-10-16 删除。

## 基础智能能力 › 世界理解 › 时空理解

### #6 场景语义（P1）

- **定义**：这是什么场景?用来做什么
- **原规划来源**：OpenEQA：world knowledge（房间 / 场景类型题）；functional reasoning（区域用途题）
- **实际来源**：openeqa（150 题，B档·帧序列，分数 48.7–76.0）
- **选型依据**：沿用原规划：OpenEQA 的 world knowledge 与 functional reasoning 两类，全量各 213/217 题，各抽 75 题；每题从探索视频中均匀取 16 帧。
- **读分注意**：**不完全对口**：150 题里直接问「这是什么房间 / 用来做什么」的只有约 10 题，其余多是「我想做某事，场景里有什么能用」或识别物体、品牌。在 OpenEQA 全量里此类题也只有约 25 道，检索未找到现成的场景 / 功能区识别评测集，暂维持。开放式作答，由 qwen3.8-max 判分（官方为 GPT-4）；数据发布于 2023 年，可能有训练污染。

### #7 空间布局（P1）

- **定义**：哪里是操作台(功能区域理解)哪里是可通行区域(通行与障碍理解)
- **原规划来源**：EPIC-Bench：GroundDetection / AffordanceRegion / FeasiblePath
- **实际来源**：epic（169 题，C档·均衡，分数 8.9–33.1）
- **选型依据**：沿用原规划的 EPIC-Bench 中两类：GroundDetection（地面 / 可通行区域）与 AffordanceRegion（可操作区域），模型点选，点落在区域掩码内即对。
- **读分注意**：2026-10-20 撤下 FeasiblePath_Ego/Exo：官方协议要求输出 ≥3 个路径点、按路径掩码与目标区域打分，本地实现只让模型「定位某物体」输出一个点——模型去指目标物体，真值却是通往它的路径，测的不是可通行区域。C 档类别均衡采样；分数普遍很低（9–33）是真实水平。

### #8 相对位置（P0）

- **定义**：A在B的那边?
- **原规划来源**：VSI-Bench：object_rel_direction_easy / object_rel_direction_medium / object_rel_direction_hard / RoboBench：spatial_relation
- **实际来源**：robobench, vsibench（276 题，A/B档·全量，分数 35.1–68.8）
- **选型依据**：沿用原规划两源：RoboBench spatial_relation，加 VSI-Bench 的 object_rel_direction（easy / medium / hard 合并为一个池子逐题计分）。
- **读分注意**：VSI-Bench 帧数必须从 model_args 传入（max_frames_num=32，官方默认）；从 10 帧改到 32 帧后 rel_direction 显著提高（p<0.0001）。官方三档等权的总分仍保留在 summary_A_B.csv。

### #9 相对距离（P0）

- **定义**：两个物体与执行器之间的远近
- **原规划来源**：MV-RoboBench：spatial_distance_judgement / VSI-Bench：object_rel_distance / object_abs_distance
- **实际来源**：mvrobobench/distance（198 题，A/B档·全量，分数 35.9–58.1）；vsibench/object_rel_distance（157 题，A/B档·全量，分数 36.9–71.3）；vsibench/object_abs_distance（172 题，A/B档·全量，分数 0.3–50.4）
- **选型依据**：沿用原规划三源：MV-RoboBench spatial_distance_judgement、VSI-Bench object_rel_distance（相对远近）、object_abs_distance（绝对距离数值）。原规划明确要求相对 / 绝对距离分开计分。
- **读分注意**：**不合并**，三个来源各占一行：相对与绝对距离实测差 15 个点，平均会掩盖差异。gemma-4-26b 在绝对距离数值题上大多回答「看不清，无法回答」，分数接近 0，是模型行为而非判分问题。

### #10 朝向和姿态（P1）

- **定义**：物体朝向那边
- **原规划来源**：BOP-Industrial / IPD：BOP 2025 Track 10；单视角 / 多视角 6D 检测
- **实际来源**：epic（68 题，C档·题型全量，分数 23.5–82.3）
- **选型依据**：原规划指定的 BOP-Industrial / IPD 6D 位姿检测为检测协议，无法用于 VLM 问答，未接入。2026-10-17 用 EPIC-Bench 的朝向指代题替补（「指出朝向相机左侧的物体」），并补跑凑成该题型全量 376 题。2026-10-21 进一步只保留物体朝向的 68 题。
- **读分注意**：替补来源测的是「按朝向找物体」，不是直接问朝向。376 题里 227 题问人、81 题问动物，与「物体朝向」不符，已剔除，只剩 68 题（椅子、沙发、车、鞋等），一题约 1.5 分。C 档题型全量，单独成行。

### #11 拓扑关系（P1）

- **定义**：物体之间的承载、包含、容纳
- **原规划来源**：3DSSG：支撑关系标注 / CausalPhys：Perception/containability / BDDL：inside 场景
- **实际来源**：causalphys（99 题，A/B档·全量，分数 35.4–77.8）
- **选型依据**：沿用原规划中可用的一源：CausalPhys Perception/containability（容纳、承载关系）。3DSSG 与 BDDL 是标注 / 构题源，不是现成的问答子集，未接入。

### #12 时间指代定位（P0）

- **定义**：“刚才”,“某一步”之后对应什么具体的事件
- **原规划来源**：Ego4D NLQ：Natural Language Queries / RoboBench：timestamp_analysis
- **实际来源**：robobench（91 题，A/B档·全量，分数 28.6–60.4）
- **选型依据**：沿用原规划：RoboBench timestamp_analysis（在操作视频中定位「某一步」对应的时刻）。Ego4D NLQ 为时间片段检索协议，未接入。
- **读分注意**：91 题，单一来源。

### #13 事件顺序理解（P0）

- **定义**：多个事件发生的先后顺序(多帧)
- **原规划来源**：EgoTaskQA：时序 / 动作依赖相关题 / RoboBench：timestamp_analysis（事件先后 / 序列排序所用时间标注）
- **实际来源**：tvbench_cap13（400 题，A/B档·全量，分数 59.5–92.2）
- **选型依据**：2026-10-21 改挂 TVBench：action_sequence 200 题（二选一「这个人先做了哪件事」，按标注起止时间裁好片段，正确答案在 A/B 各 100）+ egocentric_sequence 200 题（四选一，选项是同一组动作的不同排列），16 帧。原挂的 EgoTaskQA event_order 切片里**没有顺序题**（描述 66% / 反事实 21% / 解释 12%）；TVBench 的设计目标就是单帧或纯文本答不出。
- **读分注意**：盲测 action_sequence 54.5（随机 50）、egocentric 59.0（随机 25）：egocentric 只有 25 套实验流程，常识能排除一部分干扰项。有视频时高 15–40 分，说明测的是看视频排序。egocentric 部分强模型已到 96–98，接近满分。

## 基础智能能力 › 世界理解 › 指令理解

### #14 指令具化（P0）

- **定义**：识别开放性指令,推断出具体执行任务(收拾桌子是擦掉污渍还是)
- **原规划来源**：RoboBench：implicit_demand_goal / ERIQ：Human Intention Comprehension
- **实际来源**：robobench_plan（100 题，A/B档·全量，分数 34.4–59.1）
- **选型依据**：只挂 RoboBench implicit_demand_goal 的 Q1：给出隐含需求（如「我渴了」），要求推出具体的多步执行计划，100 题，GPT-4o 判分（与官方一致）。
- **读分注意**：2026-10-20 撤下两份：ERIQ human_intention（看手势猜意图，没有指令可具化；56 题里 52 题只有正确项带 possibly/likely 等词，不看图也能选对，10 个模型挤在 84–100）；RoboBench implicit_goal 的 Q2/Q3（看执行视频预测下一步、判断某步是否完成，测的是执行跟踪而非具化）。

### #15 指令歧义识别（P1）

- **定义**：同一指令有多种合理性解读
- **原规划来源**：Ambi3D：指令在场景中是否具有唯一解释
- **实际来源**：asimov（396 题，B档·文本化，分数 75.2–85.1）
- **选型依据**：原规划指定的 Ambi3D 未接入。用 ASIMOV uncertainty 替补：给出指令与场景，判断指令是否有歧义，有歧义 / 无歧义各 198 题。
- **读分注意**：替补来源；原规划特别说明「识别歧义」不等于「提出最有效的澄清问题」，本项只测前者。文本化口径。

### #16 目标理解（P0）

- **定义**：理解任务的最终目标状态是什么
- **原规划来源**：Embodied Agent Interface（EAI）：Goal Interpretation / RoboBench：explicit_object_goal / implicit_demand_goal（终态判定所用场景）
- **实际来源**：eai（438 题，B档·纯文本，分数 48.4–60.2）
- **选型依据**：沿用原规划主测：Embodied Agent Interface 的 Goal Interpretation，BEHAVIOR 100 题 + VirtualHome 338 题，纯文本，官方原协议。RoboBench 在原规划中只作场景源，没有原生的终态判定题。
- **读分注意**：报表是逐题 F1 的平均，官方榜单是 micro F1（结果文件里另有 eai_f1）。数据从 PyPI 的 eai-eval 包取（HF 上那份不能用来判分）；生成 VirtualHome 提示词时必须固定 PYTHONHASHSEED。

### #17 否定理解（P0）

- **定义**：正确区分"不要拿红色的"和"拿不是红色的"的行为差异
- **原规划来源**：NegBench：MCQs with Negated Captions / BEHAVIOR / BDDL：行为否定相关场景
- **实际来源**：negvqa（91 题，A/B档·全量，分数 62.6–81.3）
- **选型依据**：原规划指定的 NegBench 不适配生成式模型：它用 CLIP 嵌入相似度判分，且正确答案恒定放在第一个选项，改成生成式作答时全选 A 就能拿分。改用 NegVQA（原生二选一 VQA，官方已在 VLMEvalKit 评过 20 个模型），并经四层过滤从 dev 1000 例筛出 91 例自然场景题（剔除否定改写后答案翻转不成立的题）。
- **读分注意**：2026-10-21 起改为**配对口径**：同一题原版和否定版都答对才算对。原版没有否定词，单看测的是普通 VQA，之前两版直接平均，原版比否定版高 8–26 分，一半分数不测否定。

### #18 条件指令理解（P0）

- **定义**：理解带条件的指令,比如"如果杯子是空的就去装水,否则放回原处"
- **原规划来源**：BEHAVIOR / BDDL：if–then–else 条件指令相关场景 / MomaGraph-Bench：Precondition & Effect Reasoning
- **实际来源**：momagraph（168 题，A/B档·全量，分数 87.5–95.8）
- **选型依据**：原规划的主测是 BEHAVIOR / BDDL 的 if-then-else 条件指令派生题，属于构题源而非现成子集；按「不自建」的原则检索了相关论文，未找到对口的公开数据集，暂挂 MomaGraph-Bench effect_prediction（原规划的另一来源）。
- **读分注意**：**不对口且无区分度**：测的是「插上插头灯会亮」式的常识因果，不是条件指令；不看图 92.9%（有图 91.1%），10 个模型 88–96。同一份题曾挂 #31，已于 2026-10-20 撤下。

### #19 属性指代（P0）

- **定义**：属性指代序数指代(第三本书等。。。)
- **原规划来源**：RefSpatial-Expand-Bench：location（属性 / 组合约束题） / Point-It-Out：s1 → restriction / object part（属性相关题）
- **实际来源**：pio, refspatial（98 题，A/B档·全量，分数 6.1–70.4）
- **选型依据**：沿用原规划两源：RefSpatial-Expand-Bench 的 location 中属性 / 组合约束题，加 Point-It-Out s1 的 restriction / object part 题。
- **读分注意**：RefSpatial 只有 3 道人工确认过的题（其余需人工审标），逐题口径下只占本项的 3%，基本由 PIO 决定。PIO 是标点题，分数跨度很大（6–70）。

### #20 序数指代（P0）

- **定义**：按序数/次序表达定位目标(第三本、第2个、最左边第二个)
- **原规划来源**：RefSpatial-Expand-Bench：location（序数指代题） / Point-It-Out：s1 序数对照题所用场景 / RoboBench：number 规划题中的序数对照场景
- **实际来源**：pio, refspatial（22 题，A/B档·全量，分数 9.1–68.2）
- **选型依据**：原规划三源：RefSpatial 的 4 道序数种子题；PIO 从 s1 的 95 题里逐条审出 18 道序数表达（「左边第二个」等），分数从已有样本按 doc_id 取，不发新请求；RoboBench 那一源原规划自己注明「尚未锁定序数指代题」，没有可用切片。
- **读分注意**：只有 22 题，一题约 4.5 分，波动大。

## 基础智能能力 › 世界理解 › 因果理解

### #21 因果链推理（P0）

- **定义**：理解多个动作或事件之间的连续因果关系,识别间接效果和连锁反应
- **原规划来源**：CausalPhys：含明确多跳因果依赖的题（按因果图与题意筛选）
- **实际来源**：causalphys（83 题，A/B档·全量，分数 49.4–81.9）
- **选型依据**：沿用原规划：从 CausalPhys 的 Perception / Anticipation / Intervention 目录中，按因果图链长 ≥3 条边筛出跨越 ≥2 个因果环节的题，正是「多跳因果」的判据。
- **读分注意**：与 #23 取自同一批目录，两者有重叠。

### #22 反事实推理（P1）

- **定义**：推理"如果不执行某个动作会怎样",用于评估动作的必要性和替代路径,"如果不先打开盖子,还能倒出水吗?"
- **原规划来源**：CausalVQA：Counterfactual / Hypothetical（动作移除 / 替换题） / EgoTaskQA：Counterfactual
- **实际来源**：egotaskqa（81 题，A/B档·全量，分数 27.2–69.1）
- **选型依据**：原规划两源：CausalVQA 的反事实题（未接入）与 EgoTaskQA 的 counterfactual 类型。使用 EgoTaskQA 带题型标签的 tagged_qas.json 精确筛出 counterfactual 题，并只保留是非题。
- **读分注意**：EgoTaskQA 的开放题答案是封闭动作词表，生成式作答几乎全判错，所以只取是非题。EgoTaskQA 是非题对所有模型都接近随机（视频只带来约 11 分），经检查是题难（抽象属性、多机位剪辑），不是标注错误。

### #23 干预推理（P1）

- **定义**：判断针对某个中间状态的干预能否改变最终结果,比如"在这一步阻止它还来得及吗"
- **原规划来源**：CausalPhys：Intervention/spatial_manipulation / Intervention/temporal_shifting / Intervention/action_substitution
- **实际来源**：causalphys（197 题，A/B档·全量，分数 53.8–81.2）
- **选型依据**：沿用原规划：CausalPhys Intervention 下的 action_substitution、spatial_manipulation、temporal_shifting 三类（「如果换个动作 / 改个位置 / 提前一步，结果还会这样吗」）。
- **读分注意**：与 #21 取自同一批目录，有重叠。

### #24 归因理解（P0）

- **定义**：给定观察到的结果状态,向上推断是什么原因或操作导致了当前状态
- **原规划来源**：CausalPhys：Perception/Mechanics_Reasoning（结果→原因题） / EgoTaskQA：Explanatory
- **实际来源**：causalphys, egotaskqa（117 题，A/B档·全量，分数 74.4–82.0）
- **选型依据**：沿用原规划两源：CausalPhys Mechanics_Reasoning 中「由结果推原因」的题（按题干模式筛出），加 EgoTaskQA 的 explanatory 类型是非题。
- **读分注意**：10 个模型 74–82，区分度低。EgoTaskQA 部分同 #22，接近随机。

### #25 失败定位与原因分析（P0）

- **定义**：确定失败的原因,以及发生在哪个具体步骤或操作阶段
- **原规划来源**：RoboBench：high_level_planning_error / low_level_execution_error / ERIQ：Mistake Classification
- **实际来源**：eriq, robobench（256 题，A/B档·全量，分数 43.8–62.5）
- **选型依据**：沿用原规划两源：RoboBench high_level_error / low_level_error（规划错误、执行错误的定位），加 ERIQ Mistake Classification。
- **读分注意**：ERIQ 进能力表一律用「宽松抽取 + Yes/No 归一」口径：官方严格口径只认回复开头的字母，先分析再给答案、或照提示词回 Yes/No 的模型会被整片判错（如 deepseek 曾在 #33 严格口径只有 29.4）。ERIQ 主表仍保留官方严格口径，三种口径对照见 summary_eriq_calibers.csv。

## 基础智能能力 › 世界理解 › 科学理解

### #26 物理知识（P1）

- **定义**：模型是否掌握支撑具身推理的基础物理规律(重力、摩擦、守恒等)
- **原规划来源**：PhysBench：物理规律应用题 / PhyX：Mechanics（具身操作相关题）
- **实际来源**：phyx（39 题，C档·均衡，分数 84.6–100.0）
- **选型依据**：原规划两源：PhysBench 的物理规律应用题（无对应切片，现有 PhysBench 切片是 #5 的属性题，口径不同不并入），与 PhyX Mechanics 中「具身操作相关」的题。2026-10-21 从 PhyX Mechanics 133 题中人工挑出 39 道接触力学题：推拉、摩擦、绳与滑轮、斜面、杠杆与扳手、手指夹持、搬箱、弹簧。
- **读分注意**：之前挂的是 Mechanics 全部 133 题，多是汽车、飞机、抛体、读图计算，不符合「具身操作相关」。筛完 10 个模型 85–100，仍是天花板，只有 39 题，一题约 2.6 分。C 档均衡采样，单独成行。

## 基础智能能力 › 世界推演 › 状态变化推演

### #27 单体/多体状态理解（P0）

- **定义**：当前正在发生什么?整个过程发生了什么
- **原规划来源**：EgoTaskQA：Descriptive（对象状态 / 状态变化题） / ERIQ：Action Understanding
- **实际来源**：egotaskqa, eriq（268 题，A/B档·全量，分数 53.7–72.4）
- **选型依据**：原规划主测是 EgoTaskQA Descriptive 中的对象状态 / 状态变化题：建了 state_change 切片 200 题（状态 / 变化各 100，只取是非题，且排除已在其他切片中的题）。补充 ERIQ Action Understanding **只取判断题** 68 题。
- **读分注意**：2026-10-20 前只挂 ERIQ action_understanding 148 题，其中四选一那半不看视频答对 68.8%（随机 25%），正确字母严重偏斜，已撤；ERIQ 的 Yes/No 判断题盲测 47.1%，是干净的。EgoTaskQA 部分盲测 52.5%，有视频 63.5%。

### #28 子任务分解（P0）

- **定义**：泡一杯咖啡需要哪些步骤
- **原规划来源**：RoboBench：generalized_planning 中 unique_id 以 _Q1 结尾的题 / ERIQ：Subtask Planning / RoboVQA：planning
- **实际来源**：robobench_plan（320 题，A/B档·全量，分数 60.4–76.9）
- **选型依据**：挂 RoboBench 规划题 Q1，16 个子任务各 20 题共 320 题，GPT-4o 判分（与官方一致）。
- **读分注意**：2026-10-20 撤下 ERIQ subtask_planning：问的是「这一步该做哪个动作」，是单步判断而非分解，且 75 道选择题里 50 道的正确项就是最长选项。50 题版 RoboBench 只有一个模型跑过，20 题版是其子集，精度约 ±3 分够用。

### #29 指令顺序理解（P0）

- **定义**：识别多步指令的先后约束关系
- **原规划来源**：MomaGraph-Bench：Action Sequence Reasoning / Embodied Agent Interface（EAI）：Action Sequencing
- **实际来源**：momagraph（343 题，A/B档·全量，分数 56.0–71.1）
- **选型依据**：原规划两源：MomaGraph-Bench Action Sequence Reasoning 与 EAI Action Sequencing。EAI 那一项实际测的是「按步骤生成可执行动作序列」（已挂 #31）；按「不自建」的原则检索了 LexiCon、Plan-Bench 等，均不对口，暂挂 MomaGraph action_sequence。
- **读分注意**：**不对口**：测的是常识顺序，不是「遵守用户指定的先后约束」；不看图 56.9%（有图 62.4%，随机 25%），81% 的正确答案是 B 或 C。

### #30 物理可行性（P0）

- **定义**：判断计划中的动作在物理上是否可以执行,比如目标是否可达、操作是否违反运动约束
- **原规划来源**：MV-RoboBench：robotic_trajectory_selection / MomaGraph-Bench：Spatial Reasoning（reachability 题）
- **实际来源**：mvrobobench（198 题，A/B档·全量，分数 36.9–83.3）
- **选型依据**：沿用原规划中的 MV-RoboBench robotic_trajectory_selection（多视角下选可行的运动轨迹），198 题。
- **读分注意**：2026-10-20 撤下原规划的另一源 MomaGraph spatial_reasoning（可达性题）：不看图 57.5%（有图 66.7%，随机 25%），52% 的正确项是最长选项，且按题数计它会占本项的 61%。

### #31 约束满足检查（P0）

- **定义**：在规划任务步骤时确保所有约束都被满足,没有任何步骤违反任务要求
- **原规划来源**：MomaGraph-Bench：Precondition & Effect Reasoning / ASIMOV-2.0：Constraints / Embodied Agent Interface（EAI）：全计划约束检查相关任务 / BDDL：全计划约束检查相关场景
- **实际来源**：eai（305 题，B档·纯文本，分数 76.1–87.5）
- **选型依据**：2026-10-20 改挂 EAI VirtualHome Action Sequencing 305 题：模型生成完整动作序列，由官方模拟器逐步检查前置条件与终态，报执行成功率，正是「每一步都不违反约束」。剔除了标准答案本身都过不了终态判定的题（与官方 evaluate_results 同一判据）。
- **读分注意**：撤下 MomaGraph effect_prediction：不看图 92.9%，是常识因果。10 个模型 76–88，失败几乎都是「缺步骤」。纯文本口径。提示词须用官方现行模板（包里自带的 helm_prompts.json 是旧模板，会让所有回答都解析失败）。

### #32 进度估计（P0）

- **定义**：估计当前任务完成了多少比例,还剩哪些步骤未执行,进度百分比(VLM对具身模型进行评估的关键能力)
- **原规划来源**：ERIQ：Task Progress
- **实际来源**：eriq（69 题，A/B档·全量，分数 31.9–72.5）
- **选型依据**：原规划唯一来源：ERIQ Task Progress，宽松抽取口径。
- **读分注意**：只有 69 题且是唯一来源。严格口径与宽松口径差别极大（qwen3.8-flash 严格 7.25 / 宽松 68.12），原因见 #25。

### #33 失败检测（P0）

- **定义**：实时识别当前操作是否已偏离预期轨迹或任务是否已失败
- **原规划来源**：ERIQ：Mistake Existence / Success Detection / RoboVQA：success
- **实际来源**：eriq（279 题，A/B档·全量，分数 59.9–80.3）
- **选型依据**：沿用原规划的 ERIQ Mistake Existence 与 Success Detection，宽松抽取 + Yes/No 归一口径。RoboVQA success 未接入。
- **读分注意**：success_detection（197 题）全选 B 即得 59.4%，且按题数计它占本项的 71%，读分必须对照这条基线。

### #34 恢复策略（P0）

- **定义**：根据失败原因制定恢复方案,选择重试、替代路径或任务降级
- **原规划来源**：ERIQ：Mistake Recovery / SWITCH：Recovery Reasoning（控制界面场景）
- **实际来源**：eriq（50 题，A/B档·全量，分数 64.0–96.0）
- **选型依据**：沿用原规划的 ERIQ Mistake Recovery，宽松抽取口径。SWITCH 的 Recovery Reasoning 是控制界面场景，未接入。
- **读分注意**：只有 50 题。

### #35 物体恒常性（P1）

- **定义**：物体被遮挡或离开视野后仍然存在判断场景中物体的数量和质量是否符合守恒预期,识别凭空消失或出现的异常
- **原规划来源**：IntPhys 2：Permanence / PhysBench：Number / Mass 相关场景
- **实际来源**：恒常性: corecognition, intphys2（338 题，A/B档·全量+B档·二分类，分数 45.6–74.5）；守恒: corecognition（71 题，A/B档·全量，分数 40.9–97.2）
- **选型依据**：**恒常性**：IntPhys 2 Main 集 permanence 240 个视频（物体被遮挡后凭空消失 / 出现，判断是否可能），官方提示词 + Yes/No 判分；2026-10-20 加 CoreCognition permanence 98 题作第二来源。**守恒**：原规划的 PhysBench 数量 / 质量题不是守恒测试（原规划自己也写明「没有已确认的 conservation 子集」），检索论文后接入 CoreCognition conservation 71 题（ICML 2025）。原规划要求两部分分别计分，拆成两行。
- **读分注意**：IntPhys 2 可能 / 不可能各半，全答 Yes 即 50%。CoreCognition 的是非题已做 YES/NO 等量采样；守恒部分 qwen 系列 90–97，可能接近天花板。IntPhys 2 为 CC BY-NC 4.0，仅限非商用。2026-10-16 前挂的 physbench/object_permanence 与 #5 完全重合，已删除。

### #36 安全监控（P0）

- **定义**：在执行过程中持续监控动作路径上是否存在碰撞风险或其他危险后果:是否可能造成人员伤害、物体损坏或违反安全规范
- **原规划来源**：ASIMOV-2.0：Video / Constraints / Point-It-Out：s2 → safety
- **实际来源**：asimov, pio（412 题，A/B档·全量，分数 64.3–84.5）
- **选型依据**：原规划主测是 ASIMOV-2.0 Video：接入 human_safety_monitoring 400 题（给 4 帧双目图像，判断 3 米内是否有人、该停还是继续，停 / 继续各 200）；补充 Point-It-Out s2 safety 12 题（在图上标出危险区域）。
- **读分注意**：2026-10-20 撤下 ASIMOV safety_tool_call（168 题）：输入是结构化的事件 JSON，不用看图，10 个模型 96–100。新来源下多个模型漏报严重（deepseek、mimo-flash 该停的只停了一半）。

## 基础智能能力 › 世界推演 › 运动学推演

### #37 轨迹理解与预测（P0）

- **定义**：理解物体当前的运动方向并预测其未来路径
- **原规划来源**：CLEVRER：motion_trajectory 标注 / CausalPhys：Anticipation/Collision_Prediction / RoboVQA：future prediction
- **实际来源**：causalphys, vlm4d_cap37（448 题，A/B档·全量，分数 54.0–78.3）
- **选型依据**：原规划三源中 CLEVRER 与 RoboVQA future prediction 未接入，原只有 CausalPhys Collision_Prediction。2026-10-21 补 VLM4D 真实视频 300 题（第一人称 Ego4D 150 + 第三人称 150），测「物体当前往哪动、往哪转」，覆盖定义的前半句；碰撞题保留，覆盖「预测」那半。
- **读分注意**：碰撞题 148 题全是「这辆车会撞吗」，是 / 否各半，10 个模型里 8 个在 47–64，58 分以下与随机无差别；本项的区分度主要来自 VLM4D。VLM4D 盲测：第三人称 30.7（随机 25）；第一人称 46.7，因为多数题实为「左还是右」二选一，按 50 读。

### #38 运动连续性（P1）

- **定义**：判断物体的运动轨迹是否连贯,识别出违反物理规律的跳变或突变
- **原规划来源**：IntPhys 2：Spatio-Temporal Continuity
- **实际来源**：intphys2（240 题，B档·二分类，分数 47.9–70.8）
- **选型依据**：沿用原规划：IntPhys 2 continuity 240 个视频（物体在遮挡物后瞬移、轨迹不连续），官方提示词 + Yes/No 判分。
- **读分注意**：全答 Yes 即 50%；仅限非商用。

## 基础智能能力 › 世界推演 › 动力学推演

### #39 单体力学效果（P0）

- **定义**：预测施加力或发生碰撞后物体的运动和形变结果,比如松手后会怎么样
- **原规划来源**：PhysBench：Physics-based Dynamics → Throwing / Collision / Manipulation（单体结果预测题）
- **实际来源**：physbench（187 题，A/B档·全量，分数 29.4–65.2）
- **选型依据**：沿用原规划：PhysBench dynamics 中 manipulation / throwing 类的单物体结果预测题。
- **读分注意**：「单物体」是按题干是否出现多个颜色词等启发式判断的，不是官方标签。

### #40 多体交互预测（P1）

- **定义**：多个物体发生交互时的联动结果,比如推倒一个物体引发的连锁反应,机械传动
- **原规划来源**：PhysBench：task_type=dynamics / sub_type=collision / ability_type=prediction（多体联动题）
- **实际来源**：physbench（68 题，A/B档·全量，分数 27.9–66.2）
- **选型依据**：沿用原规划：PhysBench dynamics / collision / prediction 中涉及多个物体联动的题。
- **读分注意**：「多物体」判据同 #39，为启发式；68 题。

### #41 刚体、流体、柔体（P1）

- **定义**：软体形变和弹性回弹,水位测试
- **原规划来源**：PhysBench：Physics-based Dynamics → Manipulation / Fluid / CausalPhys：Anticipation/deformation / Anticipation/Fluid_Flow
- **实际来源**：causalphys, physbench（306 题，A/B档·全量，分数 49.4–72.2）
- **选型依据**：原规划两源：PhysBench dynamics 中的 fluid 预测题，加 CausalPhys Anticipation 下的 deformation（柔体形变）与 Fluid_Flow（流体）。
- **读分注意**：2026-10-20 撤下 PhysBench fluid_perception：不看图 70.0%（有图 90.0%）。fluid_prediction 不看视频 26.4%（约等于随机），保留。

## 基础智能能力 › 世界推演 › 空间几何推演

### #42 多视角匹配（P1(P0)）

- **定义**：A视图中的物体在B视图中的位置
- **原规划来源**：MV-RoboBench：spatial_cross_view_matching / EPIC-Bench：VisualMatching
- **实际来源**：mvrobobench（198 题，A/B档·全量，分数 21.2–62.1）
- **选型依据**：沿用原规划中的 MV-RoboBench spatial_cross_view_matching：在一个视角里框出的物体，在另一视角中对应哪个框，198 题。
- **读分注意**：2026-10-20 撤下原规划的另一源 EPIC VisualMatching：官方协议给三张图（视角 A、参考区域叠加图、视角 B）并要求在 B 中框出对应区域，本地实现只给场景图和文字描述，凭文字就能定位，测不到跨视角匹配。

### #43 第三方视角理解（P1）

- **定义**：从说话人视角出发的前后左右
- **原规划来源**：OmniSpatial：Perspective Taking → Allocentric → Direction / Order（他人视角题）
- **实际来源**：omnispatial（200 题，C档·均衡，分数 44.5–80.5）
- **选型依据**：沿用原规划：OmniSpatial 的 Perspective Taking 类（从他人视角判断前后左右、先后顺序）。
- **读分注意**：C 档类别均衡采样，单独成行。

### #44 2D/3D理解与转换（P1）

- **定义**：从2D图像/视频恢复3D结构:深度、占用、度量尺寸
- **原规划来源**：DA3-BENCH（ScanNet++）：recon_unposed / recon_posed / VSI-Bench：object_size_estimation / room_size_estimation
- **实际来源**：vsibench（265 题，A/B档·全量，分数 0.0–67.7）
- **选型依据**：沿用原规划中的 VSI-Bench object_size_estimation / room_size_estimation（由视频估计物体尺寸、房间面积）。DA3-BENCH（ScanNet++ 三维重建）为重建协议，未接入。
- **读分注意**：数值题按 VSI-Bench 官方的数值题打分方式。gemma-4-26b 在尺寸估计 265 题里 258 题回答「看不清，无法回答」，得 0 分，是模型行为而非判分问题。

## 适应学习能力 › 学习进化 › 自主探索

### #45 场景完整判断与处置（P1）

- **定义**：关键操作区域是否在视野内
- **原规划来源**：SpatialUncertain：answerability / ViewSelect / Abstain-then-ViewSelect
- **实际来源**：crosspoint（200 题，C档·均衡，分数 46.5–83.5）
- **选型依据**：原规划的 SpatialUncertain 未接入。2026-10-17 用 CrossPoint 的跨视角可见性题替补（「图 1 中红点的位置在图 2 里被遮挡了吗」），200 题。
- **读分注意**：只测「关键区域是否可见」的判断，不测「看不到之后如何处置」（换视角、拒答）。全选同一答案约 53.5%。C 档均衡采样。

### #46 低视觉质量判断与处置（P1）

- **定义**：是否有遮挡光照是否不足
- **原规划来源**：RoboStressBench：Lighting / Geometry-Occlusion / Viewpoint
- **实际来源**：asimov（162 题，A/B档·全量，分数 66.8–83.1）
- **选型依据**：原规划的 RoboStressBench 为非商用许可，未接入。用 ASIMOV 读表题替补：图里的仪表有的能读（31 题）、有的因强反光、打码、结霜、油污而读不出（131 题），模型要么报读数、要么回答 unreadable。
- **读分注意**：两类**等权**（逐题口径的唯一例外），否则全答「看不清」就有 81%。2026-10-21 起改为判断口径：能读的图只要报了数值就算对，不看读数准不准——之前按读数是否落在区间计分，各模型 0–48，丢的分主要是读表精度，不是画质判断。

### #47 知识检索与求助（P1）

- **定义**：是否能询问必要信息和检索必要的知识
- **原规划来源**：FreeAskWorld：Direction Inquiry Task / OK-VQA：相关知识题
- **实际来源**：aqua（150 题，B档·judge，分数 28.7–58.7）；mmupd（100 题，B档·非循环，分数 46.0–80.0）
- **选型依据**：按定义的三层各取一个来源：「自我检视」用 MM-UPD（题目本身无解或缺选项时能否识别，50 道原题，正常版和无解版都答对才算对）；「询问必要信息」用 AQuA（判断该直接答还是该先问，按等级均衡抽 150 题，LLM 判分）。「检索」层的候选 LiveVQA 时效已过，决定不接。原规划的 FreeAskWorld、OK-VQA 未接入。
- **读分注意**：检索层没有覆盖。MM-UPD 只问主版本，不做官方的循环评测。AQuA 由 qwen3.8-max 判分（论文用 GPT-5-mini），qwen3.8-max 本身也是被测模型，可能有自评偏差。
