# 能力体系 / Capability Taxonomy

47 项能力，按五级层级组织（一级 → 五级），每项挂一个或多个 benchmark 子集。本文件由 `map_subtask_to_capability.py` 的映射生成，与评分代码一致。

47 capabilities in a five-level hierarchy (level 1 → level 5). Each capability is scored on one or more benchmark subsets. This file is generated from the mapping in `map_subtask_to_capability.py` and matches the scoring code.

## 计分规则 / Scoring

- 能力分 = 该能力下所有题的逐题准确率（答对题数 / 总题数）；子任务 → 来源、来源 → 能力两层都按题数加权。
  A capability score is per-question accuracy over every question mapped to it; both aggregation levels are weighted by question count.
- 唯一例外 #46：可读 / 不可读两类等权。 The one exception is #46, where the readable and unreadable classes are weighted equally.
- 不同口径（A/B 档全量、C 档类别均衡采样、文本化、LLM judge 等）不混合平均，分行给出；#9 与 #35 按 xlsx 要求拆行。
  Sources with different scoring calibers (full A/B tier, class-balanced C tier, textualized, LLM-judged, …) are not averaged together and appear as separate rows; #9 and #35 are split by design.
- 来源标 `[C]` 的为 C 档（类别均衡采样）。 Sources marked `[C]` come from the class-balanced C tier.
- 每项为什么选这些来源见 [CAPABILITY_RATIONALE.md](CAPABILITY_RATIONALE.md)。 The rationale for every source choice is in CAPABILITY_RATIONALE.md (Chinese).
- 读分前请阅读 `reports/README_capability.md`（已知局限、盲测基线）。 See `reports/README_capability.md` for known limitations and blind baselines before interpreting scores.

## 能力列表 / Capabilities

### 基础智能能力 › 世界理解 / Foundational intelligence › World understanding

| # | 三级 › 四级 / Level 3 › 4 | 能力 / Capability | 优先级 / Priority | 典型问题 / Example | 来源（子任务） / Sources (subtasks) | 题数 / #Q |
|---|---|---|---|---|---|---|
| 1 | 对象理解 › 可供性理解 | **本体能力可供性感知**<br>Embodiment affordance | P0 | 当前执行器可以执行什么任务? | asimov (vla_feasibility) | 400 |
| 2 | 对象理解 › 可供性理解 | **客体可供性感知**<br>Object affordance | P0 | 目标可以用来做什么?目标可以如何交互?交互部位标点/框 | robobench (static_affordance, dynamic_affordance); pio (affordance) | 329 |
| 3 | 对象理解 › 本体客体交互关系 | **本体客体交互关系**<br>Embodiment–object interaction | P0 | 执行器可对目标进行什么操作?执行器能否对目标进行某种操作? | mvrobobench (affordance); asimov (constraints) | 206<br>641 |
| 4 | 对象理解 › 视觉属性 | **客体视觉属性**<br>Object visual attributes | P0 | 颜色、形状、3D几何属性。。。 | robobench (visual_attribute); epic (Color, Material, Geometry) [C] | 31<br>49 |
| 5 | 对象理解 › 物理属性 | **客体物理属性**<br>Object physical attributes | P1 | 这个表面摩擦力大小如何,是不是刚体 | physbench (property_object) | 330 |
| 6 | 时空理解 › 空间理解 | **场景语义**<br>Scene semantics | P1 | 这是什么场景?用来做什么 | openeqa (world_knowledge, functional_reasoning) | 150 |
| 7 | 时空理解 › 空间理解 | **空间布局**<br>Spatial layout | P1 | 哪里是操作台(功能区域理解)哪里是可通行区域(通行与障碍理解) | epic (GroundDetection, AffordanceRegion) [C] | 169 |
| 8 | 时空理解 › 空间理解 | **相对位置**<br>Relative position | P0 | A在B的那边? | robobench (spatial_relation); vsibench (object_rel_direction) | 276 |
| 9 | 时空理解 › 空间理解 | **相对距离**<br>Relative distance | P0 | 两个物体与执行器之间的远近 | mvrobobench (distance); vsibench (object_rel_distance); vsibench (object_abs_distance) | 198<br>157<br>172 |
| 10 | 时空理解 › 空间理解 | **朝向和姿态**<br>Orientation & pose | P1 | 物体朝向那边 | epic (Orientation_object) [C] | 68 |
| 11 | 时空理解 › 结构理解 | **拓扑关系**<br>Topological relations | P1 | 物体之间的承载、包含、容纳 | causalphys (containability) | 99 |
| 12 | 时空理解 › 时序理解 | **时间指代定位**<br>Temporal reference grounding | P0 | “刚才”,“某一步”之后对应什么具体的事件 | robobench (timestamp_analysis) | 91 |
| 13 | 时空理解 › 时序理解 | **事件顺序理解**<br>Event order | P0 | 多个事件发生的先后顺序(多帧) | tvbench_cap13 (action_sequence, egocentric_sequence) | 400 |
| 14 | 指令理解 › 意图理解 | **指令具化**<br>Instruction grounding | P0 | 识别开放性指令,推断出具体执行任务(收拾桌子是擦掉污渍还是) | robobench_plan (robobench_implicit_goal_q1) | 100 |
| 15 | 指令理解 › 意图理解 | **指令歧义识别**<br>Instruction ambiguity detection | P1 | 同一指令有多种合理性解读 | asimov (uncertainty) | 396 |
| 16 | 指令理解 › 内容理解 | **目标理解**<br>Goal understanding | P0 | 理解任务的最终目标状态是什么 | eai (behavior_goal, virtualhome_goal) | 438 |
| 17 | 指令理解 › 内容理解 | **否定理解**<br>Negation understanding | P0 | 正确区分"不要拿红色的"和"拿不是红色的"的行为差异 | negvqa (paired) | 91 |
| 18 | 指令理解 › 内容理解 | **条件指令理解**<br>Conditional instructions | P0 | 理解带条件的指令,比如"如果杯子是空的就去装水,否则放回原处" | momagraph (effect_prediction) | 168 |
| 19 | 指令理解 › 指代理解 | **属性指代**<br>Attribute reference | P0 | 属性指代序数指代(第三本书等。。。) | refspatial (attribute_reference); pio (attribute_reference) | 98 |
| 20 | 指令理解 › 指代理解 | **序数指代**<br>Ordinal reference | P0 | 按序数/次序表达定位目标(第三本、第2个、最左边第二个) | refspatial (ordinal_reference); pio (ordinal_reference) | 22 |
| 21 | 因果理解 › 逻辑因果 | **因果链推理**<br>Causal-chain reasoning | P0 | 理解多个动作或事件之间的连续因果关系,识别间接效果和连锁反应 | causalphys (multi_hop) | 83 |
| 22 | 因果理解 › 逻辑因果 | **反事实推理**<br>Counterfactual reasoning | P1 | 推理"如果不执行某个动作会怎样",用于评估动作的必要性和替代路径,"如果不先打开盖子,还能倒出水吗?" | egotaskqa (counterfactual) | 81 |
| 23 | 因果理解 › 逻辑因果 | **干预推理**<br>Interventional reasoning | P1 | 判断针对某个中间状态的干预能否改变最终结果,比如"在这一步阻止它还来得及吗" | causalphys (action_substitution, spatial_manipulation, temporal_shifting) | 197 |
| 24 | 因果理解 › 逻辑因果 | **归因理解**<br>Attribution | P0 | 给定观察到的结果状态,向上推断是什么原因或操作导致了当前状态 | causalphys (mechanics_reasoning); egotaskqa (explanatory) | 117 |
| 25 | 因果理解 › 逻辑因果 | **失败定位与原因分析**<br>Failure localization & diagnosis | P0 | 确定失败的原因,以及发生在哪个具体步骤或操作阶段 | robobench (high_level_error, low_level_error); eriq (mistake_classify_lenient) | 256 |
| 26 | 科学理解 › 物理规律 | **物理知识**<br>Physical knowledge | P1 | 模型是否掌握支撑具身推理的基础物理规律(重力、摩擦、守恒等) | phyx (Mechanics_manip) [C] | 39 |

### 基础智能能力 › 世界推演 / Foundational intelligence › World simulation

| # | 三级 › 四级 / Level 3 › 4 | 能力 / Capability | 优先级 / Priority | 典型问题 / Example | 来源（子任务） / Sources (subtasks) | 题数 / #Q |
|---|---|---|---|---|---|---|
| 27 | 状态变化推演 › 任务与语义状态 | **单体/多体状态理解**<br>Single/multi-object state | P0 | 当前正在发生什么?整个过程发生了什么 | egotaskqa (state_change); eriq (action_understanding_yn) | 268 |
| 28 | 状态变化推演 › 任务与语义状态 | **子任务分解**<br>Subtask decomposition | P0 | 泡一杯咖啡需要哪些步骤 | robobench_plan (robobench_plan20_single_arm, robobench_plan20_dual_arm, robobench_plan20_human, robobench_plan20_mobile_manipulation, robobench_plan20_articulated, robobench_plan20_deformable, robobench_plan20_multi_object, robobench_plan20_rigid, robobench_plan20_special, robobench_plan20_color, robobench_plan20_number, robobench_plan20_shape, robobench_plan20_size, robobench_plan20_world_knowledge, robobench_plan20_multi_view, robobench_plan20_single_view) | 320 |
| 29 | 状态变化推演 › 任务与语义状态 | **指令顺序理解**<br>Instruction order | P0 | 识别多步指令的先后约束关系 | momagraph (action_sequence) | 343 |
| 30 | 状态变化推演 › 任务与语义状态 | **物理可行性**<br>Physical feasibility | P0 | 判断计划中的动作在物理上是否可以执行,比如目标是否可达、操作是否违反运动约束 | mvrobobench (trajectory) | 198 |
| 31 | 状态变化推演 › 任务与语义状态 | **约束满足检查**<br>Constraint satisfaction | P0 | 在规划任务步骤时确保所有约束都被满足,没有任何步骤违反任务要求 | eai (action_seq) | 305 |
| 32 | 状态变化推演 › 任务与语义状态 | **进度估计**<br>Progress estimation | P0 | 估计当前任务完成了多少比例,还剩哪些步骤未执行,进度百分比(VLM对具身模型进行评估的关键能力) | eriq (task_progress_lenient) | 69 |
| 33 | 状态变化推演 › 任务与语义状态 | **失败检测**<br>Failure detection | P0 | 实时识别当前操作是否已偏离预期轨迹或任务是否已失败 | eriq (mistake_existence_lenient, success_detection_lenient) | 279 |
| 34 | 状态变化推演 › 任务与语义状态 | **恢复策略**<br>Recovery strategy | P0 | 根据失败原因制定恢复方案,选择重试、替代路径或任务降级 | eriq (mistake_recovery_lenient) | 50 |
| 35 | 状态变化推演 › 物理状态 | **物体恒常性**<br>Object permanence & conservation | P1 | 物体被遮挡或离开视野后仍然存在判断场景中物体的数量和质量是否符合守恒预期,识别凭空消失或出现的异常 | intphys2 (permanence); corecognition (permanence); corecognition (conservation) | 338<br>71 |
| 36 | 状态变化推演 › 潜在状态 | **安全监控**<br>Safety monitoring | P0 | 在执行过程中持续监控动作路径上是否存在碰撞风险或其他危险后果:是否可能造成人员伤害、物体损坏或违反安全规范 | pio (safety); asimov (hsm) | 412 |
| 37 | 运动学推演 › 运动轨迹推演 | **轨迹理解与预测**<br>Trajectory understanding & prediction | P0 | 理解物体当前的运动方向并预测其未来路径 | causalphys (collision_prediction); vlm4d_cap37 (ego, exo) | 448 |
| 38 | 运动学推演 › 运动轨迹推演 | **运动连续性**<br>Motion continuity | P1 | 判断物体的运动轨迹是否连贯,识别出违反物理规律的跳变或突变 | intphys2 (continuity) | 240 |
| 39 | 动力学推演 › 力学效果预测 | **单体力学效果**<br>Single-body mechanics | P0 | 预测施加力或发生碰撞后物体的运动和形变结果,比如松手后会怎么样 | physbench (single_body_dynamics) | 187 |
| 40 | 动力学推演 › 力学效果预测 | **多体交互预测**<br>Multi-body interaction | P1 | 多个物体发生交互时的联动结果,比如推倒一个物体引发的连锁反应,机械传动 | physbench (multi_body_collision) | 68 |
| 41 | 动力学推演 › 材料动力学特性 | **刚体、流体、柔体**<br>Rigid / fluid / deformable | P1 | 软体形变和弹性回弹,水位测试 | physbench (fluid_prediction); causalphys (deformation, fluid_flow) | 306 |
| 42 | 空间几何推演 › 多视角一致性推演 | **多视角匹配**<br>Multi-view matching | P1(P0) | A视图中的物体在B视图中的位置 | mvrobobench (cross_view) | 198 |
| 43 | 空间几何推演 › 多视角一致性推演 | **第三方视角理解**<br>Third-person perspective | P1 | 从说话人视角出发的前后左右 | omnispatial (Perspective_Taking) [C] | 200 |
| 44 | 空间几何推演 › 三维占用与重建推演 | **2D/3D理解与转换**<br>2D/3D understanding & conversion | P1 | 从2D图像/视频恢复3D结构:深度、占用、度量尺寸 | vsibench (object_size_estimation, room_size_estimation) | 265 |

### 适应学习能力 › 学习进化 / Adaptive learning › Learning & evolution

| # | 三级 › 四级 / Level 3 › 4 | 能力 / Capability | 优先级 / Priority | 典型问题 / Example | 来源（子任务） / Sources (subtasks) | 题数 / #Q |
|---|---|---|---|---|---|---|
| 45 | 自主探索 › 信息收集与知识检索 | **场景完整判断与处置**<br>Scene completeness judgment | P1 | 关键操作区域是否在视野内 | crosspoint (Visibility Reasoning) [C] | 200 |
| 46 | 自主探索 › 信息收集与知识检索 | **低视觉质量判断与处置**<br>Low visual quality judgment | P1 | 是否有遮挡光照是否不足 | asimov (gauge_judge_readable, gauge_judge_unreadable) | 162 |
| 47 | 自主探索 › 信息收集与知识检索 | **知识检索与求助**<br>Knowledge retrieval & asking for help | P1 | 是否能询问必要信息和检索必要的知识 | mmupd (aad, iasd, ivqd); aqua (level0, level1, level2, level3) | 150<br>100 |
