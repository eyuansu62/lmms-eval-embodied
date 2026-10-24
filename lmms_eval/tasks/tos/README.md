# Theory of Space（主动探索多轮，实验性）

数据：[MLL-Lab/Theory-of-Space](https://huggingface.co/datasets/MLL-Lab/Theory-of-Space)（CC-BY-4.0），100 个三房间场景，每个可站立位置 × 4 个朝向的图全部预渲染（768×768）。
代码：[mll-lab-nu/Theory-of-Space](https://github.com/mll-lab-nu/Theory-of-Space)（MIT），固定提交 `d48f52f`。
论文：Zhang et al., *Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?*（ICLR 2026）。

```
python lmms_eval/tasks/tos/build_tos.py   # 数据解压到 media/，官方代码到 vendor/Theory-of-Space，依赖到 vendor/deps
```

**这不是能力表的一部分。** 用来测「主动探索」：模型自己决定往哪走、朝哪看、什么时候停，再用自己收集到的信息答空间题。

## 任务

| 任务 | 轮数 | 做法 |
|---|---|---|
| `tos_active` | ≤ 21 + 1 | 模型探索（`JumpTo` / `Rotate` / `Observe` / `Query` / `Term`，最多 20 步），然后在完整探索对话之后答题：27 道各一次独立请求，同一轮并发发出 |
| `tos_passive` | 1 | 官方 scout 代理先探索好，模型只答题 |

两个任务的 27 道题完全相同（9 类 × 3 题，由场景和 seed 决定），所以 `active − passive` 就是「自己探索」的代价，
这也是对本任务的捷径体检：如果 active 不比 passive 低，说明主动这一层没有测到东西。题目都是开放作答
（方向 + 距离、坐标、动作序列、物体名），不是选择题，乱猜得分接近 0（固定答 “north, near” 得 1.9%）。

环境、出题、判分全部调用官方代码；这里只把它接到 lmms-eval 的多轮接口上。官方协议是真正的多轮对话
（system + 多条 user / assistant，图片在文字中内联），因此 `evaluator._run_generate_until_agentic` 增加了「完整历史」模式：
任务给出整段对话而不是单条 user 消息，缓存键取整段对话（图片按文件名）；一轮也可以给出多段对话，同时发出（扇出）。

## 指标

- `tos_accuracy`：27 题平均得分（主指标）；`tos_route` / `tos_survey`：论文的两层分组（route：pov、bwd_pov_text、fwd_fov、bwd_nav_text；survey：dir、e2a、rot、fwd_loc、bwd_loc_text）。
- `tos_action_cost`：探索代价（Observe 1、Query 2）；`tos_node_coverage`：探索覆盖率（官方 `get_exp_summary`）。passive 下这两项是 scout 代理的。
- `tos_complete`：在步数上限内走完探索和全部问答的比例；没走完的场景 27 题全记 0 分。

## 与论文的差异

- 只做视觉世界；文字世界、认知地图探针、false-belief 修正任务未接。
- 官方脚本默认每次跑 25 个场景（`--num 25`），这里 100 个场景都在数据里，可用 `--limit` 控制。
- 图片：探索阶段 384×384、评测阶段原图，同官方；编码由 lmms-eval 负责，请求超过 6 MB 时转 JPEG（官方本就发 JPEG q85）。

## 结果（2026-10-24，前 25 个场景，每种设定 675 题）

| 模型 | passive | active | active − passive（按场景配对，95% 区间） | active 探索代价 | active 覆盖率 |
|---|---|---|---|---|---|
| qwen3.8-flash | 68.6 | 61.5 | −7.1 [−11.3, −3.2] | 14.2 | 0.95 |
| qwen3.8-27b | 62.0 | 43.5 | −18.5 [−24.2, −12.7] | 15.4 | 0.98 |
| doubao-seed-2.1-pro | 67.2 | 65.4 | −1.8 [−4.6, +0.9] | 9.1 | 1.00 |

passive 下的探索代价是 scout 代理的 8.9。

读法：
- **自己探索有代价，且因模型而异**：三个模型 active 都低于 passive，与论文结论一致；flash、27b 的差距显著，doubao 不显著。
- **差距不在覆盖率**：三个模型都几乎看全了（0.95–1.00），差在探索方式。doubao 的代价（9.1）与 scout 代理相当，差距最小；27b 走了 15 步以上，覆盖率最高，分数却掉得最多——冗长、无序的探索历史本身会拖累推理。
- route 层（自我中心、动作相关）比 survey 层（全局地图）掉得更多。

注意：最初跑完时 27b passive 报 56.3%，因为有 3 个场景的答案里出现了 `</think>`，评测器默认的思考剥离从那里截断了结果 JSON，整场记 0；
按原始输出重算为 62.0%（上表）。`utils.py` 已改为写 JSON 时转义 `<`，不会再发生。
