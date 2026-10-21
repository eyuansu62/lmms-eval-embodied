# NegVQA · 自然场景筛选切片

具身智能 VLM 评测 —— **视觉否定理解**。来自 能力体系（CAPABILITIES.md） 的 P0 条目 #11。

- 论文：https://arxiv.org/abs/2505.22946（ACL 2025 Findings）
- 代码：https://github.com/yuhui-zh15/NegVQA
- 数据：https://huggingface.co/datasets/yuhuizhang/NegVQA

## 为什么不用 NegBench

原方案要接 NegBench，但它不适配生成式 VLM：

| 维度 | NegBench 的实际情况 |
|---|---|
| 评测协议 | CLIP 嵌入相似度取最大 —— **不是生成式作答** |
| 答案位置 | `create_mcq.py` 恒定把正确答案放 `caption_0`，无随机化 |
| 数据 | 不在仓库里，要自己下 COCO/VOC2007/MSR-VTT 并用 vLLM 三步合成 |

答案恒定放首位，对 CLIP 无所谓（它算相似度），但搬成生成式评测的话
**模型一律选 A 就能拿分**。改造它等于"用 NegBench 的素材重建一个生成式版本"，
会偏离官方协议。

NegVQA 三个条件都满足：原生两选项 VQA、模型直接生成选项字母、
官方已在 VLMEvalKit 上评过 Qwen2-VL / LLaVA / Molmo 等 20 个模型。

## 为什么不用 NegVQA 全量

NegVQA 的否定是 **LLM 自动改写**生成的，答案机械翻转。实测 dev 全量 1000 例
的答案是**完美一一翻转**（488A/512B → 488B/512A），说明缺陷是系统性的。

核实到的两类缺陷：

**① 禁止行为 ≠ 预测错值**
```
index=326 (MMMU)
  original: ...Predict the overtime wages if 6,500 invoices are processed.      -> B
  negated : ...Do not predict the overtime wages if 6,500 invoices are processed. -> A
```
"不要预测"是禁止*预测行为本身*，不等于"预测一个错误数值"，答案翻转不成立。

**② 数量类的补集不是另一个数**
```
How many bananas are there in the image?   A: 4   B: 2
negated: How many bananas are not there in the image?  -> 答案标为 4
```
"图里有几个 X"的补集不是另一个数字。这类比 ① 更隐蔽 —— 语法通顺，
只有细看语义才发现问题。

**③ 语法不通**
```
What type of book is this?  ->  What type of book is this not?
```

## 筛选规则（全部自动，从严）

`screen_negvqa.py`，四层过滤。**dev 1000 例 → 保留 91 例。**

| 层 | 规则 | 淘汰 | 剩余 |
|---|---|---|---|
| 1 | 只保留自然场景来源 | −549 | 451 |
| 2 | 否定句式构成可判定二选一 | −312 | 139 |
| 3 | 答案确实翻转且在 {A,B} 内 | −0 | 139 |
| 4 | 排除数量类 | −48 | **91** |

**第 1 层：来源。** 只保留
`OKVQA / A-OKVQA / GQA / VQAv2 / RealWorldQA / VizWiz / SEEDBench / MMStar`。
排除文档图表（InfoVQA、DocVQA、ChartQA、TableVQABench、OCRVQA、TextVQA）
与学科推理（AI2D、ScienceQA、MathVision、MathVista、MMMU、MMVet）。

依据是离线用官方预测实测的**落差**（原题 acc − 否定 acc）：

| 档位 | n | 原题 acc | 否定 acc | 成对全对 | 落差 |
|---|---|---|---|---|---|
| 全部 | 1000 | 0.593 | 0.438 | 0.271 | +0.155 |
| **自然场景** | 451 | 0.705 | 0.441 | 0.337 | **+0.264** |
| 文档图表 | 329 | 0.547 | 0.450 | 0.243 | +0.097 |
| 学科推理 | 220 | 0.432 | 0.414 | 0.177 | **+0.018** |

三档差 15 倍。学科推理档落差近零（模型在那儿本来就答不好，原题只有 0.432），
混入只会稀释"否定理解"这个信号。

**第 2 层：否定句式。** 只保留否定后**有唯一答案**的形式：

- 保留：`not in/on/present/visible/shown/parked/...`（存在性补集）
- 排除：`is not the [X]`（"不是什么"开放无解，68 例）
- 排除：句末缀 `not`（语法不通，45 例）
- 排除：`do not predict/answer/...`（禁止行为句式）

**第 4 层：排除数量类。** `how many/much` 或以纯数字为选项的题一律不要。
留下来的全是**互斥二选一**：朝向、位置、类别、月相、材质这类
"选了 A 就不可能是 B"的属性题。只有在这些题上，"不是 A"才等价于"是 B"。

> ⚠️ 这是**自定义评测切片**，不是官方子集。引用时请注明筛选规则。

## 三个指标

`lmms-eval` 的 task 是「一 doc 一问」，没法让同一条样本在一次运行里问两遍。
所以两个 task 分开跑（与官方 `evaluate_models.sh` 的
`NegVQA-original` / `NegVQA-negated` 做法一致），再用配对脚本补第三个指标。

| task | 问什么 |
|---|---|
| `negvqa_natural_scene_original` | 原题 |
| `negvqa_natural_scene_negated` | 否定题 |

配对脚本 `tools/pair_negvqa_results.py` 按 `index` 对齐两边，算出：

- 原题准确率
- 否定题准确率
- **成对全对率** —— 同一视觉内容下两问都答对

成对全对率是这套评测的关键指标：它排除"靠猜"。若模型在两问上独立乱猜
（各 50%），期望成对全对率约 25%；单看否定准确率则可能被运气掩盖。

## 用法

```bash
conda activate lmms-eval
cd <repo-root>

# 两个 task 一起跑，落到同一目录
CUDA_VISIBLE_DEVICES=4 python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks negvqa_natural_scene_original,negvqa_natural_scene_negated \
  --batch_size 1 --log_samples --output_path ./results

# 配对算成对全对率
python lmms_eval/tasks/negvqa/tools/pair_negvqa_results.py results/
```

`--log_samples` 是必须的 —— 配对脚本读 samples jsonl。

跑之前先确认有空闲 GPU —— 这台机器上 0/1/2/3/6/7 常年被占满，
4 和 5 通常空闲。

## 目录内容

```
negvqa/
├── screen_negvqa.py                            # 筛选脚本（四层规则）
├── utils.py                                    # 题面构造 + 判分
├── utils_filter.py                             # process_docs 按索引过滤
├── negvqa_natural_scene_original.yaml
├── negvqa_natural_scene_negated.yaml
├── tools/
│   └── pair_negvqa_results.py                  # 配对算成对全对率
├── README.md
└── data/
    ├── dev-00000-of-00001.parquet              # 原始 dev（图片内嵌，99 MB）
    ├── negvqa_natural_scene_dev.jsonl          # 筛选产物（91 例）
    └── negvqa_natural_scene_dev_indices.json   # 保留索引，yaml 靠它过滤
```

数据源用原始 parquet（图片内嵌其中，不便单独落盘），
yaml 里 `dataset_path` 指向 parquet，靠 `utils_filter.process_docs_screened`
按索引过滤。

## 验证状态

- ✅ 2/2 task 注册
- ✅ 筛选四层规则可复跑（`screen_negvqa.py`）
- ✅ 保留样本人工抽验：0 例数量类残留，全为互斥二选一
- ✅ 端到端冒烟（91 例 × 2 task）
- ✅ 配对脚本产出成对全对率

冒烟成绩（Qwen2.5-VL-3B-Instruct，**全量 91 例**）：

| 指标 | 值 |
|---|---|
| 原题准确率 | 0.9121 |
| 否定题准确率 | 0.4396 |
| **成对全对率** | **0.4176** |
| 落差 | +0.4725 |

四象限分布：

| | 否定✓ | 否定✗ |
|---|---|---|
| **原题✓** | 38 (41.8%) | **45 (49.5%)** |
| **原题✗** | 2 (2.2%) | 6 (6.6%) |

**49.5% 是"原题答对、否定答错"，反向只有 2.2%。** 这是真实的否定失效 ——
若模型在乱猜，两个方向应当对称。接近随机的否定准确率（0.4396）配上
极高落差，说明模型严重依赖原题中的肯定表述。

## 与表格的对应

你建议的替换（`修订映射` G21）：

> **NegVQA：视觉否定问答，自然场景筛选＋人工审题，作为基础补测。**
> **行为否定最小对照题：区分动作禁止、目标对象否定与任务义务，作为主测，待构建。**

本目录实现的是**前半句**（基础补测）。

⚠️ **注意本切片的覆盖范围**：筛选后的题是"**选择不符合某属性的对象**"
这一类否定（朝向/位置/类别）。它**不覆盖**"**不要拿那个对象**"这类
**动作禁止** —— 那需要单列专项，不能靠替换 benchmark 名字就算覆盖完成。

## 待办

- [ ] 接 test split（6379 例，635 MB）—— 走同一套筛选脚本
- [ ] 行为否定最小对照题（动作禁止 / 目标对象否定 / 任务义务），需另行构建
- [ ] Ref-Adv 作为"找到不是红色的那个对象"的补测（公开 1142 条 Ref-Adv-s）
- [ ] 筛完 test 后更新 `utils_filter._INDEX_FILES` 的优先级
