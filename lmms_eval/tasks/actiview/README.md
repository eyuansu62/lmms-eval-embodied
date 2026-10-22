# ActiView（主动感知多轮，实验性）

数据：[wangphoebe/ActiView](https://huggingface.co/datasets/wangphoebe/ActiView)（Apache-2.0），325 题，8 类问题。
论文：Wang et al., *ActiView: Evaluating Active Perception Ability for Multimodal Large Language Models*（ACL 2025）。

```
python lmms_eval/tasks/actiview/build_actiview.py   # 解压 1575 张图到 media/，写 data/actiview.jsonl
```

**这不是能力表的一部分。** 用来把「多轮交互」这种评测形态建起来：模型要自己决定看哪里、还缺什么，
环境（这里是 `utils.py` 里的状态机）按它的要求给图，再让它作答。

## 任务

| 任务 | 轮数 | 做法（论文附录 J） |
|---|---|---|
| `actiview_full` | 1 | 全图直接作答（J.1），对照组 |
| `actiview_zoom` | 2 | 看全图，选要放大的切片（1–4，可多选）；再看全图 + 所选切片作答（J.2） |
| `actiview_shift_{easy,medium,hard}` | 2–4 | 只给一个切片，每轮问「还要看别的吗」；答 NO 或四片看全后作答（J.3） |

shift 三档的初始切片来自 `shifting_level.json`：easy 初始切片含完整线索（325/325 落在 `groundtruth_views` 里），medium 只含部分线索（241/325 落在里面），
hard 不含线索（例外是四个切片都有线索的 105 题，这些题 hard 档的初始切片也含线索，三档难度没有区别）。
模型点名的切片无法解析时，补一个相邻切片（论文：“an adjacent view will be given until the model can answer”），并记一次 invalid。

## 指标

- `actiview_accuracy`：答对率（主指标）。
- `actiview_num_views`：最终用到的切片数（zoom = 选中的，shift = 看过的）；full 恒为 0。
- `actiview_clue_recall`：用到的切片覆盖了多少比例的线索切片（`groundtruth_views.json`）；full 无意义（恒 0）。
- `actiview_invalid_rate`：至少一轮选片回复无法解析的题目比例。
- `actiview_parse_rate`：最终答案能抽出字母的比例。

## 与论文的差异

- 官方评测代码未公开，提示词照论文附录 J 原文；选项改成 `(A) ...` 字母形式，结尾加 “Answer with the option letter only.”。
- 图片长边限 1600 px（原图最大 6327×6066，5 张图 PNG 编码会超网关请求体上限）。切片本已放大到全图尺寸，限幅后仍保留放大的意义。
- 没做 mixed 流程（J.4）和 Shift-R。
