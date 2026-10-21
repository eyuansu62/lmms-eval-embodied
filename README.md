# lmms-eval-embodied

[English](#english) | [中文](#中文)

## 中文

基于 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 的具身智能 VLM 评测套件。新增 20 个 benchmark 接入，并映射到一套 47 项能力体系（世界理解、指令理解、因果与物理推理、失败处理、安全等），一次评测即可得到逐项能力分数。能力体系与每项的来源见 [CAPABILITIES.md](CAPABILITIES.md)。

上游 lmms-eval 的全部功能保持可用；本仓库在其基础上新增任务、修正 OpenAI 兼容后端的若干问题，并提供报表流程。

**安装**

```bash
git clone <this-repo> && cd lmms-eval-embodied
pip install -e .
```

视频任务（TVBench 片段裁剪、IntPhys 2 抽帧）需要 `ffmpeg`。

**准备数据**

本仓库**不附带任何 benchmark 数据**。`prepare_data.py` 从各数据集的原始地址下载，运行任务目录下的构建脚本，再按 `lmms_eval/tasks/<task>/data/MANIFEST.json`（逐题哈希，见 `tools/data_manifest.py`）核对，确保生成的题目与我们评测所用的逐字节一致。

```bash
python prepare_data.py --list         # 需要准备的任务
python prepare_data.py                # 全部
python prepare_data.py tvbench vlm4d  # 指定任务
```

可设置 `HF_ENDPOINT` 使用 Hugging Face 镜像。ASIMOV 为 gated 数据集，需先在 Hugging Face 页面同意条款并 `huggingface-cli login`。ERIQ、MomaGraph-Bench 及上游任务在评测时直接从 Hugging Face 读取，无需准备。

**运行**

任意 lmms-eval 模型都可直接用；OpenAI 兼容端点的 API 模型可用 `run_model.sh` 按我们的设置（帧数、并发、请求体上限、重试）跑完整套件：

```bash
cp .env.example .env                  # 填入端点与 key
./run_model.sh check <model>          # 连通性与模态预检
./run_model.sh all <model>            # A + B 档；allc 再加 C 档
./run_model.sh retry <model>          # 补跑失败题并合并
./run_model.sh judge <model>          # RoboBench 规划题的 LLM judge（GPT-4o，经 OpenRouter）
```

**报表**：`python dump_reports.py` 由全部结果目录生成 `reports/*.csv`。`summary_capability_agg.csv` 为每项能力一行，`summary_capability.csv` 为能力 × 来源明细；读分前请看 `reports/README_capability.md`（计分口径、judge、已知局限）。能力分 = 该能力下全部题目的逐题准确率。

**评测结果**：10 个模型的完整逐题结果：[eyuansu71/lmms-eval-embodied-results](https://huggingface.co/datasets/eyuansu71/lmms-eval-embodied-results)（Hugging Face）。

**数据来源与许可**见下方英文部分的表格。代码沿用 lmms-eval 的 MIT 许可；IntPhys 2 与 TVBench 中的 Charades 片段**仅限非商用**。

---

## English

An embodied-intelligence evaluation suite for vision-language models, built on
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). It adds 20 benchmark integrations and maps them onto a
47-item capability taxonomy (world understanding, instruction understanding, causal and physical reasoning, failure
handling, safety, ...), so that one run produces a per-capability score table. The taxonomy and the sources of every
capability are listed in [CAPABILITIES.md](CAPABILITIES.md).

Everything from upstream lmms-eval still works; this repository adds tasks, a few fixes to the OpenAI-compatible
backend, and the reporting pipeline.

### Install

```bash
git clone <this-repo> && cd lmms-eval-embodied
pip install -e .
```

`ffmpeg` is needed for the video tasks (TVBench clips, IntPhys 2 frames).

### Prepare data

Benchmark data is **not** redistributed. `prepare_data.py` downloads each benchmark from its original source, runs
the task's builder scripts, and checks that the regenerated files match the evaluated subsets byte for byte
(`lmms_eval/tasks/<task>/data/MANIFEST.json` stores per-row hashes; see `tools/data_manifest.py`).

```bash
python prepare_data.py --list         # tasks that need preparation
python prepare_data.py                # all of them
python prepare_data.py tvbench vlm4d  # selected ones
```

Set `HF_ENDPOINT` to use a Hugging Face mirror. ASIMOV is gated: accept its terms on Hugging Face and log in
(`huggingface-cli login`) first. ERIQ, MomaGraph-Bench and the upstream tasks load directly from Hugging Face at
evaluation time and need no preparation.

### Run

Any lmms-eval model works, e.g.

```bash
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks tvbench_cap13_action_sequence,tvbench_cap13_egocentric_sequence --batch_size 1 --output_path results/
```

For API models behind an OpenAI-compatible endpoint, `run_model.sh` runs the whole suite with the settings used for
our results (frame counts, concurrency, request-size limits, retries):

```bash
cp .env.example .env                  # fill in endpoint and key
./run_model.sh check <model>          # connectivity and modality checks
./run_model.sh all <model>            # A + B tiers; ./run_model.sh allc <model> adds the C tier
./run_model.sh retry <model>          # re-run failed requests and merge
./run_model.sh judge <model>          # LLM judge for RoboBench planning questions (GPT-4o via OpenRouter)
```

### Reports

```bash
python dump_reports.py                # writes reports/*.csv from all result directories
```

| File | Content |
|---|---|
| `reports/summary_capability_agg.csv` | one row per capability (47 items; #9 and #35 are split by caliber) |
| `reports/summary_capability.csv` | capability × source benchmark |
| `reports/summary_subtasks.csv` | every evaluated subtask |
| `reports/README_capability.md` | how to read the table: scoring caliber, judges, known limitations (Chinese) |

`map_subtask_to_capability.py` is self-contained (standard library only) and turns `summary_subtasks.csv` into the
two capability tables. A capability score is the per-question accuracy over all questions mapped to it.

Full per-question results for the 10 evaluated models: [eyuansu71/lmms-eval-embodied-results](https://huggingface.co/datasets/eyuansu71/lmms-eval-embodied-results) on Hugging Face.

### Benchmarks

| Benchmark | Tasks | Source | License |
|---|---|---|---|
| AQuA | `aqua_level{0-3}` | [jihyoung/AQuA](https://huggingface.co/datasets/jihyoung/AQuA) | CC BY 4.0 |
| ASIMOV | `asimov_*` | [google/asimov_agentic](https://huggingface.co/datasets/google/asimov_agentic) (gated) | CC BY 4.0 |
| CausalPhys | `causalphys_*` | [haorentang/causalphys](https://huggingface.co/datasets/haorentang/causalphys) | Apache-2.0 |
| CoreCognition | `corecognition_{conservation,permanence}` | [williamium/CoreCognition](https://huggingface.co/datasets/williamium/CoreCognition) | Apache-2.0 |
| Embodied Agent Interface | `eai_*` | PyPI [`eai-eval==1.0.5`](https://pypi.org/project/eai-eval/) | see package |
| EgoTaskQA | `egotaskqa_*` | [eyuansu71/EgoTaskQA](https://huggingface.co/datasets/eyuansu71/EgoTaskQA) (mirror; official: [egotaskqa](https://sites.google.com/view/egotaskqa)) | see official site |
| ERIQ | `eriq_*` | [KineMind/ERIQ](https://huggingface.co/datasets/KineMind/ERIQ) | CC BY-SA 4.0 |
| IntPhys 2 | `intphys2_{permanence,continuity}` | [facebook/IntPhys2](https://huggingface.co/datasets/facebook/IntPhys2) | **CC BY-NC 4.0** |
| MM-UPD | `mmupd_*` | [MM-UPD/MM-UPD](https://huggingface.co/datasets/MM-UPD/MM-UPD) | Apache-2.0 |
| MomaGraph-Bench | `momagraph_*` | [cheryyunl/MomaGraph-Bench](https://huggingface.co/datasets/cheryyunl/MomaGraph-Bench) | MIT |
| MV-RoboBench | `mvrb_*` | [AaronFengZY24/MV_Robobench](https://huggingface.co/datasets/AaronFengZY24/MV_Robobench) | not stated |
| NegVQA | `negvqa_*` | [yuhuizhang/NegVQA](https://huggingface.co/datasets/yuhuizhang/NegVQA) | not stated |
| OpenEQA | `openeqa_*` | [thomas-yanxin/open-eqa-mirror](https://huggingface.co/datasets/thomas-yanxin/open-eqa-mirror) (mirror; official: [facebookresearch/open-eqa](https://github.com/facebookresearch/open-eqa)) | see official repo |
| PhysBench | `physbench_*` | [lmms-lab-eval/PhysBench](https://huggingface.co/datasets/lmms-lab-eval/PhysBench) | Apache-2.0 |
| Point-It-Out | `pio_*` | [xavihart/PIO](https://github.com/xavihart/PIO) | not stated |
| RefSpatial | `refspatial_*` | [JingkunAn/RefSpatial-Expand-Bench](https://huggingface.co/datasets/JingkunAn/RefSpatial-Expand-Bench) | not stated |
| RoboBench | `robobench_*` | [LeoFan01/RoboBench](https://huggingface.co/datasets/LeoFan01/RoboBench) | CC BY 4.0 |
| TVBench | `tvbench_cap13_*` | [FunAILab/TVBench](https://huggingface.co/datasets/FunAILab/TVBench) | CC BY 4.0 (Charades clips: non-commercial) |
| VLM4D | `vlm4d_cap37_{ego,exo}` | [shijiezhou/VLM4D](https://huggingface.co/datasets/shijiezhou/VLM4D) | MIT |
| VSI-Bench | `vsibench_*` | [nyu-visionx/VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | Apache-2.0 |

Most tasks evaluate a subset of the original benchmark selected for one capability; each task directory documents
its selection rule. Licenses are those declared by the dataset authors — check them before any non-evaluation use.
The C-tier tasks (CrossPoint, MultiHopSpatial, OmniSpatial, RoboSpatial, SPAR-Bench, EmbSpatial, PhyX, EPIC-Bench)
are upstream lmms-eval tasks run on class-balanced samples (`lmms_eval/tasks/c_tier_sampling.py`).

### License

Code: MIT, inherited from lmms-eval (see `LICENSE`). Data: see the table above; this repository contains no benchmark
data, only selection manifests.

### Acknowledgements

Built on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Please cite the original papers of the
benchmarks you use.
