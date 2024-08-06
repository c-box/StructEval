![logo](/asset/logo.png)


<p align="center">
    <a href="https://huggingface.co/spaces/Bowieee/StructEval_leaderboard"><img src="https://img.shields.io/badge/%F0%9F%8F%86-leaderboard-8A2BE2"></a>
    <!-- <a href="https://arxiv.org/abs/2407.11470"><img src="https://img.shields.io/badge/arXiv-2407.11470-b31b1b.svg"></a> -->
    <a href="https://github.com/c-box/StructEval/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/evalplus"></a>
</p>


<p align="center">
    <a href="https://huggingface.co/spaces/Bowieee/StructEval_leaderboard">🏆Leaderboard</a> •
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-issues">🐛Issues</a> •
    <a href="#-citation">📜Citation</a>
</p>

## 📣 About

Evaluation is the baton for the development of large language models.
Current evaluations typically employ *a single-item assessment paradigm* for each atomic test objective, which struggles to discern whether a model genuinely possesses the required capabilities or merely memorizes/guesses the answers to specific questions.
To this end, we propose a novel evaluation framework referred to as ***StructEval***. 
Starting from an atomic test objective, StructEval deepens and broadens the evaluation by conducting a **structured assessment across multiple cognitive levels and critical concepts**, and therefore offers a comprehensive, robust and consistent evaluation for LLMs.
Experiments on three widely-used benchmarks demonstrate that **StructEval serves as a reliable tool for resisting the risk of data contamination and reducing the interference of potential biases**, thereby providing more reliable and consistent conclusions regarding model capabilities. 
Our framework also sheds light on the design of future principled and trustworthy LLM evaluation protocols.

This repo provides easy-to-use scripts for both [evaluating LLMs on existing StructEval benchmarks](#️-evaluate-models-on-structeval-benchmarks) and [generating new benchmarks based on StructEval framework](#-generate-new-benchmarks-based-on-structeval-framework).

📰 Read our paper [**StructEval: Deepen and Broaden Large Language Model Assessment via Structured Evaluation**]() to get more information.

![logo](/asset/new_head.png)

## 🚀 News

* [2024.8.6] We released the first version of [StructEval leaderboard](https://huggingface.co/spaces/Bowieee/StructEval_leaderboard), which includes 22 open-sourced language models, more datasets and models as comming soon🔥🔥🔥.

* [2024.7.31] We regenerated the StructEval Benchmark based on the latest [Wikipedia](https://www.wikipedia.org/) pages (20240601) using [GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) model, which could minimize the impact of data contamination. Please refer to the [struct_benchmark](struct_benchmark) folder for our evaluation data and scripts 🔥🔥🔥.

## 🔥 Quick Start

### ✏️ Evaluate models on StructEval benchmarks

For facilitate evaluation, we have adapted StructEval to [Opencompass 2.0](https://github.com/open-compass/OpenCompass/), making it easy to quickly evaluate multiple models.

For instance, if you want to evaluate `llama-3-8b-instruct` model on the `StructMMLU` dataset, you just need import the corresponding model configuration in [struct_benchmark/eval_config/eval_struct_mmlu_v1_instruct.py](struct_benchmark/eval_config/eval_struct_mmlu_v1_instruct.py):
```python
from mmengine.config import read_base
with read_base():
    from ..data_config.struct_mmlu.struct_mmlu_v1_instruct import struct_mmlu_V1_datasets
    from ..model_configs.hf_llama.hf_llama3_8b_instruct import models as hf_llama3_8b_instruct_model
datasets = [*struct_mmlu_V1_datasets]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
```

Then start the following command:
```bash
cd struct_benchmark
python run.py eval_config/eval_struct_mmlu_v1_instruct.py -w output/struct_mmlu_v1_instruct
```
The evaluation results will be saved in `struct_benchmark/output/struct_mmlu_v1_instruct`

Please refer to [struct_benchmark/README.md](struct_benchmark/README.md) for more detailed guidance.


### 🔨 Generate new benchmarks based on StructEval framework

The [struct_generate](struct_generate) folder provides the source code as well as an running example for benchmark construction based on StructEval. Specifically, StructEval consists of two modules which deepen and broaden current evaluation respectively.
Given a seed instance, the first module identifies its underlying test objective, and then generates multiple test instances around this test objective which are aligned with the six cognitive levels outlined in Bloom’s Taxonomy. 
Meanwhile, the second module extracts the key concepts that must be understood to answer the seed question, and then develop a series of instances revolving around these concepts.

You can construct a structured benchmark for LLM evaluation based on some seed instances by executing the following command:

```bash
cd struct_generate
bash scripts/run_bloom_generate.bash demo test
bash scripts/run_concept_generation.bash demo test
bash scripts/run_data_combine.bash demo test
```

Please refer to [struct_generate/README.md](struct_generate/README.md) for more detailed guidance.


## 📜 Citation

```bibtex
```
