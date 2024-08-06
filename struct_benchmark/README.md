# 📚 About

This is the guidance for evaluating existing models on **StructEval benchmarks**. 
Currently, we implement StructEval on 3 seed benchmarks including MMLU, ARC-Challenge and OpenbookQA. And we select GPT-4o-mini for generation tasks and the latest version of Wikipedia as knowledge source to minimize the risk of data contamination.
The test instances can be found in `struct_data` folder.


# 🔥  Quick Start
For facilitate evaluation, we have adapted StructEval to [Opencompass 2.0](https://github.com/open-compass/OpenCompass/), making it easy to quickly evaluate multiple models.

## 💻 Environment Setup

```bash
cd struct_benchmark
conda create --name structbench python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
pip install -e 'git+https://github.com/c-box/opencompass.git#egg=opencompass'
```

## 📂  Model Config

Select the model you want to evaluate and add it to the corresponding config file in `eval_config` folder.
The predefined model configurations from OpenCompass can be found in `model_configs`.
For instance, if you want to evaluate the `llama3-8b` model on the all 3 datasets, you just need to find `eval_config/eval_struct_all_v1_ppl.py` and import the corresponding model configuration:

```python
from mmengine.config import read_base

with read_base():
    from ..data_config.struct_arc_challenge.struct_arc_challenge_v1_ppl import struct_arc_challenge_v1_datasets
    from ..data_config.struct_openbook.struct_openbook_v1_ppl import struct_openbookqa_v1_datasets
    from ..data_config.struct_mmlu.struct_mmlu_v1_ppl import struct_mmlu_V1_datasets
    from model_configs.hf_llama.hf_llama3_8b import models as hf_llama3_8b_model
    
datasets = [*struct_arc_challenge_v1_datasets, *struct_openbookqa_v1_datasets, *struct_mmlu_V1_datasets]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
```

## ✏️ Evaluation

```bash
python run.py eval_config/eval_struct_all_v1_ppl.py -w output/struct_all_v1_ppl 
```

Then the evaluation results will be saved in `output/struct_all_v1_ppl`.

#  Leaderboard

Please refer to [StructEval leaderboard](https://huggingface.co/spaces/Bowieee/StructEval_leaderboard) for our current leaderboard, which will be regularly updated. We also welcome everyone to submit new evaluation results.

# 📒 Notes
* To enhance the quality and difficulty of instances, we establish a comprehensive pool of diverse LMs. Currently, this LLM pool consists of 18 LLMs including `llama-2-7B`, `llama-2-7B-chat`, `llama-3-8B`, `llama-3-8B-chat`, `mistral-7b-v0.3`, `mistral-7b-instruct-v0.3`, `qwen1.5-7b`, `qwen1.5-7b-chat`, `qwen2-7b`, `qwen2-7b-instruct`, `deepseek-v2-lite`, `deepseek-v2-lite-chat`, `yi-6b`, `yi-6b-chat`, `yi-1.5-9b`, `yi-1.5-9b-chat`, `baichuan2-7b-base`, `baichuan2-7b-chat`. *Questions that more than 12 models could answer correctly were eliminated*, thus ensuring the discriminative efficacy.

* Considering the cost of API, we randomly sampled 50 samples from each subject of MMLU as seed instances. 


