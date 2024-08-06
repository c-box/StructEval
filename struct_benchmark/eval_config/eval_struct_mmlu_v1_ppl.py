from mmengine.config import read_base

with read_base():
    from ..data_config.struct_mmlu.struct_mmlu_v1_ppl import struct_mmlu_V1_datasets
    from ..model_configs.hf_llama.hf_llama3_8b import models as hf_llama3_8b_model
    

datasets = [*struct_mmlu_V1_datasets]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
