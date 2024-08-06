from mmengine.config import read_base

with read_base():
    from ..data_config.struct_arc_challenge.struct_arc_challenge_v1_instruct import struct_arc_challenge_v1_datasets
    from ..data_config.struct_openbook.struct_openbook_v1_instruct import struct_openbookqa_v1_datasets
    from ..data_config.struct_mmlu.struct_mmlu_v1_instruct import struct_mmlu_V1_datasets
    from model_configs.hf_llama.hf_llama3_8b_instruct import models as hf_llama3_8b_instruct_model
    
datasets = [*struct_arc_challenge_v1_datasets, *struct_openbookqa_v1_datasets, *struct_mmlu_V1_datasets]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
