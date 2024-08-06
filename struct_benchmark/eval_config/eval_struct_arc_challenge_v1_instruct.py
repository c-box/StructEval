from mmengine.config import read_base

with read_base():
    from ..data_config.struct_arc_challenge.struct_arc_challenge_v1_instruct import struct_arc_challenge_v1_datasets
    from model_configs.hf_llama.hf_llama3_8b_instruct import models as hf_llama3_8b_instruct_model
    
datasets = [*struct_arc_challenge_v1_datasets]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])