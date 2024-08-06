from mmengine.config import read_base

with read_base():
    from ..data_config.struct_openbook.struct_openbook_v1_ppl import struct_openbookqa_v1_datasets
    
    from model_configs.hf_llama.hf_llama3_8b import models as hf_llama3_8b_model


datasets = [*struct_openbookqa_v1_datasets]
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
