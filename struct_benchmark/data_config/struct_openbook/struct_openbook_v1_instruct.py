from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator, AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import StructOpenbook_V1


openbook_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev',
    test_split="test"
)


_hint = f'There is a single choice question. Answer the question by replying A, B, C or D.'

openbook_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
            ),
            dict(role='BOT', prompt='{target}\n')
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(
                    role='HUMAN',
                    prompt=f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                ),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

openbook_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')
)


struct_openbookqa_v1_datasets = [
    dict(
        abbr="struct_openbookqa_v2",
        type=StructOpenbook_V1,
        path="./struct_data/struct_openbookqa",
        reader_cfg=openbook_reader_cfg,
        infer_cfg=openbook_infer_cfg,
        eval_cfg=openbook_eval_cfg,
    )
]
