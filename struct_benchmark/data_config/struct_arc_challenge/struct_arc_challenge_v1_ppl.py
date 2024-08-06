from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator, AccEvaluator
from opencompass.datasets import StructARC_V1

arc_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev',
    test_split="test"
)


_hint = f'The following are multiple choice questions (with answers).\n\n'
question_overall = '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
arc_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={opt: f'{question_overall}\nAnswer: {opt}\n' for opt in ['A', 'B', 'C', 'D']},
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template={opt: f'{_hint}</E>{question_overall}\nAnswer: {opt}' for opt in ['A', 'B', 'C', 'D']},
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )

arc_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

struct_arc_challenge_v1_datasets = [
    dict(
        abbr="struct_arc_challenge_v1",
        type=StructARC_V1,
        path="./struct_data/struct_arc_challenge",
        reader_cfg=arc_reader_cfg,
        infer_cfg=arc_infer_cfg,
        eval_cfg=arc_eval_cfg,
    )
]