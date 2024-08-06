from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator, AccEvaluator
from opencompass.datasets import StructOpenbook_V1


openbook_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D', "question_idx"],
    output_column='target',
    train_split='dev',
    test_split="test"
)


_hint = f'The following are multiple choice questions (with answers).\n\n'
question_overall = '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
openbook_infer_cfg = dict(
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

openbook_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

struct_openbookqa_v1_datasets = [
    dict(
        abbr="struct_openbookqa_v1",
        type=StructOpenbook_V1,
        path="./struct_data/struct_openbookqa",
        reader_cfg=openbook_reader_cfg,
        infer_cfg=openbook_infer_cfg,
        eval_cfg=openbook_eval_cfg,
    )
]