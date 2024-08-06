from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator, AccEvaluator
from opencompass.datasets import StructMMLU_V1

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev',
    test_split="test"
)


mmlu_all_sets = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

struct_mmlu_V1_datasets = []


for _name in mmlu_all_sets:
    _hint = f'The following are multiple choice questions (with answers) about  {_name.replace("_", " ")}.\n\n'
    question_overall = '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
    mmlu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={opt: f'{question_overall}\nAnswer: {opt}\n' for opt in ['A', 'B', 'C', 'D']},
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template={opt: f'{_hint}</E>{question_overall}\nAnswer: {opt}' for opt in ['A', 'B', 'C', 'D']},
            ice_token='</E>',
        ),
        # retriever=dict(type=FixKRetriever, fix_id_list=[0,1,2]),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer),
    )

    mmlu_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator), )

    struct_mmlu_V1_datasets.append(
        dict(
            abbr=f'struct_mmlu_{_name}',
            type=StructMMLU_V1,
            path='./strcut_data/struct_mmlu',
            name=_name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))

del _name, _hint
