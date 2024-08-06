from common_utils.utils import load_json_dic, save_json_dic, load_file
from common_utils.constant import MMLU_TASKS
from tqdm import tqdm
import random
import os
import argparse


def filter_bloom_data(bloom_questions, idx):
    global all_question
    global right_question
    
    
    new_data = {}
    for level in bloom_questions:
        new_data[level] = []
        for question_id, line in enumerate(bloom_questions[level]):
            all_question += 1
            if line["RAG"]["ans"]:
                
                line["question_idx"] = "{}-{}-{}-{}".format(idx, "bloom", level, question_id)
                
                new_data[level].append(line)
                right_question += 1
    return new_data
    
    

def filter_concept_data(concept_questions, idx):
    global all_question
    global right_question
    
    new_data = []
    for entity_id, line in enumerate(concept_questions):
        line["entity_idx"] = entity_id
        entity_name = line["name"]
        filter_questions = []
        for question_id, question in enumerate(line["concept_questions"]):
            all_question += 1
            if question["RAG"]["ans"]:
                question["question_idx"] = "{}-{}-{}-{}".format(idx, "concept", entity_id, question_id)
                filter_questions.append(question)
                right_question += 1
        
        line["concept_questions"] = filter_questions
        new_data.append(line)
    return new_data
    


def data_combine(benchmark, split="test", task=None, model_type="gpt-4o-mini"):
    if task is not None:
        bloom_data = load_json_dic(f"processed_data/{benchmark}/{task}/5_{split}_with_bloom_question_{model_type}_rag.json")
        concept_data = load_json_dic(f"processed_data/{benchmark}/{task}/e_{split}_with_concept_question_{model_type}_rag.json")
        save_data_path = f"struct_data/{benchmark}/{task}/struct_{split}_{model_type}.json"
        if not os.path.exists(f"struct_data/{benchmark}/{task}"):
            os.makedirs(f"struct_data/{benchmark}/{task}")
    else:
        bloom_data = load_json_dic(f"processed_data/{benchmark}/5_{split}_with_bloom_question_{model_type}_rag.json")
        concept_data = load_json_dic(f"processed_data/{benchmark}/e_{split}_with_concept_question_{model_type}_rag.json")
        save_data_path = f"struct_data/{benchmark}/struct_{split}_{model_type}.json"
        if not os.path.exists(f"struct_data/{benchmark}"):
            os.makedirs(f"struct_data/{benchmark}")
    
    
    
    new_data = []
    for concept_sample in tqdm(concept_data):
        flag = False
        for bloom_sample in bloom_data:
            if concept_sample["idx"] == bloom_sample["idx"]:
                flag = True
                concept_sample["entities"] = filter_concept_data(concept_sample["entities"], concept_sample["idx"])
                bloom_sample["bloom_questions"] = filter_bloom_data(bloom_sample["bloom_questions"], bloom_sample["idx"])
                
                concept_sample["topic"] = bloom_sample["topic"]
                concept_sample["topic_wiki_info"] = bloom_sample["topic_wiki_info"]
                concept_sample["bloom_questions"] = bloom_sample["bloom_questions"]
                new_data.append(concept_sample)
                
                break
        
        if flag is False:
            concept_sample["bloom_questions"] = {}
            concept_sample["entities"] = filter_concept_data(concept_sample["entities"], concept_sample["idx"])
            new_data.append(concept_sample)
            
    save_json_dic(new_data, save_data_path)
    
    
        
if __name__ == "__main__":
    
    all_question = 0
    right_question = 0
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    
    if args.benchmark == "mmlu":
        for task in MMLU_TASKS:
            data_combine(benchmark=args.benchmark, task=task, split=args.split)
    else:
        data_combine(benchmark=args.benchmark, split=args.split)
