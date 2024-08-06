import argparse
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from common_utils.utils import load_file, load_json_dic, save_jsonl_data, save_json_dic, set_seed, build_example, CHOICES, get_text_chunks, random_select_choice
from common_utils.prompt_utils import query_gpt, multi_query_gpt, parse_json_response, try_to_query, entity_match_multithreaded
import os
from common_utils.constant import MMLU_TASKS
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from common_utils.wiki_search import search_for_wikipedia, search_with_wikipedia_api, trunc_wikipedia_page, rank_gpt, bge_rank, batch_bge_rank, wiki_retrieve_by_entity
import threading
import queue
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


set_seed()



def format_entity_example(data):
    question = data["question"]
    choice = "\n".join(
            [
                "A. " + data["choices"][0],
                "B. " + data["choices"][1],
                "C. " + data["choices"][2],
                "D. " + data["choices"][3],
            ]
        )
    answer = CHOICES[data["answer"]]
    entities = [json.dumps(e, ensure_ascii=False) for e in data["entities"]]
    entity_prompt = "\n".join(entities)
    
    entity_prompt = "```json\n{}\n```".format(entity_prompt)
    
    prompt = f"Question: {question}\nChoices:\n{choice}\nAnswer: {answer}\nThe important entities need to understand about this question include:\n{entity_prompt}"
    return prompt


import re
import demjson
def parse_jsonline_response(response):
    json_pattern = r'```json\n(.*?)\n```'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        entities = match.group(1)
    else:
        entities = response.strip()
        if "```json" in entities:
            entities = entities.replace("```json", "")

    json_entities = []
    for entity in entities.strip("\n").split("\n"):
        entity = entity.strip("\n")
        if entity == "":
            continue
        json_entities.append(demjson.decode(entity))
    return json_entities


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def get_entities(data, few_shot_examples, shot=3, temp=0.5):
    sys_prompt = "As an expert in education and assessment, your task is to identify the most important entities and their related knowledge that must be understood in order to answer a given seed question. For each seed question, list up to 5 important entities and provide a brief description for each entity. I will first provide some reference examples. Please ensure that your responses follow a consistent format in line with the provided examples."
    
    response_format = """Response in the following format, each line include an entity:\n```json\n\{"name": <entity_name>, "description": <entity_description>\}\n"""
    
    samples = few_shot_examples[:shot]
    few_shot_prompts = [format_entity_example(sample) for sample in samples]
    few_shot_prompt ="Here is some reference examples:\n" + "\n\n".join(few_shot_prompts)
    
    question = build_example(data, with_answer=True, with_explain=False)
    question_prompt = f"Here is the seed question:\n{question}\nThe most important entities need to understand about this question include:"
    
    prompt = f"{sys_prompt}\n{response_format}\n\n{few_shot_prompt}\n\n{question_prompt}"
    
    
    for t in range(5):
        try:
            response = query_gpt(prompt, temp=0.7)
            json_entities = parse_jsonline_response(response)
            break
        except Exception as e:
            print(e)
            json_entities = {}
            continue

    return json_entities



def entity_extract_for_task(benchmark, task=None, split=None, shot=3, temp=0.5):
    dev_data_path = "processed_data/few_shot_example.json"
    dev_data = load_json_dic(dev_data_path)
    
    print(">>>> extracting concepts for {} {} {}".format(benchmark, task, split))
    
    if task is None:
        candidate_data = load_file("processed_data/{}/0_{}_with_idx.json".format(benchmark, split))
        save_data_path = "processed_data/{}/a_{}_with_entities.json".format(benchmark, split)
    else:
        candidate_data = load_file("processed_data/{}/{}/0_{}_with_idx.json".format(benchmark, task, split))
        save_data_path = "processed_data/{}/{}/a_{}_with_entities.json".format(benchmark, task, split)
    
    
    if DEBUG:
        candidate_data = candidate_data[:3]
    
    new_data = []
    if os.path.exists(save_data_path):
        new_data = load_json_dic(save_data_path)
        
    
    for i in tqdm(range(len(new_data), len(candidate_data))):
        data = candidate_data[i]
        
        try:
            entities = get_entities(data, dev_data, shot=shot, temp=temp)
        except Exception as e:
            print(e)
            entities = []
        
        data["entities"] = entities
        new_data.append(data)
        if len(new_data) % 5 == 0:
            save_json_dic(new_data, save_data_path)
    
    save_json_dic(new_data, save_data_path)
    
    

# 为每个concept检索wikipedia内容
def retrieve_wiki_for_concept(args, benchmark, task=None, split=None):
    print(">>>>>retrieve_wiki_for_concept on {} {} {}".format(benchmark, task, split))
    
    if task is None:
        candidate_data_path = "processed_data/{}/a_{}_with_entities.json".format(benchmark, split)
        save_data_path = "processed_data/{}/a_{}_with_entities_info.json".format(benchmark, split)
    else:
        candidate_data_path = "processed_data/{}/{}/a_{}_with_entities.json".format(benchmark, task, split)
        save_data_path = "processed_data/{}/{}/a_{}_with_entities_info.json".format(benchmark, task, split)
    
    candidate_data = load_json_dic(candidate_data_path)
        
    if args.rank_method == "bge":
        bge_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large").cuda()
        bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        bge_model.eval()
    else:
        bge_model = None
        bge_tokenizer = None
    
    new_data = []
    if os.path.exists(save_data_path):
        new_data = load_json_dic(save_data_path)
        print(len(new_data), len(candidate_data))
    
    for idx in tqdm(range(len(new_data), len(candidate_data))):
        data = candidate_data[idx]
        entities = data["entities"]
        seed_question = build_example(data, with_answer=True, with_explain=False)
        for i in range(len(entities)):
            name = entities[i]["name"]
            des = entities[i]["description"]
            
            for try_times in range(5):
                try:
                    wiki_info = wiki_retrieve_by_entity(args, name, des, seed_question, bge_model=bge_model, beg_tokenizer=bge_tokenizer, doc_num=args.para_num)
                    entities[i]["wiki_info"] = wiki_info
                    break
                except Exception as e:
                    print(e)
                    entities[i]["wiki_info"] = {}
        
        data["entities"] = entities
        new_data.append(data)
        
        
        if len(new_data) % 5 == 0:
            save_json_dic(new_data, save_data_path)
    
    save_json_dic(new_data, save_data_path)



def label_incorrect_entity(benchmark, task=None, split=None):
    print("working on label entity {} {} {}".format(benchmark, task, split))
    
    if task is None:
        original_data_file = "processed_data/{}/a_{}_with_entities_info.json".format(benchmark, split)
        original_dataset = load_json_dic(original_data_file)
        
        save_data_path = "processed_data/{}/b_{}_with_label_entity.json".format(benchmark, split)
    else:
        original_data_file = "processed_data/{}/{}/a_{}_with_entities_info.json".format(benchmark, task, split)
        original_dataset = load_json_dic(original_data_file)
        
        save_data_path = "processed_data/{}/{}/b_{}_with_label_entity.json".format(benchmark, task, split)
    
    new_data = []
    
    if os.path.exists(save_data_path):
        new_data = load_json_dic(save_data_path)
    
    for i in tqdm(range(len(new_data), len(original_dataset))):
        data = original_dataset[i]
        entities = data["entities"]
        topic_infos = []
        for entity in entities:
            try:
                topic_infos.append(
                    (entity["name"], entity["description"], 
                    entity["wiki_info"]["wiki_name"], entity["wiki_info"]["wiki_intro"])
                )
            except:
                topic_infos.append(
                    (entity["name"], entity["description"], 
                    "empty", "empty")
                )

        predictions = entity_match_multithreaded(topic_infos)
        assert len(predictions) == len(entities)
        for idx in range(len(entities)):
            entities[idx]["concept_match"] = predictions[idx]
        
        data["entities"] = entities
        new_data.append(data)
    
        
        if len(new_data) % 5 == 0:
            save_json_dic(new_data, save_data_path)
    
    save_json_dic(new_data, save_data_path)
    
    
def filter_match_entity(benchmark, task=None, split=None):
    if task is None:
        data_file = "processed_data/{}/b_{}_with_label_entity.json".format(benchmark, split)
        original_dataset = load_json_dic(data_file)
        save_data_path = "processed_data/{}/c_{}_with_matched_entity.json".format(benchmark, split)
    else:
        data_file = "processed_data/{}/{}/b_{}_with_label_entity.json".format(benchmark, task, split)
        original_dataset = load_json_dic(data_file)
        save_data_path = "processed_data/{}/{}/c_{}_with_matched_entity.json".format(benchmark, task, split)
    
    new_data = []
    
    if os.path.exists(save_data_path):
        new_data = load_json_dic(save_data_path)
    
    for i in tqdm(range(len(new_data), len(original_dataset))):
        data = original_dataset[i]
        entities = data["entities"]
        new_entities = []
        for entity in entities:
            if entity["concept_match"]:
                entity["wiki_info"]["wiki_page"] = ""
                new_entities.append(entity)
        data["entities"] = new_entities
        new_data.append(data)
    
    save_json_dic(new_data, save_data_path)


DEBUG=False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--split", type=str)
    parser.add_argument("--model-type", type=str, default="gpt-4o-mini")
    
    
    parser.add_argument("--rank-method", type=str, choices=["gpt", "bge"], required=False, default="bge")
    parser.add_argument("--para-num", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--use-openai-chunk", type=bool, default=True)
    
    
    args = parser.parse_args()
    
    
    if args.benchmark == "mmlu":
        for task in MMLU_TASKS:
            entity_extract_for_task(benchmark=args.benchmark, task=args.task, split=args.split)
            retrieve_wiki_for_concept(args, benchmark=args.benchmark, task=args.task, split=args.split)
            label_incorrect_entity(benchmark=args.benchmark, task=args.task, split=args.split)
            filter_match_entity(benchmark=args.benchmark, task=args.task, split=args.split)
    else:
        entity_extract_for_task(benchmark=args.benchmark, split=args.split)
        retrieve_wiki_for_concept(args, benchmark=args.benchmark, split=args.split)
        label_incorrect_entity(benchmark=args.benchmark, split=args.split)
        filter_match_entity(benchmark=args.benchmark, split=args.split)
