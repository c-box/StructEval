import argparse
from tqdm import tqdm
import openai
import requests
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from common_utils.utils import load_file, load_json_dic, save_jsonl_data, save_json_dic, set_seed, build_example, CHOICES, get_text_chunks, parse_json_response, find_answer, random_select_choice
from common_utils.prompt_utils import query_gpt, multi_query_gpt, remove_questions_document
import os
from common_utils.constant import MMLU_TASKS
import re
import random
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from common_utils.wiki_search import search_for_wikipedia, search_with_wikipedia_api, trunc_wikipedia_page, rank_gpt, bge_rank, batch_bge_rank, wiki_retrieve_by_entity
import threading
import queue
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed
from common_utils.prompts import *


set_seed()


def try_to_query(prompt, max_time=5, model_type="gpt-4o-mini"):
    for t in range(max_time):
        try:
            response = query_gpt(prompt, temp=0.7, model_type=model_type)
            # print(response)
            json_question = parse_json_response(response)
            # print(json_question)
            return json_question
        except Exception as e:
            # print(response)
            print(e)
            continue
    return {}


import threading
def generate_question(level, topic, document, model_type, bloom_expand_questions, lock):
    INS_PROMPT = {
        "remember": MUL_REMEMBER_INS,
        "understand": MUL_UNDERSTAND_INS,
        "apply": MUL_APPLY_INS,
        "analyze": MUL_ANALYZ_INS,
        "evaluation": MUL_EVAL_INS,
        "create": MUL_CREATE_INS
    }
    
    instruction_prompt = INS_PROMPT[level]
    prompt = instruction_prompt.format(topic=topic, document=document)
    response = try_to_query(prompt, model_type=model_type)

    with lock:
        bloom_expand_questions[level] = response


def generate_multi_questions_multi_threaded(topic, document, model_type="gpt-4o-mini"):
    INS_PROMPT = {
        "remember": MUL_REMEMBER_INS,
        "understand": MUL_UNDERSTAND_INS,
        "apply": MUL_APPLY_INS,
        "analyze": MUL_ANALYZ_INS,
        "evaluation": MUL_EVAL_INS,
        "create": MUL_CREATE_INS
    }

    bloom_expand_questions = {}
    threads = []
    lock = threading.Lock()

    for level in INS_PROMPT:
        thread = threading.Thread(target=generate_question, args=(level, topic, document, model_type, bloom_expand_questions, lock))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return bloom_expand_questions

# 问题生成
def bloom_question_generation(benchmark, task=None, split=None, model_type=None):
    print(">>>>>working on bloom generation {} {} {}".format(benchmark, task, split))
    
    if not task:
        original_data_path = "processed_data/{}/3_{}_with_matched_topic.json".format(benchmark, split)
        wiki_page_data = load_json_dic(original_data_path)
        saved_data_path = "processed_data/{}/4_{}_with_bloom_question_{}.json".format(benchmark, split, model_type)
    else:
        original_data_path = "processed_data/{}/{}/3_{}_with_matched_topic.json".format(benchmark, task, split)
        wiki_page_data = load_json_dic(original_data_path)
        saved_data_path = "processed_data/{}/{}/4_{}_with_bloom_question_{}.json".format(benchmark, task, split, model_type)
    
    new_data = []
    
    if DEBUG:
        wiki_page_data = wiki_page_data[:2]
    
    if os.path.exists(saved_data_path):
        new_data = load_json_dic(saved_data_path)
    
    for i in tqdm(range(len(new_data), len(wiki_page_data))):
        data = wiki_page_data[i]
        data["gpt_model"] = model_type
        
        if len(data["topic_wiki_info"]) == 0:
            data["bloom_questions"] = []
            new_data.append(data)
        else:
            topic_name = data["topic_wiki_info"]["wiki_name"]
            wiki_content = data["topic_wiki_info"]["related_content"]
            bloom_questions = generate_multi_questions_multi_threaded(topic_name, wiki_content, model_type=model_type)
            bloom_questions = remove_bloom_questions_document(bloom_questions)
            data["bloom_questions"] = bloom_questions
            new_data.append(data)
    
    
        if len(new_data) % 3 == 0:
            save_json_dic(new_data, saved_data_path)
            print("saved {} cased".format(len(new_data)))
    
    save_json_dic(new_data, saved_data_path)



# 改写需要依赖context的question
def remove_bloom_questions_document(original_questions):
    for level in original_questions:
        for idx in range(len(original_questions[level])):
            if "question" not in original_questions[level][idx]:
                continue
            question = original_questions[level][idx]["question"]
            original_questions[level][idx]["question"] = remove_questions_document(question)
    
    return original_questions


def multiple_RAG_prediction(data):
    print(data["question"])
    try:
        bloom_questions = data["bloom_questions"]
        subject = data["subject"]
        related_content = data["topic_wiki_info"]["related_content"]
    except Exception as e:
        return data
    

    for key in bloom_questions:
        level_questions = bloom_questions[key]
        num_of_questions = len(level_questions)
        prompt = f"### Instruction:\nRefer to the document, select the correct answer for each of the following {num_of_questions} multiple choice questions about {subject}.\nFor each question, if you can find the correct answer in the document, response with the correct choice such as ‘A/B/C/D’. \nIf you cannot find the correct answer in the document, response with 'I cannot answer'\nIf the choices contain more than one correct option, Response with 'I cannot answer' \nThe reply must consist of exactly {num_of_questions} lines, with each line corresponding in order to the answer of each question. Ensure the answer begin with the correct choice and do not output any other content.\n\n"
        
        document = "### Document:\n{}".format(related_content)
        
        prompt = prompt + document
        
        for idx, example in enumerate(level_questions):
            level_questions[idx] = random_select_choice(level_questions[idx])
            question = "### Question {}:\n{}".format(idx+1, build_example(example, with_answer=False, expand_eval=True))
            prompt = prompt + "\n\n" + question
        
        # print(prompt)
        
        responses = query_gpt(prompt, temp=0).split("\n")
        
        print(responses)
        
        assert(len(responses) == num_of_questions)
        
        for idx, response in enumerate(responses):
            correct_answer = level_questions[idx]["answer"]
            pred = find_answer(response)
            if correct_answer.upper() == pred[0].upper():
                ans = True
            else:
                ans = False
            
            level_questions[idx]["RAG"] = {
                "response": response,
                "ans": ans
            }
        
        data["bloom_questions"][key] = level_questions
    return data
    


def RAG_prediction(data):
    try:
        bloom_questions = data["bloom_questions"]
        subject = data["subject"]
        related_content = data["topic_wiki_info"]["related_content"]
    except Exception as e:
        return data
        
    for key in bloom_questions:
        level_questions = bloom_questions[key]
        queries = []
        for idx, example in enumerate(level_questions):
            level_questions[idx] = random_select_choice(level_questions[idx])
            
            prompt = f"### Instruction:\nRefer to the document, select the correct answer for the multiple choice questions about {subject}.\nIf you can find the correct answer in the document, response with the correct choice such as ‘A/B/C/D’. \nIf you cannot find the correct answer in the document, response with 'I cannot answer'\nIf the choices contain more than one correct option, Response with 'I cannot answer' \nEnsure your response begin with the correct choice and do not output any other content.\n\n"
            
            document = "### Document:\n{}".format(related_content)
            
            question = "### Input:\n{}".format(build_example(example, with_answer=False, expand_eval=True))
            
            queries.append("{}\n\n{}\n\n{}".format(prompt, document, question))
        
        responses = multi_query_gpt(queries, temp=0)
        assert len(responses) == len(queries)
        
        for idx in range(len(level_questions)):
            correct_answer = level_questions[idx]["answer"]
            response = responses[idx]
            pred = find_answer(response)
            
            if correct_answer.upper() == pred[0].upper():
                ans = True
            else:
                ans = False
            
            level_questions[idx]["RAG"] = {
                "response": response,
                "ans": ans
            }
        data["bloom_questions"][key] = level_questions
    return data


# RAG模块判断正误
def RAG_test(benchmark, task=None, split="test", model_type=None):
    print("working on RAG test, {} {} {}".format(benchmark, task, split))
    
    if task is None:
        original_dataset = load_json_dic("processed_data/{}/4_{}_with_bloom_question_{}.json".format(benchmark, split, model_type))
        save_data_path = "processed_data/{}/5_{}_with_bloom_question_{}_rag.json".format(benchmark, split, model_type)
    else:
        original_dataset = load_json_dic("processed_data/{}/{}/4_{}_with_bloom_question_{}.json".format(benchmark, task, split, model_type))
        save_data_path = "processed_data/{}/{}/5_{}_with_bloom_question_{}_rag.json".format(benchmark, task, split, model_type)
    
    if DEBUG:
        original_dataset = original_dataset[:1]
    
    new_dataset = []
    
    if os.path.exists(save_data_path):
        new_dataset = load_json_dic(save_data_path)
        if len(new_dataset) == len(original_dataset):
            print(">>>>>jump")
            return
    
    for i in tqdm(range(len(new_dataset), len(original_dataset))):
        
        # data = RAG_prediction(original_dataset[i])
        data = multiple_RAG_prediction(original_dataset[i])
        
        new_dataset.append(data)
        if i % 5 == 0:
            # print(right, total, round(right/total, 2))
            save_json_dic(new_dataset, save_data_path)
    
    # print(right, total, round(right/total, 2))
    save_json_dic(new_dataset, save_data_path)
    


DEBUG = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model-type", type=str, default="gpt-4o-mini")
    
    args = parser.parse_args()
    
    if args.benchmark == "mmlu":
        for task in MMLU_TASKS:
            bloom_question_generation(benchmark=args.benchmark, task=task, split=args.split, model_type=args.model_type)
            RAG_test(benchmark=args.benchmark, task=task, split=args.split, model_type=args.model_type)
    else:
        bloom_question_generation(benchmark=args.benchmark, split=args.split, model_type=args.model_type)
        RAG_test(benchmark=args.benchmark, split=args.split, model_type=args.model_type)

