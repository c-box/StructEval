import argparse
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from common_utils.utils import load_file, load_json_dic, save_jsonl_data, save_json_dic, set_seed, build_example, CHOICES, get_text_chunks, parse_json_response, find_answer, random_select_choice
from common_utils.prompt_utils import query_gpt, multi_query_gpt, remove_questions_document
import os
from common_utils.constant import MMLU_TASKS
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


def concept_query_expand(data, concept, document, model_type):
    question = build_example(data, with_answer=True, with_explain=False, expand_eval=False)
    instruction_prompt = CONCEPT_GENRATION_PROMPT.format(concept=concept, question=question, document=document)
    # print(instruction_prompt)
    response = try_to_query(instruction_prompt, model_type=model_type)
    return response



def concept_question_generation(benchmark, task=None, split=None, model_type=None):
    print(">>>>>working on concept_question_generation, {} {} {}".format(benchmark, task, split))
    
    if task is None:
        original_data_path = "processed_data/{}/c_{}_with_matched_entity.json".format(benchmark, split)
        
        concept_data = load_json_dic(original_data_path)
        
        saved_data_path = "processed_data/{}/d_{}_with_concept_question_{}.json".format(benchmark, split, model_type)
    else:
        original_data_path = "processed_data/{}/{}/c_{}_with_matched_entity.json".format(benchmark, task, split)
        
        concept_data = load_json_dic(original_data_path)
        
        saved_data_path = "processed_data/{}/{}/d_{}_with_concept_question_{}.json".format(benchmark, task, split, model_type)
        
    
    if DEBUG:
        concept_data = concept_data[:2]
    
    new_data = []
    
    if os.path.exists(saved_data_path):
        new_data = load_json_dic(saved_data_path)
    
    for i in tqdm(range(len(new_data), len(concept_data))):
        data = concept_data[i]
        data["gpt_model"] = model_type
        entities = data["entities"]
        for entity_id, entity in enumerate(entities):
            try:
                entity_name = entity["name"]
                document = entity["wiki_info"]["related_content"]
                concept_questions = concept_query_expand(data=data, concept=entity_name, document=document, model_type=args.model_type)
                for idx in range(len(concept_questions)):
                    concept_questions[idx]["question"] = remove_questions_document(concept_questions[idx]["question"])
                entities[entity_id]["concept_questions"] = concept_questions
            except Exception as e:
                print(e)
                continue
        
        new_data.append(data)
           
        if len(new_data) % 3 == 0:
            save_json_dic(new_data, saved_data_path)
            print("saved {} cased".format(len(new_data)))
    
    save_json_dic(new_data, saved_data_path)
# def post_process_data()


def mutilple_RAG_prediction(entities, subject):
    for entity_idx, entity in enumerate(entities):
        concept_questions = entity["concept_questions"]
        related_content = entity["wiki_info"]["related_content"]
        
        num_of_questions = len(concept_questions)
        
        prompt = f"### Instruction:\nRefer to the document, select the correct answer for each of the following {num_of_questions} multiple choice questions about {subject}.\nFor each question, if you can find the correct answer in the document, response with the correct choice such as ‘A/B/C/D’. \nIf you cannot find the correct answer in the document, response with 'I cannot answer'\nIf the choices contain more than one correct option, Response with 'I cannot answer' \nThe reply must consist of exactly {num_of_questions} lines, with each line corresponding in order to the answer of each question. Ensure the answer begin with the correct choice and do not output any other content.\n\n"
        
        document = "### Document:\n{}".format(related_content)
        prompt = prompt + document
        
        for idx, example in enumerate(concept_questions):
            question = "### Question {}:\n{}".format(idx+1, build_example(example, with_answer=False, expand_eval=True))
            prompt = prompt + "\n\n" + question
        
        responses = query_gpt(prompt, temp=0).split("\n")
        # print(prompt)
        # print(responses)
        
        assert(len(responses) == num_of_questions)
        
        for idx, response in enumerate(responses):
            correct_answer = concept_questions[idx]["answer"]
            pred = find_answer(response)
            if correct_answer.upper() == pred[0].upper():
                ans = True
            else:
                ans = False
            
            concept_questions[idx]["RAG"] = {
                "response": response,
                "ans": ans
            }
        
        entities[entity_idx]["concept_questions"] = concept_questions
    return entities



def RAG_prediction(entities, subject):
    for entity_idx, entity in enumerate(entities):
        concept_questions = entity["concept_questions"]
        related_content = entity["wiki_info"]["related_content"]
        queries = []
        for example in concept_questions:
            prompt = f"### Instruction:\nRefer to the document, select the correct answer for the multiple choice questions about {subject}.\nIf you can find the correct answer in the document, response with the correct choice such as ‘A/B/C/D’. \nIf you cannot find the correct answer in the document, response with 'I cannot answer'\nIf the choices contain more than one correct option, response with 'I cannot answer' \nEnsure your response begin with the correct choice and do not output any other content.\n\n"
            
            document = "### Document:\n{}".format(related_content)
            
            question = "### Input:\n{}".format(build_example(example, with_answer=False, expand_eval=True))
            
            queries.append("{}\n\n{}\n\n{}".format(prompt, document, question))
        
        responses = multi_query_gpt(queries, temp=0)
        # print(responses)
        assert len(responses) == len(queries)
        
        for idx in range(len(concept_questions)):
            correct_answer = concept_questions[idx]["answer"]
            response = responses[idx]
            pred = find_answer(response)
            
            if correct_answer.upper() == pred[0].upper():
                ans = True
            else:
                ans = False
            
            concept_questions[idx]["RAG"] = {
                "response": response,
                "ans": ans
            }
        
        entities[entity_idx]["concept_questions"] = concept_questions
    return entities

    


def RAG_test(benchmark, task=None, split="test", model_type="gpt-4o-mini"):
    if task is None:
        original_dataset = load_json_dic("processed_data/{}/d_{}_with_concept_question_{}.json".format(benchmark, split, model_type))
        save_data_path = "processed_data/{}/e_{}_with_concept_question_{}_rag.json".format(benchmark, split, model_type)
    else:
        original_dataset = load_json_dic("processed_data/{}/{}/d_{}_with_concept_question_{}.json".format(benchmark, task, split, model_type))
        save_data_path = "processed_data/{}/{}/e_{}_with_concept_question_{}_rag.json".format(benchmark, task, split, model_type)
    
    
    if DEBUG:
        original_dataset = original_dataset[:2]
    
    new_dataset = []
    
    if os.path.exists(save_data_path):
        new_dataset = load_json_dic(save_data_path)
        if len(new_dataset) == len(original_dataset):
            print(">>>>>jump")
            return
    
    for i in tqdm(range(len(new_dataset), len(original_dataset))):
        
        for entity_id, entity in enumerate(original_dataset[i]["entities"]):
            concept_questions = entity["concept_questions"]
            for idx in range(len(concept_questions)):
                original_dataset[i]["entities"][entity_id]["concept_questions"][idx] \
                    = random_select_choice(concept_questions[idx])
            
        
        entities = original_dataset[i]["entities"]
        original_dataset[i]["entities"] = mutilple_RAG_prediction(entities, subject=task)
        
        new_dataset.append(original_dataset[i])
        if i % 3 == 0:
            save_json_dic(new_dataset, save_data_path)
    
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
            concept_question_generation(benchmark=args.benchmark, task=task, split=args.split, model_type=args.model_type)
            RAG_test(benchmark=args.benchmark, task=task, split=args.split, model_type=args.model_type)
    else:
        concept_question_generation(benchmark=args.benchmark, split=args.split, model_type=args.model_type)
        RAG_test(benchmark=args.benchmark, split=args.split, model_type=args.model_type)