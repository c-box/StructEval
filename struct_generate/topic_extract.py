import argparse
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from common_utils.utils import load_file, load_json_dic, save_jsonl_data, save_json_dic, set_seed, build_example, CHOICES, get_text_chunks
from common_utils.prompt_utils import query_gpt, multi_query_gpt, entity_match_multithreaded
import os
from common_utils.constant import MMLU_TASKS
import re
import random
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from common_utils.wiki_search import search_for_wikipedia, search_with_wikipedia_api, trunc_wikipedia_page, rank_gpt, bge_rank, batch_bge_rank, wiki_retrieve_by_entity
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed


set_seed()


def format_topic_example(data):
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
    topic = data["topic"]["name"]
    topic_des = data["topic"]["description"]
    
    prompt = f"Question: {question}\nChoices:\n{choice}\nAnswer: {answer}\nResponse:\nTopic: {topic}\nnDescription: {topic_des}"
    return prompt



@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def get_topic(data, few_shot_examples, shot=3, temp=0.5):
    sys_prompt = "As an expert in education and assessment, your task is to accurately identify the core topic of the seed questions I present, and provide a brief description of that topic. I will first provide some reference examples. Please ensure that your responses follow a consistent format in line with the provided examples."
    
    response_format = "Response in the following format:\nTopic: <core topic of the question>\nDescription: <description of the topic>"
    
    samples = few_shot_examples[:shot]
    
    few_shot_prompts = [format_topic_example(sample) for sample in samples]
    few_shot_prompt ="Here is some reference examples:\n" + "\n\n".join(few_shot_prompts)
    
    question = build_example(data,with_answer=True, with_explain=False)
    question_prompt = f"Here is the seed question:\n{question}\nResponse:"
    
    prompt = f"{sys_prompt}\n{response_format}\n\n{few_shot_prompt}\n\n{question_prompt}"
    
    response = query_gpt(prompt)
    
    assert "Topic: " in response and "Description: " in response
    
    lines = response.split('\n')

    # Extracting topic and description
    topic = lines[0].split('Topic: ', 1)[1] if 'Topic: ' in lines[0] else ''
    description = lines[1].split('Description: ', 1)[1] if 'Description: ' in lines[1] else ''

    # Creating a dictionary with the extracted data
    return {"name": topic, "description": description}


# 存储topic抽取的中间结果
def topic_extract_for_task(benchmark, task=None, split=None, shot=3, temp=0.5):
    print(">>>>>working on  topic extract, benchmark: {},  task: {}, split: {}".format(benchmark, task, split))
    
    dev_data_path = "processed_data/few_shot_example.json"
    
    if task is None:
        candidate_data_path = "processed_data/{}/0_{}_with_idx.json".format(benchmark, split)
        save_data_path = "processed_data/{}/1_{}_with_topic.json".format(benchmark, split)
    else:
        candidate_data_path = "processed_data/{}/{}/0_{}_with_idx.json".format(benchmark, task, split)
        save_data_path = "processed_data/{}/{}/1_{}_with_topic.json".format(benchmark, task, split)
    
    
    candidate_data = load_file(candidate_data_path)
    dev_data = load_json_dic(dev_data_path)
    
    if DEBUG:
        candidate_data = candidate_data[:2]
    
    new_data = []
    for data in tqdm(candidate_data):
        topic = get_topic(data, dev_data, shot=shot, temp=temp)
        data["topic"] = topic
        new_data.append(data)
        if len(new_data) % 5 == 0:
            save_json_dic(new_data, save_data_path)
    save_json_dic(new_data, save_data_path)
    
    

# 检索topic对应的内容
def topic_based_wiki_page(args, benchmark, task=None, split=None):
    print(">>>>>working on wiki page retrieve, benchmark: {}， task: {}， split: {}".format(benchmark, task, split))
    
    if task is None:
        candidate_data_path = "processed_data/{}/1_{}_with_topic.json".format(benchmark, split)
        candidate_data = load_json_dic(candidate_data_path)
        
        save_data_path = "processed_data/{}/1_{}_with_topic_info.json".format(benchmark, split)
    else:
        candidate_data_path = "processed_data/{}/{}/1_{}_with_topic.json".format(benchmark, task, split)
        candidate_data = load_json_dic(candidate_data_path)
        
        save_data_path = "processed_data/{}/{}/1_{}_with_topic_info.json".format(benchmark, task, split)
        

    saved_data = []
    
    if os.path.exists(save_data_path):
        saved_data = load_json_dic(save_data_path)
    
    
    if args.rank_method == "bge":
        bge_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large").cuda()
        bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        bge_model.eval()
    else:
        bge_model = None
        bge_tokenizer = None
    
    
    for idx in tqdm(range(len(saved_data), len(candidate_data))):
        data = candidate_data[idx]
        question = data["question"]
        
        topic_name = data["topic"]["name"]
        topic_des = data["topic"]["description"]
        
        seed_question = build_example(data, with_answer=True, with_explain=False)
        # print(seed_question)
        for try_time in range(5):
            try:
                topic_wiki_info = wiki_retrieve_by_entity(args, topic_name, topic_des, seed_question, bge_model=bge_model, beg_tokenizer=bge_tokenizer, doc_num=args.para_num)
                data["topic_wiki_info"] = topic_wiki_info
                break
            except Exception as e:
                print(e)
                data["topic_wiki_info"] = {}
        saved_data.append(data)
        print("\n-------------------]\n")
        if len(saved_data) % 5 == 0:
            save_json_dic(saved_data, save_data_path)
    
    save_json_dic(saved_data, save_data_path)






def label_incorrect_topics(args, task=None, split=None, batch_size=8):
    print("working on label_incorrect_topics. benchmark:{}, task:{}, split:{}".format(args.benchmark, task, split))
    
    if task is None:
        original_data_file = "processed_data/{}/1_{}_with_topic_info.json".format(args.benchmark, split)
        original_dataset = load_json_dic(original_data_file)
        save_data_path = "processed_data/{}/2_{}_with_label_topic.json".format(args.benchmark, split)
    else:
        original_data_file = "processed_data/{}/{}/1_{}_with_topic_info.json".format(args.benchmark, task, split)
        original_dataset = load_json_dic(original_data_file)
        save_data_path = "processed_data/{}/{}/2_{}_with_label_topic.json".format(args.benchmark, task, split)
        
    
    topic_infos = []
    saved_idx = []
    
    if os.path.exists(save_data_path):
        saved_data = load_json_dic(save_data_path)
        if len(saved_data) >= len(original_dataset):
            print("jump {}".format(task))
            return
    
    
    for i in tqdm(range(len(original_dataset))):
        
        data = original_dataset[i]
        original_name = data["topic"]["name"]
        original_des = data["topic"]["description"]
        
        saved_idx.append(i)
        
        try:
            if "wiki_name" not in data["topic_wiki_info"]:
                topic_infos.append((
                original_name, original_des, "None", "None"
            ))
            else:
                searched_name = data["topic_wiki_info"]["wiki_name"]
                serached_des = data["topic_wiki_info"]["wiki_intro"]
                
                topic_infos.append((
                    original_name, original_des, searched_name, serached_des
                ))
        except Exception as e:
            print(e)
            exit
    
    all_ans = []
    
    for i in tqdm(range(0, len(topic_infos), batch_size)):
        batch_info = [topic_info for topic_info in topic_infos[i:i+batch_size]]
        ans = entity_match_multithreaded(batch_info)
        all_ans.extend(ans)
        
        temp_data = []
        for idx, ans in zip(saved_idx, all_ans):
            temp_data.append({
                "idx": idx,
                "ans": ans
            })
        
    assert(len(all_ans) == len(saved_idx))
    
    for idx, ans in zip(saved_idx, all_ans):
        original_dataset[idx]["topic_match"] = ans
    
    for i in range(len(original_dataset)):
        if "topic_match" not in original_dataset[i]:
            original_dataset[i]["topic_match"] = False
    
    save_json_dic(original_dataset, save_data_path)


def filter_match_samples(benchmark, task=None, split=None):
    
    if task is None:
        original_data_file = "processed_data/{}/2_{}_with_label_topic.json".format(benchmark, split)
        original_dataset = load_json_dic(original_data_file)
        save_data_path = "processed_data/{}/3_{}_with_matched_topic.json".format(benchmark, split)
    else:
        original_data_file = "processed_data/{}/{}/2_{}_with_label_topic.json".format(benchmark, task, split)
        original_dataset = load_json_dic(original_data_file)
        save_data_path = "processed_data/{}/{}/3_{}_with_matched_topic.json".format(benchmark, task, split)
    
    filter_data = []
    for data in original_dataset:
        if data["topic_match"] is True:
            data["topic_wiki_info"]["wiki_page"] = ""
            filter_data.append(data)
    save_json_dic(filter_data, save_data_path)
        
DEBUG=False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    
    
    parser.add_argument("--rank-method", type=str, choices=["gpt", "bge"], required=False, default="bge")
    parser.add_argument("--para-num", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--use-openai-chunk", type=bool, default=True)
    
    args = parser.parse_args()
    
    
    if args.benchmark == "mmlu":
        for task in tqdm(MMLU_TASKS):
            topic_extract_for_task(benchmark=args.benchmark, task=task, split=args.split)
            topic_based_wiki_page(args, benchmark=args.benchmark, task=task, split=args.split)
            label_incorrect_topics(args, task=task, split=args.split)
            filter_match_samples(benchmark=args.benchmark, task=task, split=args.split)
    else:
        topic_extract_for_task(benchmark=args.benchmark, split=args.split)
        topic_based_wiki_page(args, benchmark=args.benchmark, split=args.split)
        label_incorrect_topics(args, split=args.split)
        filter_match_samples(benchmark=args.benchmark, split=args.split)
    