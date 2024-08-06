import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import demjson
import os
import torch
import threading
import re
import queue
from os import getenv
from common_utils.constant import client





@retry(stop=stop_after_attempt(30), wait=wait_exponential(multiplier=1, max=10))
def chatgpt_completion(messages, temp=0.7, model_type="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model_type,
        messages=messages,
        temperature=temp
    )
    return completion.choices[0].message.content


def query_gpt(query, sys=None, temp=0.7, model_type="gpt-4o-mini"):
    if sys is None:
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-10\nCurrent date: 2024-07-24"
    else:
        system_prompt = sys
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": query}
    ]
    
    try:
        response = chatgpt_completion(messages, temp=temp, model_type=model_type)
    except Exception as e:
        print(e)
        return "no response"
    return response


def multi_query_gpt(queries, temp=0.7, model_type="gpt-4o-mini"):
    def worker(query, index, result_dict):
        result_dict[index] = query_gpt(query, temp=temp,  model_type=model_type)

    threads = []
    results = {}
    for index, query in enumerate(queries):
        thread = threading.Thread(target=worker, args=(query, index, results))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    ordered_responses = [results[index] for index in range(len(queries))]
    return ordered_responses


def parse_json_response(response):
    json_pattern = r'```json\n(.*?)\n```'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        entities = match.group(1)
    else:
        entities = response.strip()
        if "```json" in entities:
            entities = entities.replace("```json", "")
    
    json_entities = demjson.decode(entities.strip())
    return json_entities


def try_to_query(prompt, max_time=5, model_type="gpt-3.5-turbo-1106"):
    for t in range(max_time):
        try:
            response = query_gpt(prompt, temp=0.7, model_type=model_type)
            # print(response)
            json_question = parse_json_response(response)
            # print(json_question)
            return json_question
        except Exception as e:
            print("\n\n")
            print(response)
            print("\n\n")
            print(e)
            continue
    return {}



def process_entity_match(entity_info, result_queue, index):
    original_name, original_des, searched_name, searched_des = entity_info
    prompt = f"Determine whether the following two mentions with description might refer to the same or relevant entity. Only response with yes or no without any other content.\n\nFirst Mention: {original_name}: {original_des}\n\nSecond Mention: {searched_name}: {searched_des}\n\nDecision: "
    # print(prompt)
    response = query_gpt(prompt, temp=0)  
    response = response.lower()
    # print(response)
    is_match = "yes" in response
    result_queue.put((index, is_match))


def entity_match_multithreaded(entity_infos):
    result_queue = queue.Queue()
    threads = []

    for index, entity_info in enumerate(entity_infos):
        thread = threading.Thread(target=process_entity_match, args=(entity_info, result_queue, index))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results = [None] * len(entity_infos)
    while not result_queue.empty():
        index, result = result_queue.get()
        results[index] = result

    return results



def remove_questions_document(question):
    if "document" in question:
        pattern_list = [
            ", as mentioned in the document, ",
            "According to the document,",
            "According to the provided document,",
            "According to the provided document,",
            " , according to the document",
            " , as per the document",
            " , as suggest in the document",
            " , as per the provided document",
            " , as per the document",
            " as per the provided document",
            " as per the document",
            " according to the given document",
            " , as stated in the provided document",
            " , according to the provided document",
            " according to the provided document",
            " according to the document",
            " mentioned in the provided document",
            ", as described in the provided document",
            ", as described in the document",
            " as mentioned in the document",
            " mentioned in the document",
            "In the provided document, ",
            " , as mentioned in the provided document",
            " , as highlighted in the provided document",
            " as described in the provided document",
            " , as described in the provided document",
            " within the document",
            " , as discussed in the document",
            " , as indicated in the provided document",
            "Based on the document,",
            "Based on the provided document,",
            "As per the provided document, ",
            " , as suggested by the document",
            " in the provided document",
            " based on the document",
            " based on the provided document",
            "In the context of the document, ",
            "In the document, ",
            " in the document",
        ]
        
        
        for pattern in pattern_list:
            if pattern in question:
                question = question.replace(pattern, "").strip()
                question = question.capitalize()
                question = question.replace(", ?", "?")
                question = question.replace(",?", "?")
                question = question.replace(" ?", "?")
                # refine_question = question
                break
        
        
        if " document " in question:
            print("originla: {}".format(question))
            prompt = "Remove expressions about a specific document such as \"according to the provided documents\", \"based on the document\", \"in the document\", 'as mentioned in the document', \"How does the document describe...\" from the following question, return the question text only. Ensure that the style and semantics of the question remain unchanged. Ensure the output question do not require any external document to answer.\n\n{}".format(question)
            response = query_gpt(query=prompt, temp=0.7)
            question = response
            
    return question


        
