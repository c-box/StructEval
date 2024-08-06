import csv
import json
import torch
import random
import numpy as np
from prettytable import PrettyTable
import tiktoken
import random
import json
from hashlib import md5


def set_seed(seed_num=1023):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    

def get_csv_dic(filename):
    with open(filename, encoding="utf-8-sig", mode="r") as f:
        reader = csv.DictReader(f)
        data_dic = []
        for row in reader:
            data_dic.append(row)
    return data_dic


def write_csv(filename, data):
    fields_names = [key for key in data[0]]
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields_names)
        writer.writeheader()
        writer.writerows(data)
        

def save_jsonl_data(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            
            
def load_file(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line == "":
                continue
            data.append(json.loads(line))
        f.close()
    return data


def save_json_dic(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json_dic(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.loads(f.read())
    

def r2(x):
    return round(x*100, 2)



CHOICES = ["A", "B", "C", "D"]


def build_example(data, with_answer=True, with_explain=False, expand_eval=False):
        if not expand_eval:
            question = data["question"]
            choice = "\n".join(
                [
                    "A. " + data["choices"][0],
                    "B. " + data["choices"][1],
                    "C. " + data["choices"][2],
                    "D. " + data["choices"][3],
                ]
            )
            answer = CHOICES[data["answer"]] if with_answer else ""
            
            if with_answer:
                answer_span = data["choices"][data["answer"]]
                answer = f"{answer}. {answer_span}"
            
            if "explanation" in data:
                ex = data["explanation"]
            else:
                ex = ""
            if with_explain:
                return f"Question: {question}\n{choice}\nAnswer: {answer}\nExplanation:{ex}"
            else:
                return f"Question: {question}\n{choice}\nAnswer: {answer}" 
        else:
            question = data["question"]
            choice = "\n".join(
                [
                    "A. " + data["A"],
                    "B. " + data["B"],
                    "C. " + data["C"],
                    "D. " + data["D"],
                ]
            )
            answer = data["answer"] if with_answer else ""
            if with_answer:
                answer_span = data[data[answer]]
                answer = f"{answer}. {answer_span}"
            
            
            if "explanation" in data:
                ex = data["explanation"]
            else:
                ex = ""
            if with_explain:
                return f"Question: {question}\n{choice}\nAnswer: {answer}\nExplanation:{ex}"
            else:
                return f"Question: {question}\n{choice}\nAnswer: {answer}" 



import os
os.environ["TIKTOKEN_CACHE_DIR"] = "cache/tiktoken"

chunk_tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)

def get_text_chunks(text: str, chunk_size: int, tokenizer=chunk_tokenizer):

    '''
    modified from https://github.com/openai/chatgpt-retrieval-plugin/blob/main/services/chunks.py
    '''
    if not text or text.isspace():
        return []

    # Tokenize the text
    tokens = tokenizer.encode(text)

    # Initialize an empty list of chunks
    chunks = []

    # Initialize a counter for the number of chunks
    num_chunks = 0

    # Loop until all tokens are consumed
    while tokens:
        # Take the first chunk_size tokens as a chunk
        chunk = tokens[:chunk_size]

        # Decode the chunk into text
        chunk_text = tokenizer.decode(chunk)

        # Skip the chunk if it is empty or whitespace
        if not chunk_text or chunk_text.isspace():
            # Remove the tokens corresponding to the chunk text from the remaining tokens
            tokens = tokens[len(chunk) :]
            # Continue to the next iteration of the loop
            continue

        # Find the last period or punctuation mark in the chunk add Chinese prunctuation
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("。"),
            chunk_text.rfind("？"),
            chunk_text.rfind("！"),
            chunk_text.rfind("\n"),
        )

        # If there is a punctuation mark, and the last punctuation index is before MIN_CHUNK_SIZE_CHARS
        if last_punctuation != -1:
            # Truncate the chunk text at the punctuation mark
            chunk_text = chunk_text[: last_punctuation + 1]

        # Remove any newline characters and strip any leading or trailing whitespace
        chunk_text_to_append = chunk_text.replace("\n", " ").strip()

        if len(chunk_text_to_append) > 5:
            chunks.append(chunk_text_to_append)
        # Remove the tokens corresponding to the chunk text from the remaining tokens
        tokens = tokens[len(tokenizer.encode(chunk_text)) :]

        # Increment the number of chunks
        num_chunks += 1

    # Handle the remaining tokens
    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > 5:
            chunks.append(remaining_text)

    return chunks


import re
import demjson
import json
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
    # for entity in entities.strip("\n").split("\n"):
    #     entity = entity.strip("\n")
    #     if entity == "":
    #         continue
    #     json_entities.append(demjson.decode(entity))
    return json_entities


def find_answer(text, patterns=[r"answer is:(.*)"]):
    patterns=[
        r"answer is:(.*)",
        r"answer for the multiple choice question:(.*)",
        r"Answer:(.*)",
        r"answer is (.*)",
        r"answer:(.*)"
    ]
    
    for pattern in patterns:
        text = text.replace("\n", " ")
        match = re.search(pattern, text)
        if match:
            result = match.group(1).strip()
            return result
    
    return text


# count tokens with openai API
def token_count(input_text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(input_text)
    return len(tokens)


def random_select_choice(sample):
    CHOICES = ["A", "B", "C", "D"]
    import random
    
    select_ans = random.choice(CHOICES)
    
    original_ans = sample["answer"]
    
    sample[select_ans], sample[original_ans] = sample[original_ans], sample[select_ans]
    
    sample["answer"] = select_ans
    
    return sample


if __name__ == "__main__":
    input_text = "I like beijing as you are"
    print(token_count(input_text))
