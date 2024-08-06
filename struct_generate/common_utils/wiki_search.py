from tenacity import retry, stop_after_attempt, wait_exponential
from .es_client import ESClient
import torch
from .prompt_utils import query_gpt
from .utils import get_text_chunks
from tqdm import tqdm


import requests


def get_wikidata_id_by_wikipedia(wikipedia_title):
    # URL for the Wikipedia API
    wikipedia_api_url = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "prop": "pageprops",
        "format": "json",
        "titles": wikipedia_title,
        "srlimit": 1
    }

    # Make the request to the Wikipedia API
    response = requests.get(wikipedia_api_url, params=params)
    data = response.json()

    # Extracting the Wikidata ID from the response
    pages = data.get("query", {}).get("pages", {})
    for page_id in pages:
        wikidata_id = pages[page_id].get("pageprops", {}).get("wikibase_item")
        return wikidata_id

    return None


def search_wikidata_id_and_title(entity_name):
    # URL for the Wikidata search API
    wikidata_search_url = "https://www.wikidata.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "limit": 1  # Limit the search to the first result
    }

    # Make the request to the Wikidata API
    response = requests.get(wikidata_search_url, params=params)
    data = response.json()
    
    print(data)

    # Extracting the Wikidata ID and title from the response
    search_results = data.get("search", [])
    if search_results:
        wikidata_id = search_results[0].get("id")
        title = search_results[0].get("label")
        des = search_results[0]["display"]["description"]["value"]
        return wikidata_id, title, des
    else:
        return None, None, None

# Example usage
# wikipedia_title = "Albert Einstein"
# wikidata_id = get_wikidata_id(wikipedia_title)
# print(f"Wikidata ID for '{wikipedia_title}': {wikidata_id}")


def search_wikipedia_id(title, description):
    # URL for the Wikipedia search API
    wikipedia_search_url = "https://en.wikipedia.org/w/api.php"

    # Combine title and description for a more accurate search
    search_query = f"{title} {description}"

    # Parameters for the API request
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "format": "json",
        "srlimit": 1  # Limit the search to the first result
    }

    # Make the request to the Wikipedia API
    response = requests.get(wikipedia_search_url, params=params)
    data = response.json()

    # Extracting the Wikipedia ID (page title) from the response
    search_results = data.get("query", {}).get("search", [])
    if search_results:
        wikipedia_id = search_results[0].get("title")
        return wikipedia_id
    else:
        return "No matching entity found"


def get_entity_description(wikidata_id):
    # URL for the Wikidata API
    wikidata_api_url = "https://www.wikidata.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "format": "json",
        "props": "descriptions",
        "languages": "en"  # Assuming you want the description in English
    }

    # Make the request to the Wikidata API
    response = requests.get(wikidata_api_url, params=params)
    data = response.json()

    # Extracting the description from the response
    entities = data.get("entities", {})
    entity_data = entities.get(wikidata_id, {})
    description = entity_data.get("descriptions", {}).get("en", {}).get("value")

    return description


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def search_with_wikipedia_api(name):
    import requests

    S = requests.Session()

    URL = "https://en.wikipedia.org/w/api.php"

    # SEARCHPAGE = "Which of the following represents Cushing's response?"

    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": name
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    # print(DATA)
    ans = []
    for line in DATA["query"]["search"]:
        ans.append({
            "title": line["title"],
            "id": line["pageid"],
            "snippet": line["snippet"]
        })
    return ans


def search_for_wikipedia(entity_name=None, entity_id=None, entity_des="", 
                         topic=None, topic_des="", lang="eng", 
                         search_for="entity", index=None
                        ):
    
    if not index:
        index = "wikipedia-monthly-enwiki"
    
    es = ESClient()
    
    if search_for == "entity":
        assert entity_name is not None or entity_id is not None
        
        if entity_id is not None:
            body = {
                "query": {
                    "match": {
                        "id": entity_id
                    }
                }
            }
            
            results = es.search(index=index, body=body)
            if results["hits"]["total"]["value"] == 0:
                return None, None, None, None
        else:
            body = {
                "query": {
                    "match": {
                        "title": entity_name
                    }
                }
            }
            results = es.search(index=index, body=body)
            if results["hits"]["total"]["value"] == 0:
                return None, None, None, None
    
    elif search_for == "topic":
        assert topic is not None
        
        body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "title": {
                                    "query": topic,
                                    "boost": 3
                                }
                            }
                        },
                        {
                            "match": {
                                "text": {
                                    "query": topic_des,
                                    "boost": 1
                                }
                            }
                        }
                    ]
                }
            }
        }
        results = es.search(index=index, body=body)
        if results["hits"]["total"]["value"] == 0:
            return None, None, None, None
        
    
    res = results["hits"]["hits"][0]["_source"]
    # print(res)
    idx = res["id"]
    url = res["url"]
    title = res["title"]
    text = res["text"]
    return idx, title, url, text


def filter_para(paras):
    new_paras = []
    for line in paras:
        words = len(line.split(" "))
        if words <= 20:
            continue
        else:
            new_paras.append(line)
    return new_paras


def count_words(text):
    return len(text.split())

def trunct(text, num):
    first = text.split()[:num]
    second = text.split()[num:]
    return " ".join(first), "".join(second)



def trunc_wikipedia_page(text, title_length=5, min_words=256, max_words=2048):
    paras = text.split("\n")
    
    intro = ""
    for para in paras[:5]:
        if count_words(para) >= 15:
            intro = para
            break
    if intro == "":
        intro = paras[0]
    
    new_paras = []
    cur_text = ""
    for para in paras:
        # print(count_words(text))
        if count_words(para) <= title_length:
            if cur_text != "" and count_words(cur_text) >= min_words:
                if count_words(cur_text) > max_words:
                    cur_text, second = trunct(cur_text, max_words)
                    new_paras.append(cur_text)
                    new_paras.append(second)
                else:
                    new_paras.append(cur_text)
                cur_text = para
            else:
                cur_text = cur_text + "\n" + para
        else:
            cur_text = cur_text + "\n" + para
    
    if cur_text != "":
        new_paras.append(cur_text)
    
    return new_paras, intro


def parse_sorted_string(sorted_str):
    elements = sorted_str.replace('[', '').replace(']', '').replace(' ', '').split('>')
    
    result_list = [int(element) for element in elements if element]
    
    return result_list


def rank_gpt(query, passages):
    system_prompt = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query"
    
    rank_prompt = "I will provide you with {doc_num} passages, each indicated by number identifier []. Rank them based on their relevance to query".format(doc_num=len(passages))
    
    passage_prompt = ""
    
    for idx, doc in enumerate(passages):
        passage_prompt += "[{}] {}\n\n".format(idx, doc)
        
    
    query_prompt = "<begin of query>\n{}\n<end of query>".format(query)
    
    instruct_prompt = "Rank the {doc_num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be [] > [], e.g., [0] > [2] > [1]. Only response the ranking results, do not say any word or explain.".format(doc_num=len(passages))
    
    prompt = "{}\n\n{}\n\n{}\n\n{}".format(
        rank_prompt, passage_prompt, query_prompt, instruct_prompt
    )
    
    response = query_gpt(prompt, sys=system_prompt)
    rank_list = parse_sorted_string(response)
    return rank_list

def bge_rank(query, passages, model, tokenizer):
    model.eval()

    pairs = []
    for passage in passages:
        pairs.append([query, passage])
        
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # print(scores)
        rank = torch.argsort(scores, descending=True).tolist()
        # print(rank)
        return rank
    
    
def batch_bge_rank(query, passages, model, tokenizer, batch_size=8):
    model.eval()

    # 分割输入数据为批次
    batched_pairs = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        pairs = [[query, passage] for passage in batch]
        batched_pairs.append(pairs)

    all_scores = []
    all_indices = []

    with torch.no_grad():
        for pairs in tqdm(batched_pairs):
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.tolist())
            all_indices.extend(list(range(len(pairs))))

    # 对所有分数进行排序
    rank = [x for _, x in sorted(zip(all_scores, all_indices), key=lambda pair: pair[0], reverse=True)]
    # ranked_para = [passages[i] for i in rank]

    return rank



def wiki_retrieve_by_entity(args, topic_name, topic_des, seed_question, bge_model=None, beg_tokenizer=None, doc_num=1, wiki_info=None):
    if wiki_info is None:
        print(">>>Working on {}:{}".format(topic_name, topic_des))
        
        print(">>>Search on Wikipedia...")
        seach_ans = search_with_wikipedia_api(topic_name)
        if len(seach_ans) > 0:
            topic_id = seach_ans[0]["id"]
            topic_title = seach_ans[0]["title"]
            topic_snip = seach_ans[0]["snippet"]
            entity_id, entiti_title, url, text = search_for_wikipedia(entity_name=topic_name, entity_id=topic_id, search_for="entity")
        else:
            entity_id, entiti_title, url, text = search_for_wikipedia(topic=topic_name, topic_des=topic_des, search_for="topic")
        # print(seed_question)
        print(topic_name)
        # print(topic_snip)
        print("Search res by wikipedia: id-{}, name-{}".format(entity_id, entiti_title))
        
        paragraphs, intro = trunc_wikipedia_page(text, min_words=args.chunk_size)
        
        if args.use_openai_chunk:
            paragraphs = get_text_chunks(text, args.chunk_size)
        
        # print(paragraphs)
        
        print(">>> rank paras")
        if args.rank_method == "bge":
            rank = bge_rank(seed_question, paragraphs, bge_model, beg_tokenizer)
        elif args.rank_method == "gpt":
            rank = rank_gpt(seed_question, paragraphs)
        
        best_match = []
        for i in range(min(doc_num, len(paragraphs))):
            para = paragraphs[rank[i]]
            best_match.append(para)
        
        final_doc = "\n".join(best_match)
        
        if intro not in final_doc:
            final_doc = intro + "\n" + final_doc
        
        # print(final_doc)
        
        wiki_info = {
            "wiki_id": entity_id,
            "wiki_name": entiti_title,
            "wiki_intro": intro,
            "wiki_page": text,
            "related_content": final_doc
        }
        return wiki_info

