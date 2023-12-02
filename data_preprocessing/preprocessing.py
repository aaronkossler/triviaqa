# Import Dataset
import json
import os
import datasets
import wget
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")

# Execute create splits to create the required data splits and write the evaluation sets as jsons

# create data splits
# Alternatively, set "web" as domain
def create_splits(hf_datasets = False, as_list_of_dicts = False, create_eval = True, write_path = "../eval_splits", domain = "wikipedia"):
    # download via datasets module
    if hf_datasets:
        if domain == "wikipedia":
            trivia_qa = datasets.load_dataset('trivia_qa', name="rc.wikipedia")
        elif domain == "web":
            trivia_qa = datasets.load_dataset('trivia_qa', name="rc.web")

        train_split = trivia_qa["train"].train_test_split(shuffle=False, train_size=7900)
        validation = train_split["train"]
        train = train_split["test"]
        test = trivia_qa["validation"]
    # download from website
    else:
        data_path = "../triviaqa_data"
        #print(bool(os.path.exists(data_path) and os.listdir(data_path)))
        #exit()
        if not (os.path.exists(data_path) and os.listdir(data_path)):
            print("Downloading data...")
            wget.download("https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz", out="../triviaqa-rc.tar.gz")
            with tarfile.open("../triviaqa-rc.tar.gz", "r:gz") as tar:
                tar.extractall(path=data_path)

        train_val = pd.DataFrame(pd.read_json(data_path+'/qa/wikipedia-train.json', encoding='utf-8'))["Data"]
        validation, train = train_test_split(train_val, shuffle=False, train_size=7900)
        test = pd.DataFrame(pd.read_json(data_path+'/qa/wikipedia-dev.json', encoding='utf-8'))["Data"]


    if as_list_of_dicts:
        splits = {
            "train": train.tolist(),
            "validation": validation.tolist(),
            "test": test.tolist()
        }
    else:
        splits = {
            "train": train,
            "validation": validation,
            "test": test
        }
    """
    if create_eval:
        #eval_data = preprocess_eval_datasets(splits)
        eval_data = {
            "validation": splits["validation"],
            "test": splits["test"]
        }
        write_files(eval_data, write_path, domain)"""

    return splits

# Convert the evaluation data (= validation and test) to the desired format
def preprocess_eval_datasets(data, convert_eval = ["validation", "test"]):
    evaluation = {}

    for split in convert_eval:
        converted_data = []
        for item in data[split]:
            answer = {
                "Aliases": item["answer"]["aliases"],
                "MatchedWikiEntityName": item["answer"]["matched_wiki_entity_name"],
                "NormalizedAliases": item["answer"]["normalized_aliases"],
                "NormalizedMatchedWikiEntityName": item["answer"]["normalized_matched_wiki_entity_name"],
                "NormalizedValue": item["answer"]["normalized_value"],
                "Type": item["answer"]["type"],
                "Value": item["answer"]["value"],
            }
            entity_pages = [
                {
                    "DocSource": item["entity_pages"]["doc_source"][index],
                    "Filename": item["entity_pages"]["filename"][index],
                    "Title": item["entity_pages"]["title"][index],
                }
                for index in range(len(item["entity_pages"]["filename"]))
            ]
            question = item["question"]
            question_id = item["question_id"]
            question_source = item["question_source"]
            search_results = []
            data_item = {
                "Answer": answer,
                "EntityPages": entity_pages,
                "Question": question,
                "QuestionId": question_id,
                "QuestionSource": question_source,
                "SearchResults": search_results,
            }
            converted_data.append(data_item)

        evaluation[split] = converted_data

    return evaluation

def write_files(eval_data, write_path, domain):
    for key, val in eval_data.items():
        output = {
            "Data": val,
            "Domain": domain,
            "VerifiedEval": False,
            "Version": 1.0,
        }
        # Write the output to a JSON file
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        with open(write_path + "/{}_{}.json".format(key, domain), "w") as f:
            json.dump(output, f)

def build_context(item, domain):
    texts = []
    for pages in item["EntityPages"]:
        filename = pages["Filename"]
        text = open(f"../triviaqa_data/evidence/{domain}/{filename}", mode="r", encoding="utf-8").read()
        texts.append(text)
    context = " ".join(texts)

    return context