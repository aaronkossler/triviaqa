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
import re


# Execute create splits to create the required data splits and write the evaluation sets as jsons

def build_abs_path():
    # Get the current working directory
    current_working_directory = os.getcwd()

    # Find the last occurrence of "triviaqa" in the current working directory
    last_occurrence_index = current_working_directory.rfind("trivia_qa")

    # Truncate the path after the last occurrence of "triviaqa"
    truncated_path = current_working_directory[:last_occurrence_index + len("trivia_qa") + 1]
    data_path = truncated_path + "triviaqa_data/"

    return data_path


# create data splits
# Alternatively, set "web" as domain
def create_splits(hf_datasets=False, as_list_of_dicts=False, create_eval=False, write_path="../eval_splits",
                  domain="wikipedia"):
    if domain == "wikipedia":
        val_size = 7900
    elif domain == "web":
        val_size = 9500
    # download via datasets module
    if hf_datasets:
        trivia_qa = datasets.load_dataset('trivia_qa', name=f"rc.{domain}")
        train_split = trivia_qa["train"].train_test_split(shuffle=False, train_size=val_size)

        validation = train_split["train"]
        train = train_split["test"]
        test = trivia_qa["validation"]
    # download from website
    else:
        data_path = build_abs_path()
        # print(bool(os.path.exists(data_path) and os.listdir(data_path)))
        # exit()
        if not (os.path.exists(data_path) and os.listdir(data_path)):
            print("Downloading data...")
            wget.download("https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz", out="../triviaqa-rc.tar.gz")
            with tarfile.open("../triviaqa-rc.tar.gz", "r:gz") as tar:
                tar.extractall(path=data_path)

        train_val = pd.DataFrame(pd.read_json(data_path + f'/qa/{domain}-train.json', encoding='utf-8'))["Data"]
        validation, train = train_test_split(train_val, shuffle=False, train_size=val_size)
        test = pd.DataFrame(pd.read_json(data_path + f'/qa/{domain}-dev.json', encoding='utf-8'))["Data"]

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

    if create_eval and as_list_of_dicts:
        # eval_data = preprocess_eval_datasets(splits)
        eval_data = {
            "validation": splits["validation"],
            "test": splits["test"],
            "train": splits["train"]
        }
        write_files(eval_data, write_path, domain)

    return splits


# Convert the evaluation data (= validation and test) to the desired format
def preprocess_eval_datasets(data, convert_eval=["validation", "test", "train"]):
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
            search_results = [
                {
                    "Description": item["search_results"]["description"][index],
                    "Filename": item["search_results"]["filename"][index],
                    "Rank": item["search_results"]["rank"][index],
                    "Title": item["search_results"]["title"][index],
                    "Url": item["search_results"]["url"][index]
                }
                for index in range(len(item["search_results"]["filename"]))
            ]
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


def cleanup_context(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'File:.*\n', '', text)
    return text


def page_to_context(page, domain, format_text):
    filename = page["Filename"]
    text = open(f"{build_abs_path()}/evidence/{domain}/{filename}", mode="r", encoding="utf-8").read()
    if format_text:
        text = cleanup_context(text)
    return text


def build_context(item, domain, format_text=False):
    context = ""
    if domain == "wikipedia":
        texts = []
        for page in item["EntityPages"]:
            text = page_to_context(page, domain, format_text)
            texts.append(text)
        context = " ".join(texts)
    if domain == "web":
        context = {}
        for page in item["EntityPages"]:
            text = page_to_context(page, domain, format_text)
            context[page["Filename"]] = text
        for result in item["SearchResults"]:
            text = page_to_context(result, domain, format_text)
            context[result["Filename"]] = text

    return context
