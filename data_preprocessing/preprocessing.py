# Import Dataset
import json
import os
import datasets

# Execute create splits to create the required data splits and write the evaluation sets as jsons

# create data splits
# Alternatively, set "Web" as domain
def create_splits(create_eval = True, write_path = "../eval_splits", domain = "Wikipedia"):
    if domain == "Wikipedia":
        trivia_qa = datasets.load_dataset('trivia_qa', name="rc.wikipedia")
    elif domain == "Web":
        trivia_qa = datasets.load_dataset('trivia_qa', name="rc.web")

    train_split = trivia_qa["train"].train_test_split(shuffle=False, train_size=7900)
    validation = train_split["train"]
    train = train_split["test"]
    test = trivia_qa["validation"]

    splits = {
        "train": train,
        "validation": validation,
        "test": test
    }

    if create_eval:
        eval_data = preprocess_eval_datasets(splits)
        write_files(eval_data, write_path)

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