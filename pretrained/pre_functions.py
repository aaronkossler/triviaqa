import json
import os
from tqdm import tqdm
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline


def save_predictions(predictions, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    # Convert the dictionary to a JSON string
    json_string = json.dumps(predictions)

    # Write the JSON string to a file
    with open(f"{path}/{filename}", "w") as f:
        f.write(json_string)


def article_to_document_store(article, question_id):
    document_store = InMemoryDocumentStore(use_bm25=True)
    document = {
        "content": article,
        "meta": {
            "question_id": question_id
        },
    }
    document_store.write_documents([document])
    return document_store


class Predictor:
    def __init__(self, model, domain, test, gpu):
        self.model = model
        self.domain = domain
        self.test = test
        if gpu == "yes":
            self.gpu = True
        else:
            self.gpu = False

    def build_document_stores(self):
        documents = {}
        if self.domain == "wikipedia":
            for row in tqdm(self.test, desc="Building Document Stores"):
                document_store = InMemoryDocumentStore(use_bm25=True)
                for page in row["EntityPages"]:
                    filename = page["Filename"]
                    article = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r",
                                   encoding="utf-8").read()
                    document = {
                        "content": article,
                        "meta": {
                            "question_id": row["QuestionId"]
                        },
                    }
                    document_store.write_documents([document])
                documents[row["question_id"]] = document_store
        if self.domain == "web":
            for row in self.test:
                for index, page in enumerate(row["EntityPages"]):
                    filename = page["Filename"]
                    article = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r",
                                   encoding="utf-8").read()
                    document_store = article_to_document_store(article, row["QuestionId"])
                    documents[f"{row['QuestionId']}--{row['EntityPages']['Filename'][index]}"] = document_store
                for index, result in enumerate(row["SearchResults"]):
                    filename = result["Filename"]
                    article = open(f"../triviaqa_data/evidence/web/{filename}", mode="r",
                                   encoding="utf-8").read()
                    document_store = article_to_document_store(article, row["QuestionId"])
                    documents[f"{row['QuestionId']}--{row['SearchResults']['Filename'][index]}"] = document_store

        return documents

    def reader(self):
        return FARMReader(model_name_or_path=self.model, use_gpu=self.gpu)

    def predict(self):
        documents = self.build_document_stores()
        reader = self.reader()
        predictions = {}
        if self.domain == "wikipedia":
            for entry in tqdm(self.test, desc="Predicting Answers"):
                retriever = BM25Retriever(document_store=documents[entry['QuestionId']])
                pipe = ExtractiveQAPipeline(reader, retriever)
                prediction = pipe.run(
                    query=entry["Question"],
                    params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
                predictions[entry['QuestionId']] = prediction["Answers"][0].answer
        if self.domain == "web":
            for entry in tqdm(self.test, desc="Predicting Answers"):
                for index, page in enumerate(entry["EntityPages"]):
                    retriever = BM25Retriever(document_store=documents[f"{entry['QuestionId']}--{page['Filename']}"])
                    pipe = ExtractiveQAPipeline(reader, retriever)
                    prediction = pipe.run(
                        query=entry["Question"],
                        params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
                    predictions[f"{entry['QuestionId']}--{page['Filename']}"] = prediction["Answers"][0].answer
                for index, result in enumerate(entry["SearchResults"]):
                    retriever = BM25Retriever(document_store=documents[f"{entry['QuestionId']}--{result['Filename']}"])
                    pipe = ExtractiveQAPipeline(reader, retriever)
                    prediction = pipe.run(
                        query=entry["Question"],
                        params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}})
                    predictions[f"{entry['QuestionId']}--{result['Filename']}"] = prediction["Answers"][0].answer
        return predictions