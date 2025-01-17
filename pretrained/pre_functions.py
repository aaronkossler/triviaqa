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


def read_json(path):
    return open(path, mode="r", encoding="utf-8").read()


def run_pipeline(documents, reader, query, top_k):
    retriever = BM25Retriever(document_store=documents)
    pipe = ExtractiveQAPipeline(reader, retriever)
    prediction = pipe.run(
        query=query,
        params={"Retriever": {"top_k": top_k}, "Reader": {"top_k": top_k}})
    if prediction["answers"]:
        return prediction["answers"][0].answer
    else:
        return ""


class Predictor:
    def __init__(self, model, domain, test, gpu, debug):
        self.model = model
        self.domain = domain
        self.test = test
        if gpu == "yes":
            self.gpu = True
        else:
            self.gpu = False
        if debug == "yes":
            self.debug = True
        else:
            self.debug = False

    def build_document_stores(self):
        documents = {}
        if self.domain == "wikipedia":
            for row in tqdm(self.test, desc="Building Document Stores"):
                document_store = InMemoryDocumentStore(use_bm25=True)
                for page in row["EntityPages"]:
                    filename = page["Filename"]
                    article = read_json(f"../triviaqa_data/evidence/wikipedia/{filename}")
                    document = {
                        "content": article,
                        "meta": {
                            "question_id": row["QuestionId"]
                        },
                    }
                    document_store.write_documents([document])
                documents[row["QuestionId"]] = document_store
        if self.domain == "web":
            for row in self.test:
                for index, page in enumerate(row["EntityPages"]):
                    filename = page["Filename"]
                    article = read_json(f"../triviaqa_data/evidence/wikipedia/{filename}")
                    document_store = article_to_document_store(article, row["QuestionId"])
                    documents[f"{row['QuestionId']}--{filename}"] = document_store
                for index, result in enumerate(row["SearchResults"]):
                    filename = result["Filename"]
                    article = read_json(f"../triviaqa_data/evidence/web/{filename}")
                    document_store = article_to_document_store(article, row["QuestionId"])
                    documents[f"{row['QuestionId']}--{filename}"] = document_store
        return documents

    def reader(self):
        return FARMReader(model_name_or_path=self.model, use_gpu=self.gpu)

    def predict(self):
        documents = self.build_document_stores()
        reader = self.reader()
        predictions = {}
        if self.domain == "wikipedia":
            for entry in tqdm(self.test, desc="Predicting Answers"):
                prediction = run_pipeline(documents[entry['QuestionId']], reader, entry['Question'], 1)
                predictions[entry['QuestionId']] = prediction
        if self.domain == "web":
            for entry in tqdm(self.test, desc="Predicting Answers"):
                for page in entry["EntityPages"]:
                    filename = page["Filename"]
                    prediction = run_pipeline(documents[f"{entry['QuestionId']}--{filename}"], reader,
                                              entry['Question'], 1)
                    if self.debug:
                        print(f"Question: {entry['Question']}")
                        print(f"Answers: {prediction}")
                        print(f"Filename: {filename}")
                    predictions[f"{entry['QuestionId']}--{filename}"] = prediction
                for result in entry["SearchResults"]:
                    filename = result["Filename"]
                    prediction = run_pipeline(documents[f"{entry['QuestionId']}--{filename}"], reader,
                                              entry['Question'], 1)
                    if self.debug:
                        print(f"Question: {entry['Question']}")
                        print(f"Answers: {prediction}")
                        print(f"Filename: {filename}")
                    predictions[f"{entry['QuestionId']}--{filename}"] = prediction

        return predictions
