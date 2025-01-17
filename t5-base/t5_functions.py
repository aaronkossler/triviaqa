import torch
import json
import os
from tqdm import tqdm
from data_preprocessing.preprocessing import cleanup_context


def save_predictions(predictions, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    # Convert the dictionary to a JSON string
    json_string = json.dumps(predictions)

    # Write the JSON string to a file
    with open(f"{path}/{filename}", "w") as f:
        f.write(json_string)


class Predictor:
    def __init__(self, model, tokenizer, domain, test, q_len=256, device="cuda:0", retriever=None):
        self.model = model
        self.tokenizer = tokenizer
        self.domain = domain
        self.test = test
        self.q_len = q_len
        self.device = device
        self.retriever = retriever

    def predict_answer(self, context, question):
        inputs = self.tokenizer(question, context, max_length=self.q_len, padding="max_length", truncation=True,
                                add_special_tokens=True)

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(self.device).unsqueeze(0)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(self.device).unsqueeze(0)

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)

        predicted_answer = self.tokenizer.decode(outputs.flatten(), skip_special_tokens=True)

        return predicted_answer

    # Model Prediction
    def predict(self, format_text=False):
        predictions = {}
        analysis = {}
        for entry in tqdm(self.test, desc="Predicting Answers"):
            question = entry["Question"]

            if self.domain == "wikipedia":
                texts = []
                for pages in entry["EntityPages"]:
                    filename = pages["Filename"]
                    text = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r",
                                encoding="utf-8").read()
                    texts.append(text)
                context = " ".join(texts)
                if format_text:
                    context = cleanup_context(context)
                if self.retriever:
                    context = self.retriever.retrieve(question, context)
                answer = self.predict_answer(context, question)
                predictions[entry["QuestionId"]] = answer
                analysis[entry["QuestionId"]] = {"question": question, "context": context, "answer": answer}
            elif self.domain == "web":
                for result in entry["SearchResults"]:
                    filename = result["Filename"]
                    context = open(f"../triviaqa_data/evidence/web/{filename}", mode="r", encoding="utf-8").read()
                    predictions[f"{entry['QuestionId']}--{filename}"] = self.predict_answer(context, question)
                for page in entry["EntityPages"]:
                    filename = page["Filename"]
                    context = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r", encoding="utf-8").read()
                    predictions[f"{entry['QuestionId']}--{filename}"] = self.predict_answer(context, question)

        return predictions, analysis
