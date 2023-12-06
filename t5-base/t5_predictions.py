import sys
sys.path.append("../")
import torch
from torch.optim import Adam
import json
import os
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from data_preprocessing.preprocessing import create_splits
import argparse
from tqdm import tqdm
import re

# server specific fix
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# add cli args
parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model",
    help="Specify which model should be used. Either set a path or a huggingface model name."
)

parser.add_argument(
    "-t", "--tokenizer",
    help="Specify which tokenizer should be used. Either set a path or a huggingface model name."
)

parser.add_argument(
    "-d", "--domain",
    default="wikipedia",
    help="Specify the domain. Either wikipedia or web should be chosen"
)

args = parser.parse_args()

# Setting Hyperparameters
TOKENIZER = T5TokenizerFast.from_pretrained(args.tokenizer)
MODEL = T5ForConditionalGeneration.from_pretrained(args.model)
MODEL.to("cuda")
Q_LEN = 256  # Question Length
DEVICE = "cuda:0"


def predict_answer(context, question, ref_answer=None):
    inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True,
                       add_special_tokens=True)

    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)

    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)

    return predicted_answer


# Loading test split
data_splits = create_splits(domain=args.domain)
test = data_splits["test"]

# Model Prediction
predictions = {}
for entry in tqdm(test, desc="Predicting Answers"):
    question = entry["Question"]

    if args.domain == "wikipedia":
        texts = []
        for pages in entry["EntityPages"]:
            filename = pages["Filename"]
            text = file = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r", encoding="utf-8").read()
            texts.append(text)
        context = " ".join(texts)
        predictions[entry["QuestionId"]] = predict_answer(context, question)
    elif args.domain == "web":
        for pages in entry["SearchResults"]:
            filename = pages["Filename"]
            context = open(f"../triviaqa_data/evidence/web/{filename}", mode="r", encoding="utf-8").read()
            predictions[f"{entry['QuestionId']}--{filename}"] = predict_answer(context, question)

if not os.path.exists("predictions"):
    os.makedirs("predictions")

# Convert the dictionary to a JSON string
json_string = json.dumps(predictions)

# Write the JSON string to a file
modelname = re.sub("/", "-", args.model)
with open(f"predictions/{args.domain}_{modelname}_predictions.json", "w") as f:
    f.write(json_string)
