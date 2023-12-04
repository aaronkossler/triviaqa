import sys
sys.path.append("../")
import evaluate
import torch
from torch.optim import Adam
import json
import os
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from data_preprocessing.preprocessing import create_splits

TOKENIZER = T5TokenizerFast.from_pretrained("google/flan-t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("google/flan-t5-baset5_pipeline.py")
MODEL.to("cuda")
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 256  # Question Length
T_LEN = 32  # Target Length
BATCH_SIZE = 8
DEVICE = "cuda:0"


def predict_answer(context, question, ref_answer=None):
    inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True,
                       add_special_tokens=True)

    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)

    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)

    if ref_answer:
        # Load the Bleu metric
        bleu = evaluate.load("google_bleu")
        score = bleu.compute(predictions=[predicted_answer],
                             references=[ref_answer])

        print("Context: \n", context)
        print("\n")
        print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer,
            "Predicted Answer: ": predicted_answer,
            "BLEU Score: ": score
        }
    else:
        return predicted_answer


# Loading test split
# test = pd.read_json('data/verified-wikipedia-dev.json', encoding='utf-8')["Data"]
domain = "wikipedia"
data_splits = create_splits(domain=domain)
test = data_splits["test"]

# Model Prediction
predictions = {}
for entry in test:
    question = entry["Question"]
    answer = entry["Answer"]["Value"]

    texts = []
    for pages in entry["EntityPages"]:
        filename = pages["Filename"]
        text = file = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r", encoding="utf-8").read()
        texts.append(text)
    context = " ".join(texts)
    predictions[entry["QuestionId"]] = predict_answer(context, question)

if not os.path.exists("predictions"):
    os.makedirs("predictions")

# Convert the dictionary to a JSON string
json_string = json.dumps(predictions)

# Write the JSON string to a file
with open("predictions/t5_predictions.json", "w") as f:
    f.write(json_string)
