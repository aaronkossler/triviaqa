# -*- coding: utf-8 -*-

import os
import torch
import json
from tqdm import tqdm
from torch.optim import Adam
import evaluate  # Bleu
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import warnings

warnings.filterwarnings("ignore")


# Function to extract contexts, questions, and answers from the dataset
def prepare_data(data):
    articles = []

    for item in tqdm(data, desc="Preparing Data"):
        question = item["Question"]
        answer = item["Answer"]["Value"]

        texts = []
        for pages in item["EntityPages"]:
            filename = pages["Filename"]
            text = open(f"evidence/wikipedia/{filename}", mode="r", encoding="utf-8").read()
            texts.append(text)
        context = " ".join(texts)

        inputs = {"context": context, "question": question, "answer": answer}
        articles.append(inputs)

    return articles


class QA_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.context = self.data["context"]
        self.answer = self.data['answer']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]

        question_tokenized = self.tokenizer(question, context, max_length=self.q_len, padding="max_length",
                                            truncation=True, pad_to_max_length=True, add_special_tokens=True)
        answer_tokenized = self.tokenizer(answer, max_length=self.t_len, padding="max_length",
                                          truncation=True, pad_to_max_length=True, add_special_tokens=True)

        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100

        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }

# Setting Hyperparameters
# TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
# MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
TOKENIZER = T5TokenizerFast.from_pretrained("models/qa_model")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
MODEL.to("cuda")
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 256  # Question Length
T_LEN = 32  # Target Length
BATCH_SIZE = 8
DEVICE = "cuda:0"

# Loading train split from trivia_qa dataset and writing it into a dataframe
train = pd.read_json('data/wikipedia-train.json', encoding='utf-8')["Data"]
data = pd.DataFrame(prepare_data(train))
# Splitting train split according to task (first 7900 validation)
val_data, train_data = train_test_split(data, shuffle=False, train_size=7900)

# Setting up Samplers and Dataloaders
train_sampler = RandomSampler(train_data.index)
val_sampler = RandomSampler(val_data.index)

qa_dataset = QA_Dataset(TOKENIZER, data, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# Initializing training variables
train_loss = 0
val_loss = 0
train_batch_count = 0
val_batch_count = 0


if not os.path.exists("models"):
    os.makedirs("models")

# Training
for epoch in range(3):
    MODEL.train()
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        train_loss += outputs.loss.item()
        train_batch_count += 1

    # Evaluation
    MODEL.eval()
    for batch in tqdm(val_loader, desc="Validation batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        val_loss += outputs.loss.item()
        val_batch_count += 1

    print(
        f"{epoch + 1}/{2} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss / val_batch_count}")

    # Saving Model after an epoch
    MODEL.save_pretrained(f"models/t5-model-epoch-{epoch+3}")
    TOKENIZER.save_pretrained(f"models/t5_tokenizer-epoch-{epoch+3}")


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
test = pd.read_json('data/verified-wikipedia-dev.json', encoding='utf-8')["Data"]

# Model Prediction
predictions = {}
for entry in test:
    question = entry["Question"]
    answer = entry["Answer"]["Value"]

    texts = []
    for pages in entry["EntityPages"]:
        filename = pages["Filename"]
        text = file = open(f"evidence/wikipedia/{filename}", mode="r", encoding="utf-8").read()
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
