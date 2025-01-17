# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import warnings
from data_preprocessing.preprocessing import create_splits
from t5_functions import *
import pandas as pd
import argparse
import re

warnings.filterwarnings("ignore")

# server specific fix
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# add cli args
parser = argparse.ArgumentParser()

parser.add_argument(
    "-b", "--batch_size",
    default="8",
    help="Set batch size for training."
)

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

parser.add_argument(
    "-e", "--epochs",
    default="5",
    help="Specify for how many epochs the model should be trained"
)

args = parser.parse_args()

modelname = re.sub("/", "-", args.model)

# Setting Hyperparameters
TOKENIZER = T5TokenizerFast.from_pretrained(args.tokenizer)
MODEL = T5ForConditionalGeneration.from_pretrained(args.model, return_dict=True)
MODEL.to("cuda")
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 256  # Question Length
T_LEN = 32  # Target Length
BATCH_SIZE = int(args.batch_size)
DEVICE = "cuda:0"


# Function to extract contexts, questions, and answers from the dataset
def prepare_data(data):
    articles = []

    for item in tqdm(data, desc="Preparing Data"):
        question = item["Question"]
        answer = item["Answer"]["Value"]

        if args.domain == "wikipedia":
            texts = []
            for pages in item["EntityPages"]:
                filename = pages["Filename"]
                text = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r", encoding="utf-8").read()
                texts.append(text)
            context = " ".join(texts)
            inputs = {"context": context, "question": question, "answer": answer}
            articles.append(inputs)

        elif args.domain == "web":
            for result in item["SearchResults"]:
                filename = result["Filename"]
                context = open(f"../triviaqa_data/evidence/web/{filename}", mode="r", encoding="utf-8").read()
                inputs = {"context": context, "question": question, "answer": answer}
                articles.append(inputs)
            for page in item["EntityPages"]:
                filename = page["Filename"]
                context = open(f"../triviaqa_data/evidence/wikipedia/{filename}", mode="r", encoding="utf-8").read()
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


# Splitting train split according to task
data_splits = create_splits(domain=args.domain)
train = data_splits["train"]
validation = data_splits["validation"]
data = pd.DataFrame(prepare_data(pd.concat([validation, train], ignore_index=True)))

# Setting up Samplers and Dataloaders
train_sampler = RandomSampler(train.index)
val_sampler = RandomSampler(validation.index)

qa_dataset = QA_Dataset(TOKENIZER, data, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# Initializing training variables
train_loss = 0
val_loss = 0
train_batch_count = 0
val_batch_count = 0

if not os.path.exists(f"models/{args.domain}"):
    os.makedirs(f"models/{args.domain}")

# Training
for epoch in range(int(args.epochs)):
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
    MODEL.save_pretrained(f"models/{args.domain}/{modelname}/{modelname}-epoch-{epoch + 1}")
    TOKENIZER.save_pretrained(f"models/{args.domain}/{modelname}/{modelname}-epoch-{epoch + 1}")

# Loading test split
test = data_splits["test"]

# Generating predictions
predictor = Predictor(MODEL, TOKENIZER, args.domain, test, Q_LEN, DEVICE)
predictions = predictor.predict()
save_predictions(predictions, f"predictions/{args.domain}", f"{args.batch_size}_{modelname}_predictions.json")
