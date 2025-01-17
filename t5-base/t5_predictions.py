import sys
sys.path.append("../")
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from data_preprocessing.preprocessing import create_splits
from t5_functions import *
import argparse
import re
from rag.retrievers.retriever import Retriever

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
    help="Specify which tokenizer should be used. Either set a path or a huggingface tokenizer name."
)

parser.add_argument(
    "-d", "--domain",
    default="wikipedia",
    help="Specify the domain. Either wikipedia or web should be chosen."
)

parser.add_argument(
    "-r", "--retriever",
    default=None,
    help="Specify which retriever should be used to obtain the context."
)

parser.add_argument(
    "--type",
    default="validation",
    help="Specify whether the pipeline should be tested on validation or test data. Has to be either 'validation' or "
         "'test'."
)

parser.add_argument(
   "--format_text",
   action='store_const',
   const=not False,
   default=False,
   help="Specify if the context should be cleaned."
)

args = parser.parse_args()

# Setting Hyperparameters
TOKENIZER = T5TokenizerFast.from_pretrained(args.tokenizer)
MODEL = T5ForConditionalGeneration.from_pretrained(args.model)
MODEL.to("cuda")
Q_LEN = 256  # Question Length
DEVICE = "cuda:0"

# Loading test split
data_splits = create_splits(domain=args.domain)

if args.type == "validation":
    test = data_splits["validation"]
else:
    test = data_splits["test"]

# Initialize retriever
if args.retriever:
    retriever = Retriever(args.retriever)
else:
    retriever = None

# Generating predictions
predictor = Predictor(MODEL, TOKENIZER, args.domain, test, Q_LEN, DEVICE, retriever)
predictions, analysis = predictor.predict(args.format_text)

modelname = re.sub("/", "-", args.model)
save_predictions(predictions, f"predictions/{args.domain}", f"{args.type}_{modelname}_{args.retriever}_{args.format_text}.json")
save_predictions(analysis, f"predictions/{args.domain}", f"analysis_{args.type}_{modelname}_{args.retriever}_{args.format_text}.json")
