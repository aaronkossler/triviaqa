import sys
sys.path.append("../")
from data_preprocessing.preprocessing import create_splits
from pre_functions import *
import argparse
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
    "-d", "--domain",
    default="wikipedia",
    help="Specify the domain. Either wikipedia or web should be chosen"
)

parser.add_argument(
    "-g", "--gpu",
    default="yes",
    help="Choose whether to use gpu or not. Either yes or no should be chosen"
)

parser.add_argument(
    "-b", "--debug",
    default="no",
    help="Choose whether to use debug mode or not. Either yes or no should be chosen"
)

args = parser.parse_args()

# Loading test split
data_splits = create_splits(domain=args.domain)
test = data_splits["test"]

predictor = Predictor(args.model, args.domain, test, args.gpu, args.debug)
predictions = predictor.predict()

modelname = re.sub("/", "-", args.model)
save_predictions(predictions, f"predictions/{args.domain}", f"{modelname}_predictions.json")
