import sys
sys.path.append("..")

from evaluation.triviaqa_evaluation import evaluate_triviaqa
from utils.dataset_utils import *
from utils.utils import read_json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    help="Specify the path where the model answers are stored."
)

parser.add_argument(
    "--type",
    default="test",
    help="Specify whether the pipeline should be tested on validation or test data. Has to be either 'validation' or 'test'."
)

args = parser.parse_args()


dataset_file = './{}_wikipedia.json'.format(args.type)
prediction_file = args.path

def evaluate_model():
    expected_version = 1.0
    dataset_json = read_triviaqa_data(dataset_file)
    if dataset_json['Version'] != expected_version:
        print('Evaluation expects v-{} , but got dataset with v-{}'.format(expected_version,dataset_json['Version']),
            file=sys.stderr)
    key_to_ground_truth = get_key_to_ground_truth(dataset_json)
    predictions = read_json(prediction_file)
    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)

    print(eval_dict)

evaluate_model()