# Server pecific fix
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add CLI args
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size",
    default="8",
    help="Set batch size for GPU inference."
)

parser.add_argument(
    "--variant",
    help="Specify the name of the variant the results should be logged with."
)

parser.add_argument(
    "--type",
    default="validation",
    help="Specify whether the pipeline should be tested on validation or test data. Has to be either 'validation' or 'test'."
)

parser.add_argument(
    "--retriever",
    default="hlatr",
    help="Specify which retriever should be used to obtain the context."
)

parser.add_argument(
    "--embeddings",
    default="WhereIsAI/UAE-Large-V1",
    help="Specify which embeddings the retriever should use (if necessary)."
)

parser.add_argument(
    "--model",
    default="google/flan-t5-base",
    help="Specify the model that should be applied for answer generation."
)

parser.add_argument(
    "--format_text",
    action='store_const',
    const=not False,
    default=False,
    help="Specify if the context should be cleaned."
)

parser.add_argument(
    "--with_headers",
    action='store_const',
    const=not False,
    default=False,
    help="Specify if headers should be prepended to paragraphs."
)

parser.add_argument(
    "--max_par_len",
    default=1000000,
    help="Specify the maximum length of paragraphs."
)

parser.add_argument(
    "--topx_contexts",
    default=1,
    help="Specify the number of top contexts that should be concatenated to build the context for the generator (only available with max_par_len)."
)

parser.add_argument(
    "--top_par_thresh",
    default=0,
    help="Specify the minimum/maximum score that is assigned to paragraphs outside the #1 to be appended to the context (> 1 for faiss, < 1 for hlatr)."
)

args = parser.parse_args()

# Load data
import json
import sys

sys.path.append("..")

from data_preprocessing.preprocessing import create_splits, build_context

domain = "wikipedia"

data_splits = create_splits(as_list_of_dicts=True, domain=domain)

# Import relevant modules for langchain
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline

# Load LLM that should act as a generator
if os.path.exists(args.model):
    print("Found local model, loading from disk.")
    from transformers import T5ForConditionalGeneration, T5TokenizerFast, pipeline

    # Load the model and tokenizer from a local checkpoint
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    tokenizer = T5TokenizerFast.from_pretrained(args.model)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs={"max_new_tokens": 10},
                              device_map="auto", batch_size=int(args.batch_size))
else:
    llm = HuggingFacePipeline.from_model_id(model_id=args.model, task="text2text-generation",
                                            pipeline_kwargs={"max_new_tokens": 10}, device_map="auto",
                                            batch_size=int(args.batch_size))

# Build retriever with given information
from retrievers.retriever import Retriever

retriever = Retriever(args.retriever, args.embeddings, int(args.max_par_len), args.with_headers,
                      int(args.topx_contexts), float(args.top_par_thresh))


# Batch pipeline
# Process batch of items to be prapared for batch prediction
def prepare_rag_chain_data(items):
    questions = [item["Question"] for item in items]
    contexts = [build_context(item, domain, args.format_text) for item in items]

    inputs = []
    for i, question in enumerate(questions):
        item = {"question": question, "context": retriever.retrieve(question, contexts[i])}
        inputs.append(item)

    return inputs


# Create rag chain as suggested by langchain
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
        RunnablePassthrough()
        | prompt
        | llm.bind(stop=["\n\n"])
)


# Execute prediction for a batch of questions
def batch_prediction(questions):
    inputs = prepare_rag_chain_data(questions)
    answers = rag_chain.batch(inputs)

    for i, input in enumerate(inputs):
        input["answer"] = answers[i]

    return inputs


# Save files
import os


def save_file(data, write_path, filename):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    with open(write_path + "/{}.json".format(filename), "w") as f:
        json.dump(data, f)


# Collect results for specified data set
from tqdm import tqdm
import math


def rag_prediction(model_name, batch_size, type):
    data = data_splits[type]

    results = {}

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    progress_bar = tqdm(total=math.ceil(len(data) / batch_size), desc="Validation Progress", unit="batch")

    for item in batch(data, batch_size):
        answers = batch_prediction(item)
        for i, prediction in enumerate(answers):
            print(prediction)
            qid = item[i]["QuestionId"]
            results[qid] = prediction

        progress_bar.update(1)

    save_file(results, "./results/" + model_name + "/", "{}_{}_analysis".format("wiki", type))

    eval_format = {key: inner_dict["answer"] for key, inner_dict in results.items()}
    save_file(eval_format, "./results/" + model_name + "/", "{}_{}_results".format("wiki", type))


# Start predictions with specified cli params
rag_prediction(args.variant, int(args.batch_size), args.type)
