# %% [markdown]
# RAG Pipeline with LangChain

# %%
""" Pip installs for Google colab
!pip install datasets
!pip install langchain
!pip install sentence_transformers
!pip install annoy
!pip install langchainhub
!pip3 install pinecone-client==3.0.0rc2
!pip install faiss-gpu
"""

# server specific fix
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# add cli args
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
    default="langchain-vs",
    help="Specify which retriever should be used to obtain the context."
)

parser.add_argument(
    "--embeddings",
    default="all-MiniLM-L6-v2",
    help="Specify which embeddings the retriever should use (if necessary)."
)

args = parser.parse_args()

# %%
# Build retriever with given information
from retrievers.retriever import Retriever
retriever = Retriever(args.retriever, args.embeddings)

# %% [markdown]
# ### Load eval data

# %%
import json
import sys
sys.path.append("..")

read_files = ["test_Wikipedia.json", "validation_Wikipedia.json"]

from data_preprocessing.preprocessing import create_splits, build_context

domain = "wikipedia"

data_splits = create_splits(as_list_of_dicts=True, domain=domain)

# %% [markdown]
# Import relevant modules for langchain

# %%
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough


# %% [markdown]
# Create custom LLM class to post requests to t5 (hosted by huggingface)

# %%
# from: https://github.com/AndreasFischer1985/code-snippets/blob/master/py/LangChain_HuggingFace_examples.py

from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", task="text2text-generation", pipeline_kwargs={"max_new_tokens": 10}, device_map="auto", batch_size=int(args.batch_size))

# %% [markdown]
# ## Implementation of RAG pipeline

# %% [markdown]
# Currently most basic version:
# - Use Splitter to divide text into paragraphs
# - Create Vectorstore with HuggingFaceEmbeddings
# - Retrieve most similar chunk for the respective prompt
# - Send prompt to specified LLM and print response
# - Recently added: batches for more efficiency

# %%
# Batch pipeline

# process batch of items to be prapared for batch prediction
def prepare_rag_chain_data(items):
    questions = [item["Question"] for item in items]
    contexts = [build_context(item, domain) for item in items]
    
    inputs = []
    for i, question in enumerate(questions):
      item = {"question": question, "context": retriever.retrieve(question, contexts[i])}
      inputs.append(item)

    return inputs

# create rag chain as suggested by langchain
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
  RunnablePassthrough()
  | prompt
  | llm.bind(stop=["\n\n"])
)

# execute prediction for a batch of questions
def batch_prediction(questions):
    inputs = prepare_rag_chain_data(questions)
    answers = rag_chain.batch(inputs)

    for i, input in enumerate(inputs):
       input["answer"] = answers[i]

    return inputs

# %% [markdown]
# ### Collect results for specified data set

# %%
# save files
import os
def save_file(data, write_path, filename):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    with open(write_path + "/{}.json".format(filename), "w") as f:
        json.dump(data, f)

# %%
from tqdm import tqdm
import math

def rag_prediction(model_name, batch_size, type):
    data = data_splits[type]
    
    results = {}

    def batch(iterable, n=1):
      l = len(iterable)
      for ndx in range(0, l, n):
          yield iterable[ndx:min(ndx + n, l)]

    progress_bar = tqdm(total=math.ceil(len(data)/batch_size), desc="Validation Progress", unit="batch")

    for item in batch(data, batch_size):
        #print(item)
        answers = batch_prediction(item)
        for i, prediction in enumerate(answers):
          print(prediction)
          qid = item[i]["QuestionId"]
          results[qid] = prediction

        progress_bar.update(1)

    save_file(results, "./results/"+model_name+"/", "{}_{}_analysis".format("wiki", type))

    eval_format = {key: inner_dict["answer"] for key, inner_dict in results.items()}
    save_file(eval_format, "./results/"+model_name+"/", "{}_{}_results".format("wiki", type))


# %%
# start predictions with specified cli params
rag_prediction(args.variant, int(args.batch_size), args.type)

