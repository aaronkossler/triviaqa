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
import argparse
parser = argparse.ArgumentParser()

#argument to set which game variant/rules you want to train an agent for
parser.add_argument(
    "--batch_size",
    default="8",
    help="Set batch size for GPU inference."
)

#argument to set the training scenarios for the agent
parser.add_argument(
    "--variant",
    help="Specify the name of the variant the results should be logged with."
)

args = parser.parse_args()

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

def read_file(path):
    with open("../eval_splits/" + path) as f:
        data = json.load(f)
        return data

#test = read_file(read_files[0])
#validation = read_file(read_files[1])

# %% [markdown]
# Import relevant modules for langchain

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Annoy, FAISS
from langchain import hub
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence

# %% [markdown]
# Create custom LLM class to post requests to t5 (hosted by huggingface)

# %%
# from: https://github.com/AndreasFischer1985/code-snippets/blob/master/py/LangChain_HuggingFace_examples.py

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import re
class CustomLLM(LLM):
  def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    prompt_length = len(prompt)
    model_id = "google/flan-t5-large"
    params={"max_length":50, "length_penalty":2, "num_beams":16, "early_stopping":True}
    #print("LLM prompt ->",prompt)
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    # tbd: check if parameters are useful (did not work initially with given code)
    post = requests.post(url, json={"inputs":prompt})#, "parameters":params})
    #print(post.json())
    output = post.json()[0]["generated_text"]
    return output
  @property
  def _llm_type(self) -> str:
    return "custom"

llm=CustomLLM()

# %%
"""from transformers import pipeline
model_name = "gpt2"
model= pipeline(model=model_name)
model.save_pretrained("local_llms/gpt2")"""

from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", task="text2text-generation", pipeline_kwargs={"max_new_tokens": 10}, device_map = 'auto', batch_size=4)

# %% [markdown]
# ## Implementation of RAG pipeline

# %% [markdown]
# Simple paragraph splitter

# %%
import string

def retrieve_wiki_headers_and_paragraphs(context, langchain=False):
  data = context.split("\n\n")
  current_header = "General"

  results = []

  for part in data:
    # rule of thumb for detecting headers
    if part[:-1] not in string.punctuation and len(part.split()) < 10:
      current_header = part
    else:
      results.append((current_header, part))

  if results == []:
    return [context]
  elif not langchain:
    return results
  else:
    return [item[0] + " - " + item[1] for item in results]

# %%
# For reference
#print(validation[0]["context"])

# %% [markdown]
# Currently most basic version:
# - Use Splitter to divide text into paragraphs
# - Create Vectorstore with HuggingFaceEmbeddings
# - Retrieve most similar chunk for the respective prompt
# - Send prompt to specified LLM and print response
# - Recently added: batches for more efficiency

# %%
# Batch pipeline
def format_retrieval(docs):
    par = docs[0].page_content
    return par

def build_retriever(context):
    paragraphs = retrieve_wiki_headers_and_paragraphs(context, langchain=True)
    vectorstore = FAISS.from_texts(texts=paragraphs, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, return_parents=False)

    return retriever

def prepare_rag_chain_data(items):
    questions = [item["Question"] for item in items]
    retrievers = [build_retriever(build_context(item, domain)) for item in items]
    
    inputs = []
    for i, question in enumerate(questions):
      item = {"question": question, "context": format_retrieval(retrievers[i].get_relevant_documents(question))}
      inputs.append(item)

    return inputs

prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
  RunnablePassthrough()
  | prompt
  | llm.bind(stop=["\n\n"])
)


def batch_prediction(questions):
    inputs = prepare_rag_chain_data(questions)
    answers = rag_chain.batch(inputs)

    for i, input in enumerate(inputs):
       input["answer"] = answers[i]

    return inputs

# %% [markdown]
# ### Collect results for validation and test data set

# %%
import os
def save_file(data, write_path, filename):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    with open(write_path + "/{}.json".format(filename), "w") as f:
        json.dump(data, f)

# %%
from tqdm import tqdm

def evaluate_model(model_name, batch_size = 1):
    # Validation
    context_results = {}
    answers = {}

    def batch(iterable, n=1):
      l = len(iterable)
      for ndx in range(0, l, n):
          yield iterable[ndx:min(ndx + n, l)]

    for item in tqdm(batch(data_splits["validation"], batch_size), desc="Validation Progress"):
        #print(item)
        results = batch_prediction(item)
        for i, prediction in enumerate(results):
          print(prediction)
          qid = item[i]["QuestionId"]
          context_results[qid] = prediction["context"]
          answers[qid] = prediction["answer"]

    save_file(context_results, "../results/rag/"+model_name+"/wiki", "validation_context")
    save_file(answers, "../results/rag/"+model_name+"/wiki", "validation_answers")

# %%
evaluate_model(args.variant, batch_size=args.batch_size)

