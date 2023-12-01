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

# %% [markdown]
# ### Load eval data

# %%
import json
import sys

sys.path.append("..")

read_files = ["test_Wikipedia.json", "validation_Wikipedia.json"]

from data_preprocessing.preprocessing import create_splits

data_splits = create_splits(create_eval = False)

def read_file(path):
    with open("../eval_splits/" + path) as f:
        data = json.load(f)
        return data

test = read_file(read_files[0])
validation = read_file(read_files[1])

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
llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", task="text2text-generation", pipeline_kwargs={"max_new_tokens": 10}, device=0)

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
# 
# -> Can and should be optimized performancewise!

# %%
def rag_answer(question, context, log=False):
    #splitter = RecursiveCharacterTextSplitter(
    #  chunk_size=200, chunk_overlap=0, add_start_index=False
    #)
    par = ""

    paragraphs = retrieve_wiki_headers_and_paragraphs(context, langchain=True)
    vectorstore = FAISS.from_texts(texts=paragraphs, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, return_parents=False)
    
    prompt = hub.pull("rlm/rag-prompt")

    def format_retrieval(docs):
      nonlocal par
      par = docs[0].page_content
      return par

    rag_chain = (
        {"context": retriever | format_retrieval, "question": RunnablePassthrough()}
        | prompt
        | llm
        #| StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    #for chunk in rag_chain.stream(question):
      #print(chunk, end="", flush=True)
      #answer.append(chunk)
    return {
       "context": par,
       "answer": answer
    }


# %% [markdown]
# Initial test of pipeline

# %%
def build_context(item):
    texts = []
    for text in item["entity_pages"]["wiki_context"]:
      texts.append(text)

    context = " ".join(texts)

    return context

# %%
def run_prediction(data, log=False):
    if log: print("Question:", data["Question"])
    prediction = rag_answer(data["question"], build_context(data), log=log)
    if log: print("\nCorrect answer:", data["Answer"])
    return prediction

#run_prediction(validation["Data"][5], log=True)

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

def evaluate_model(model_name):
    # Validation
    context_results = {}
    answers = {}

    for item in tqdm(data_splits["validation"], desc="Validation Progress"):
        prediction = run_prediction(item)
        print(prediction["answer"])
        qid = item["question_id"]
        context_results[qid] = prediction["context"]
        answers[qid] = prediction["answer"]

    save_file(context_results, "../results/rag/"+model_name+"/wiki", "validation_context")
    save_file(answers, "../results/rag/"+model_name+"/wiki", "validation_answers")
    

# %%

import torch

def run_predictions_batch_parallel(data_batch, log=False):
    predictions = []
    for data in data_batch:
        if log:
            print("Question:", data["question"])
        prediction = rag_answer(data["question"], build_context(data), log=log)
        if log:
            print("\nCorrect answer:", data["answer"])
        predictions.append(prediction)
    return predictions

def run_predictions_batch_parallel_wrapper(data, log=False):
    return run_predictions_batch_parallel(*data, log=log)

def parallel_run(model_name):
    context_results = {}
    answers = {}
    # Example of batching for validation data
    batch_size = 5  # You can adjust the batch size as needed
    validation_data = data_splits["validation"]

    # Split the validation data into batches
    num_batches = len(validation_data) // batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = validation_data[start_idx:end_idx]

        # Run predictions for the batch in parallel
        with torch.no_grad():
            predictions = torch.nn.parallel.parallel_apply(
                run_predictions_batch_parallel_wrapper,
                [(batch_data, False)] * len(batch_data),
                devices=[device] * len(batch_data)
            )

        # Process predictions as needed
        for idx, prediction in enumerate(predictions):
            qid = batch_data[idx]["question_id"]
            print(prediction["answer"])
            context_results[qid] = prediction["context"]
            answers[qid] = prediction["answer"]

    # Handle the last batch (which may have a size less than batch_size)
    last_batch_data = validation_data[num_batches * batch_size:]
    with torch.no_grad():
        last_batch_predictions = torch.nn.parallel.parallel_apply(
            run_predictions_batch_parallel,
            [(last_batch_data, False)] * len(last_batch_data),
            devices=[device] * len(last_batch_data)
        )
    # Process the results for the last batch as needed
    for idx, prediction in enumerate(last_batch_predictions):
            qid = last_batch_data[idx]["question_id"]
            print(prediction["answer"])
            context_results[qid] = prediction["context"]
            answers[qid] = prediction["answer"]

    save_file(context_results, "../results/rag/"+model_name+"/wiki", "validation_context")
    save_file(answers, "../results/rag/"+model_name+"/wiki", "validation_answers")

# %%
parallel_run("baseline")

# %%
item = data_splits["validation"][1108]
print(item["question"])
print(build_context(item))
print(retrieve_wiki_headers_and_paragraphs(build_context(item)))
print(item["entity_pages"]["wiki_context"])
print(item["entity_pages"]["filename"])
run_prediction(item)


