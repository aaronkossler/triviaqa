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
   "--model",
   default="google/flan-t5-base",
   help="Specify the model that should be applied for answer generation."
)

args = parser.parse_args()

# %% [markdown]
# ### Load eval data

# %%
import json
import sys
#sys.path.append("../")
sys.path.append("../..")

from data_preprocessing.preprocessing import create_splits, build_context

domain = "wikipedia"

data_splits = create_splits(as_list_of_dicts=True, domain=domain)

# %% [markdown]
# Import relevant modules for langchain

# %%
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough


# %% [markdown]

from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(model_id=args.model, task="text2text-generation", pipeline_kwargs={"max_new_tokens": 10}, device_map="auto", batch_size=int(args.batch_size))


# %%
# Build retriever with given information
from retrievers.retriever import DataGenRetriever
retriever = DataGenRetriever(topk=10)

# %% [markdown]
# ## Implementation of RAG pipeline

# process batch of items to be prapared for batch prediction
def prepare_data_chain_data(items):
    #questions = [item["Question"] for item in items]
    contexts = [build_context(item, domain) for item in items]
    
    inputs = []
    for idx, question in enumerate(items):
      item = retriever.retrieve(question["Question"], contexts[idx], question["Answer"]["NormalizedAliases"])
      inputs.append(item)

    return inputs

from langchain.prompts import PromptTemplate

context_feedback_prompt = PromptTemplate.from_template("""\
    Human: You are an assistant for question-answering tasks. Does the provided context provide the necessary information to answer the given question? Please answer with 'Yes' or 'No'. 
    Question: {question} 
    Context: {context}
    Answer:
""")

data_chain = (
   RunnablePassthrough()
   | context_feedback_prompt
   | llm.bind(stop=["\n\n"])
)

# execute prediction for a batch of questions
def batch_prediction(questions):
    context_data = prepare_data_chain_data(questions)
    
    done = False
    status_dict = {idx: [0, False] for idx in range(len(questions))}

    sol_par_idxs = [-1]*len(questions)
    while not done:
        #print(questions[0]["Question"])
        #print(context_data["paragraphs"][context_data[idx]["ranking"][status_dict[idx][0]]])
        inputs = [{"question": question["Question"], "context": context_data[idx]["paragraphs"][context_data[idx]["ranking"][status_dict[idx][0]]]} for idx, question in enumerate(questions) if not status_dict[idx][1]]
        #print(inputs)
        answers = data_chain.batch(inputs)

        done = True
        for idx, answer in enumerate(answers):
            if answer.lower() == "yes":
                status_dict[idx][1] = True
                sol_par_idxs[idx] = context_data[idx]["ranking"][status_dict[idx][0]]#status_dict[idx][0]
                #print("Best candidate:", questions[idx]["Question"], context_data[idx]["paragraphs"][sol_par_idxs[idx]])
            else:
                if status_dict[idx][0] == len(context_data[idx]["ranking"]) - 1:
                    status_dict[idx][1] = True
                    sol_par_idxs[idx] = context_data[idx]["ranking"][0]
                else:
                    status_dict[idx][0] += 1
                    done = False

    results = []
    for idx, question in enumerate(questions):
        candidate = context_data[idx]
        candidate["best_paragraph"] = sol_par_idxs[idx]
        #del candidate["ranking"]
        q_data = {
            "QuestionId": question["QuestionId"],
            "Question": question["Question"],
            "RetrieverData": candidate

        }
        results.append(q_data)

    return results

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

def rag_prediction(model_name, batch_size):
    for key in data_splits.keys():
        data = data_splits[key][:20]
        
        results = []

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        progress_bar = tqdm(total=math.ceil(len(data)/batch_size), desc="{} Progress".format(key), unit="batch")

        for item in batch(data, batch_size):
            #print(item)
            answers = batch_prediction(item)
            """
            for i, prediction in enumerate(answers):
                print(prediction)
                qid = item[i]["QuestionId"]
                #results[qid] = prediction
                results = results + prediction"""
            results += answers
            progress_bar.update(1)

        save_file(results, "./data/"+model_name+"/", "{}_{}".format("wiki", key))

        #eval_format = {key: inner_dict["answer"] for key, inner_dict in results.items()}
        #save_file(eval_format, "./results/"+model_name+"/", "{}_{}_results".format("wiki", type))


# %%
# start predictions with specified cli params
rag_prediction(args.variant, int(args.batch_size))
