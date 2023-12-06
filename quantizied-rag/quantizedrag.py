import sys, string, json, os
from langchain import LlamaCpp
from tqdm.notebook import tqdm
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

from data_preprocessing.preprocessing import create_splits

sys.path.append("..")
read_files = ["test_Wikipedia.json", "validation_Wikipedia.json"]
data_splits = create_splits(create_eval = False)

model = "/Users/kon/Downloads/orca-2-7b.Q4_0.gguf"
llm = load_qa_chain(LlamaCpp(model_path=model), chain_type="stuff")

def read_file(path):
    with open("../evaluate_models/" + path) as f:
        data = json.load(f)
        return data

test = read_file(read_files[0])
validation = read_file(read_files[1])

def save_file(data, write_path, filename):
    os.makedirs(write_path, exist_ok=True)
    with open(write_path + "/{}.json".format(filename), "w") as f:
        json.dump(data, f)


def retrieve_wiki_headers_and_paragraphs(context, langchain=False):
  data = context.split("\n\n")
  current_header = "General"
  results = []

  for part in data:
    if part[:-1] not in string.punctuation and len(part.split()) < 10:
      current_header = part
    else:
      results.append((current_header, part))

  if not langchain:
    return results
  else:
    return [item[0] + " - " + item[1] for item in results]


def rag_answer(question, context, log=False):
    paragraphs = retrieve_wiki_headers_and_paragraphs(context, langchain=True)
    vectorstore = FAISS.from_texts(texts=paragraphs, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, return_parents=False)
    par = retriever.get_relevant_documents(question)

    answer = llm.run(input_documents=par,
                     question=f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one to two words or numbers maximum and keep the answer concise. question: {question}");

    return {
        "context": par[0].page_content,
        "answer": answer
    }


def build_context(item):
    texts = []
    for text in item["entity_pages"]["wiki_context"]:
      texts.append(text)

    context = " ".join(texts)
    return context


def run_prediction(data, log=False):
    prediction = rag_answer(data["question"], build_context(data), log=log)
    return prediction


def evaluate_model(model_name, only_for: list = None):
    context_results = {}
    answers = {}
    failed = []

    trail = ('_' + '+'.join(only_for)) if only_for is not None else ''

    for item in tqdm(data_splits["validation"], desc="Validation Progress"):
        qid = item["question_id"]
        if only_for is not None and qid not in only_for:
            continue
        try:
            prediction = run_prediction(item)
            print(f"id: {qid} question: {item['question']} prediction answer: {prediction['answer']}")
            context_results[qid] = prediction["context"]
            answers[qid] = prediction["answer"]
            print("##########################################################################")
        except KeyboardInterrupt as error:
            save_file(context_results, "results/rag/" + model_name + "/wiki", "validation_context" + trail)
            save_file(answers, "results/rag/" + model_name + "/wiki", "validation_answers" + trail)
            print("saved")
            raise error
        except Exception as error:
            print(f"Failure for question {qid} ({type(error).__name__}: {error})")
            failed.append(qid)
    print(f"FAILED: {failed}")

    save_file(context_results, "results/rag/" + model_name + "/wiki", "validation_context" + trail)
    save_file(answers, "results/rag/" + model_name + "/wiki", "validation_answers" + trail)

if __name__ == '__main__':
    evaluate_model("orca-2-7b")




