import string

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TextRankingTransformersPreprocessor
from modelscope.utils.constant import Tasks

import spacy

# Load the spaCy English model (you may need to download it first)
nlp = spacy.load("en_core_web_sm")

# The implementation needs to return the string of the most relevant paragraph (with generate_data=False) or
# a dict with the keys "context_ids" of the most relevant k paragraphs and "contexts_dict" with all paragraphs 

class Retriever():
    def retrieve(self, question, context):
        pars = self.retrieve_wiki_headers_and_paragraphs(context)
        return self.retrieval_funcs[self.type](self, question, pars)

    # Save provided configuration
    def __init__(self, type = None, embeddings_id=None, max_len=10000, headers=False, topx=1, thresh=0.5) -> None:
        if type in self.retrieval_funcs.keys():
            self.type = type
        else:
            raise ValueError("This retriever key does not exist!")
        
        if max_len == 10000 and topx != 1:
            raise ValueError("To use topx_contexts it is required to adapt max_par_len to stay under the max token length of the LLM (512 for flan-t5)!")
        
        self.topx = topx
        self.thresh = thresh

        if self.type == "hlatr":
            model_id = 'damo/nlp_corom_passage-ranking_english-base'
            model = Model.from_pretrained(model_id)
            preprocessor = TextRankingTransformersPreprocessor(model.model_dir)
            self.hlatr_pipeline = pipeline(task=Tasks.text_ranking, model=model, preprocessor=preprocessor)
        
        if not embeddings_id or embeddings_id=="bm25":
            self.embeddings = None
        elif embeddings_id:
            self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_id)

        self.max_len = max_len
        self.headers = headers
            
    # Define function that retrieves wiki paragraphs from the given context, keeping whole sentences and respecting the max_len
    def retrieve_wiki_headers_and_paragraphs(self, context):
        data = context.split("\n\n")
        current_header = "General"

        results = []

        for part in data:
            # rule of thumb for detecting headers
            if part[:-1] not in string.punctuation and len(part.split()) < 10:
                current_header = part
            else:
                if self.headers:
                    part = current_header + " - " + part
                    
                # Respect sentences with spacy
                doc = nlp(part)

                current_subpar = ""
                current_length = 0

                for sentence in [sent.text for sent in doc.sents]:
                    token_length = len(sentence.split())

                    if current_length + token_length <= self.max_len:
                        current_subpar += " " + sentence
                        current_length += token_length
                    else:
                        # If the current subpar exceeds the max length, split into subparts
                        if current_length > 0:
                            if self.headers:
                                current_subpar = current_header + " - " + current_subpar
                            results.append(current_subpar.strip())
                            current_subpar = sentence
                            current_length = token_length
                        else:
                            results.append(sentence)

                if current_subpar:
                    results.append(current_subpar.strip())

        if results == []:
            return [context]
        else:
            return results

    # Format obtained retrieval into single context  
    def format_retrieval(self, docs, topx):
            if topx == 1:
                par = docs[0].page_content
            else:
                par = ""
                for i in range(topx):
                    par += docs[i].page_content

            return par
    
    # Langchain vectorstore retrieval with FAISS that can use any huggingface embedding
    def langchain_vectorstore(self, question, paragraphs):
        vectorstore = FAISS.from_texts(texts=paragraphs, embedding=self.embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.topx}, return_parents=False)
        return self.format_retrieval(retriever.get_relevant_documents(question, k=self.topx), self.topx)
            

    # Retrieve with HLATR from modelscope
    def hlatr_retrieval(self, question, paragraphs):

        input = { 
            'source_sentence': [question],
            'sentences_to_compare': paragraphs
        }

        result = self.hlatr_pipeline(input=input)

        if self.topx == 1:
            max_index = result["scores"].index(max(result["scores"]))
            return paragraphs[max_index]
        else:
            scores = result["scores"]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.topx]

            concatenated_context = ""
            for i, idx in enumerate(top_indices):
                if i != 0:
                    if scores[idx] < scores[top_indices[0]]*self.thresh:
                        break
                concatenated_context += paragraphs[idx]

            return concatenated_context
        
    # Retrieve with BM25
    def bm25_retrieval(self, question, paragraphs):
        retriever = BM25Retriever.from_texts(texts=paragraphs, k=self.topx)
        return self.format_retrieval(retriever.get_relevant_documents(question), self.topx)

    # Map keywords to functions
    retrieval_funcs = {
        "langchain-vs": langchain_vectorstore,
        "hlatr": hlatr_retrieval,
        "bm25": bm25_retrieval
    }