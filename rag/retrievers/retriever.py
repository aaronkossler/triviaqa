# implement different retrieval strategies for rag
import string
from nltk.tokenize import RegexpTokenizer

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TextRankingTransformersPreprocessor
from modelscope.utils.constant import Tasks

# The implementation needs to return the string of the most relevant paragraph (with generate_data=False) or
# a dict with the keys "context_ids" of the most relevant k paragraphs and "contexts_dict" with all paragraphs 

class Retriever():
    def retrieve(self, question, context):
        pars = self.retrieve_wiki_headers_and_paragraphs(context, self.headers, self.max_len)
        return self.retrieval_funcs[self.type](self, question, pars)

    def __init__(self, type = None, embeddings_id=None, max_len=10000, headers=False) -> None:
        if type in self.retrieval_funcs.keys():
            self.type = type
        else:
            raise ValueError("This retriever key does not exist!")
        
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
            

    def retrieve_wiki_headers_and_paragraphs(self, context, headings=True, max_par_length=100):
        data = context.split("\n\n")
        current_header = "General"

        results = []

        # Create a RegexpTokenizer
        tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

        for part in data:
            # rule of thumb for detecting headers
            if part[:-1] not in string.punctuation and len(part.split()) < 10:
                current_header = part
            else:
                if headings:
                    part = current_header + " - " + part

                # Tokenize the paragraph
                tokens = tokenizer.tokenize(part)

                current_subpar = ""
                current_length = 0

                for token in tokens:
                    token_length = len(token.split())

                    if current_length + token_length <= max_par_length:
                        current_subpar += " " + token
                        current_length += token_length
                    else:
                        # If the current subpar exceeds the max length, split into subparts
                        if current_length > 0:
                            if headings:
                                current_subpar = current_header + " - " + current_subpar
                            results.append(current_subpar.strip())
                            current_subpar = token
                            current_length = token_length
                        else:
                            results.append(token)

                if current_subpar:
                    results.append(current_subpar.strip())

        if results == []:
            return [context]
        else:
            # print(results)
            return results
    
    # Langchain vectorstore retrieval with FAISS that can use any huggingface embedding
    def langchain_vectorstore(self, question, paragraphs):
        # get text from retrieved context
        def format_retrieval(docs):
            par = docs[0].page_content
            return par

        if self.embeddings:
            vectorstore = FAISS.from_texts(texts=paragraphs, embedding=self.embeddings)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, return_parents=False)
        else:
            retriever = BM25Retriever.from_texts(texts=paragraphs)

        return format_retrieval(retriever.get_relevant_documents(question))
    
    def hlatr_retrieval(self, question, paragraphs):

        input = { 
            'source_sentence': [question],
            'sentences_to_compare': paragraphs
        }

        result = self.hlatr_pipeline(input=input)
        max_index = result["scores"].index(max(result["scores"]))
        #print (paragraphs[max_index])
        return paragraphs[max_index]

    # Map keywords to functions
    retrieval_funcs = {
        "langchain-vs": langchain_vectorstore,
        "hlatr": hlatr_retrieval
    }

# Additional class solely for data generation to train an own retriever
class DataGenRetriever(Retriever):
    # The implementation needs to return a dict with the keys "context_ids" of the most relevant k paragraphs and "contexts_dict" with all paragraphs 
    def retrieve(self, question, context, answer):
        retrieval = self.retrieve_wiki_headers_and_paragraphs(context, answer)
        return self.hlatr_retrieval(question, retrieval)
    
    def __init__(self, topk):
        model_id = 'damo/nlp_corom_passage-ranking_english-base'
        model = Model.from_pretrained(model_id)
        preprocessor = TextRankingTransformersPreprocessor(model.model_dir)
        self.hlatr_pipeline = pipeline(task=Tasks.text_ranking, model=model, preprocessor=preprocessor)

        self.topk = topk

     # Basic paragraph splitter
    def retrieve_wiki_headers_and_paragraphs(self, context, answer, max_len = 100):
        data = context.split("\n\n")

        pars = []

        idx = 0
        for part in data:
            # rule of thumb for detecting headers
            if part[:-1] not in string.punctuation and len(part.split()) < 10:
                pass
            else:
                pars.append(part)
                idx += 1

        # extract candidate paragraphs that might contain the relevant information
        candidate_ids = []
        for idx, par in enumerate(pars):
            if any(el.lower() in par.lower() for el in answer["Aliases"]+answer["NormalizedAliases"]):
                candidate_ids.append(idx)
        if candidate_ids == []:
            print("PROBLEM", answer, pars)
            candidate_ids = list(range(len(pars)))

        results = {
            "paragraphs": pars,
            "candidate_ids": candidate_ids
        }
        return results
    
    def hlatr_retrieval(self, question, context):
        paragraphs = [context["paragraphs"][idx] for idx in context["candidate_ids"]]

        input = { 
            'source_sentence': [question],
            'sentences_to_compare': paragraphs
        }

        result = self.hlatr_pipeline(input=input)
        scores = [(context["candidate_ids"][idx], score) for idx, score in enumerate(result["scores"])]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        max = len(sorted_scores)
        if self.topk < max: max = self.topk
        ranking = [val[0] for val in sorted_scores[:max]]

        context["ranking"] = ranking
        #print(len(ranking))
        return context
        #max_index = result["scores"].index(max(result["scores"]))
        #return paragraphs[max_index]