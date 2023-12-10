# implement different retrieval strategies for rag
import string

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

# The implementation needs to return the string of the relevant paragraph

class Retriever():
    def retrieve(self, question, context):
        return self.retrieval_funcs[self.type](self, question, context)

    def __init__(self, type = None, embeddings_id=None) -> None:
        if type in self.retrieval_funcs.keys():
            self.type = type
        else:
            raise ValueError("This retriever key does not exist!")
        
        if not embeddings_id or embeddings_id=="bm25":
            self.embeddings = None
        elif embeddings_id:
            self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_id)
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            #self.embeddings.to(device)
            

    # Basic paragraph splitter
    def retrieve_wiki_headers_and_paragraphs(self, context, headings=False):
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
        elif not headings:
            return [item[1] for item in results]
        else:
            return [item[0] + " - " + item[1] for item in results]
        

    # Langchain vectorstore retrieval with FAISS that can use any huggingface embedding
    def langchain_vectorstore(self, question, context):
        # get text from retrieved context
        def format_retrieval(docs):
            par = docs[0].page_content
            return par

        # build retriever
        paragraphs = self.retrieve_wiki_headers_and_paragraphs(context, False)
        if self.embeddings:
            vectorstore = FAISS.from_texts(texts=paragraphs, embedding= self.embeddings)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, return_parents=False)
        else:
            retriever = BM25Retriever.from_texts(texts=paragraphs)

        return format_retrieval(retriever.get_relevant_documents(question))

    # Map keywords to functions
    retrieval_funcs = {
        "langchain-vs": langchain_vectorstore
    }