# implement different retrieval strategies for rag
import string

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch

#from angle_emb import AnglE, Prompts

# The implementation needs to return the string of the relevant paragraph

class Retriever():
    def retrieve(self, question, context):
        return self.retrieval_funcs[self.type](self, question, context)

    def __init__(self, type = None, embeddings_id=None) -> None:
        if type in self.retrieval_funcs.keys():
            self.type = type
        else:
            raise ValueError("This retriever key does not exist!")
        
        if embeddings_id:
            self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_id)
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            #self.embeddings.to(device)
        else:
            self.embeddings = None

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
        vectorstore = FAISS.from_texts(texts=paragraphs, embedding= self.embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, return_parents=False)

        return format_retrieval(retriever.get_relevant_documents(question))
    

    # Similarity based: UAE-Large-V1
    def retrieve_uae_large(self, question, context):
        angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        angle.set_prompt(prompt=Prompts.C)
        vec = angle.encode({'text': 'hello world'}, to_numpy=True)
        print(vec)
        vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True)
        print(vecs)


    # Map keywords to functions
    retrieval_funcs = {
        "langchain-vs": langchain_vectorstore,
        "uae-large": retrieve_uae_large
    }