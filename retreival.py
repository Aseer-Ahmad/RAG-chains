from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from indexing import vectorstore_indexing

def get_retreiver():
    vectorstore = vectorstore_indexing()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    return retriever


def retreive_doc(retriever):
    docs = retriever.get_relevant_documents("What is Task Decomposition?")
    print(f"documents retreived")
    print(docs)
     
    return docs