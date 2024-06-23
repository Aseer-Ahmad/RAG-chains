from langchain_community.vectorstores import Chroma
from indexing import vectorstore_indexing

def get_retreiver(kwargs):
    vectorstore = vectorstore_indexing()
    print("creating retreiver")
    retriever = vectorstore.as_retriever(search_kwargs=kwargs)

    return retriever


def retreive_doc(retriever):
    docs = retriever.get_relevant_documents("What is Task Decomposition?")
    print(f"documents retreived")
    print(docs)

    return docs

# retriever = get_retreiver()
# retreive_doc(retriever)