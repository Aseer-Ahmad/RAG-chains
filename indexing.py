from env import setup_env
from langchain_community.vectorstores import Chroma
from helper import load_doc, get_splits, getGPT4AllEmbeddings

def vectorstore_indexing():

    setup_env()

    docs = load_doc()
    splits = get_splits(docs, 300, 50)
    embd = getGPT4AllEmbeddings()
    #using Chroma vector store here for indexing doc splits by openAI embeddings 
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=embd)
    return vectorstore

