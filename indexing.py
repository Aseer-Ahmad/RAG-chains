from env import setup_env
from langchain_community.vectorstores import Chroma
from helper import load_doc, get_splits, getGPT4AllEmbeddings, getGoogleEmbeddings

def vectorstore_indexing():

    docs = load_doc()
    splits = get_splits(docs, 300, 50)
    embd = getGoogleEmbeddings()
    #using Chroma vector store here for indexing doc splits by openAI embeddings 
    print("beginning indexing")
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=embd)
    return vectorstore

