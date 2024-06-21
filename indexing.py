from env import setup_env
from langchain_community.vectorstores import Chroma
from helper import load_doc, get_splits, get_embedding_model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings.anyscale import AnyscaleEmbeddings

#langchain has bunch of indexing, embedding, document loader integrations

def vectorstore_indexing():

    setup_env()

    docs = load_doc()
    splits = get_splits(docs, 300, 50)
    embd = get_embedding_model()
    #using Chroma vector store here for indexing doc splits by openAI embeddings 
    vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embd)
    return vectorstore

