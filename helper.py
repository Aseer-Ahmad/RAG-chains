from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings.anyscale import AnyscaleEmbeddings
import numpy as np
import bs4
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def load_doc():
    # Load Documents example 
    LINK = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    
    loader = WebBaseLoader(
        web_paths=(LINK,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load()
    
    print(f"docs loaded from {LINK}")
    print(f"doc character length : {len(str(docs))}")
    

    return docs

def getGPT4AllEmbeddings():
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embeddings = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )
    return embeddings

def get_embeddings(embd_model, question, document):
    query_result = embd_model.embed_query(question)
    document_result = embd_model.embed_query(document)

    print(f"length of query : {len(query_result)}")
    print(f"length of doc   : {len(document_result)}")

    return query_result, document_result

def get_google_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    return llm

def getGoogleEmbeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    sim = dot_product / (norm_vec1 * norm_vec2)

    print(f"cosine similarity score : {sim}")

    return sim

def get_splits(blog_docs, chunk_size, chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap)

    splits = text_splitter.split_documents(blog_docs)
    print(f"document splitted using RecursiveCharacterTextSplitter with chunk size : {chunk_size} & chunk overlap : {chunk_overlap} ")
    print(f"total splits : {len(splits)}")
    print(f"example split :\n{splits[0]}")
    print(f"example split len : {len(str(splits[0]))}")
    
    return splits