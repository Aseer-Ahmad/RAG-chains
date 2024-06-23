from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retreival import get_retreiver
from helper import get_google_llm
from env import setup_env
from query_transformations import answerindvidually, decomposition, multiquery, ragfusion, simplerag, stepback


def generate():
    
    question = "What is task decomposition for LLM agents?"
    ragchain, retrieval_chain = multiquery.getMultiQueryRagChain()
    
    print("multi query retreival chain invoked")
    retrieval_chain.invoke({"question":question})
    print("final rag chain invoked")
    ragchain.invoke({"question":question})


def router():
    pass

if __name__ == '__main__':
    setup_env()
    generate()