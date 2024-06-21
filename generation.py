from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retreival import get_retreiver
from helper import get_google_llm

def generate(question):
    rag_chain = create_rag_chain()
    rag_chain.invoke(str(question))
