from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from helper import get_google_llm
from langchain.load import dumps, loads
from retreival import get_retreiver
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough



def getRagChain():
    # one kind of prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)    
    retriever = get_retreiver({"k": 1})
    llm       = get_google_llm()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
