from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retreival import get_retreiver
from helper import get_google_llm

def create_prompt_template():
    # one kind of prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return prompt


def create_rag_chain():
    
    retriever = get_retreiver()
    llm       = get_google_llm()
    prompt    = create_prompt_template()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def use_rag_chain(question):
    rag_chain = create_rag_chain()
    rag_chain.invoke(str(question))
