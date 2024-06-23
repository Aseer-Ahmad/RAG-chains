from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from helper import get_google_llm
from langchain.load import dumps, loads
from retreival import get_retreiver
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))

    return [loads(doc) for doc in unique_docs]

def getMultiQueryRagChain():

    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)
    llm = get_google_llm()
    retriever = get_retreiver({"k": 1})

    print("creating multi query retreival chain")
    generate_queries = (
        prompt_perspectives 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    # docs = retrieval_chain.invoke({"question":question})

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    print("creating final rag chain")

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    # final_rag_chain.invoke({"question":question})
    return final_rag_chain, retrieval_chain