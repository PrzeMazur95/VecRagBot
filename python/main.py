import os

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA as VectorDBQA
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Any

load_dotenv()

OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
FILENAME = 'article.txt' #name of the file to be processed
LLM_MODEL = OpenAI(openai_api_key=OPEN_AI_API_KEY) # LLM model to interact with our context

def get_embedding_model()->Any:
    """
    Returns embedding model, that we will use to embed data from a file to Qdrant
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_data_from_file()->list:
    """
    Open the file, and assing each line to an list element. 
    Line.strip() form remove trailing whitespaces
    """
    with open(FILENAME, encoding="utf-8") as file:
        text = [line.strip() for line in file]
    return text

def get_qdrant_vectorstore_with_uploaded_data()->Any:
    """
    Return Qdrant instance with uploaded data taken from the file
    """
    feeded_qdrant_client = Qdrant.from_texts(
        get_data_from_file(),
        get_embedding_model(),
        url=QDRANT_URL,
        collection_name="article"
    )
    return feeded_qdrant_client

def get_retriever_context()->Any:
    """
    Assign context with created embeddings from .txt file
    """
    return VectorStoreRetriever(vectorstore=get_qdrant_vectorstore_with_uploaded_data())

def get_retrieval_qa_context()->Any:
    """
    Combine llm with out context to create a QA system based on our data
    """
    return VectorDBQA.from_llm(llm=LLM_MODEL, retriever=get_retriever_context())

def main():
    """
    Triggers main process initialization
    """
    retrieval_qa_context = get_retrieval_qa_context()

    while True:
        question = input("What is your question?: ")
        print()
        print(retrieval_qa_context.invoke(question)['result'])

        keep_asking = input("Do you have another question?[Y/n]:  ")

        if not keep_asking == "Y":
            print("See you again! ")
            break

if __name__ == "__main__":
    main()
