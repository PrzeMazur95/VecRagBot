import pickle

from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from app.service.file_service import FileService
from app.service.open_ai_service import OpenAiService
from flask import session, jsonify
from app.models.qdrant_connector import QdrantConnection

class Retriever():
    
    def __init__(self):
        self.FILE_SERVICE = FileService(),
        # self.QDRANT_CONNECTION = QdrantConnection(),
        self.OPEN_AI_SERVICE = OpenAiService()
        
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        session_data = session.get(session_id)
        if session_data:
            return pickle.loads(session_data)
        else:
            history = ChatMessageHistory()
            session[session_id] = pickle.dumps(history)
            return history

    def save_session_history(self, session_id: str, history: BaseChatMessageHistory):
        session[session_id] = pickle.dumps(history)
    
    def main(self, filename: str, query: str):
        
        history_session_name = filename + session['username'] + "_collection_name"
        
        QdrantConnector = QdrantConnection()
        
        client = QdrantConnector.client
        
        embedding_model=self.OPEN_AI_SERVICE.get_embeddings_model()
        
        VectorDB = Qdrant(client, collection_name=session[session['username'] + "_collection_name"], embeddings=embedding_model)
        
        retriever = VectorDB.as_retriever()
        
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=session[session['username'] + 'api_key'])
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Retrieve session history
        chat_history = self.get_session_history(history_session_name)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        answer = conversational_rag_chain.invoke(
            {"input": query},
            config={
                "configurable": {"session_id": history_session_name}
            },
        )["answer"]

        # Save the updated history back to the session
        self.save_session_history(history_session_name, chat_history)

        # Print the entire session history for debugging
        for message in chat_history.messages:
            prefix = "AI" if isinstance(message, AIMessage) else "User"
            print(f"{prefix}: {message.content}\n")

        return answer
        
