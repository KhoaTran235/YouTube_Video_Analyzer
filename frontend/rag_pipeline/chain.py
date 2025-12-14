import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
# from langchain.memory import ConversationBufferMemory
# from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_classic.memory import ConversationSummaryMemory
from rag_pipeline.prompt import RAG_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()

def get_session_retriever(top_k=6):
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return None

    return vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )

def get_session_memory():
    if "rag_memory" not in st.session_state:
        llm = ChatGoogleGenerativeAI(
            api_key=os.environ["GOOGLE_API_KEY"],
            model="gemini-2.5-flash",
            temperature=0.2,
        )
        st.session_state.rag_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",   # tên biến đưa vào prompt
            return_messages=False        # dùng summary text
        )
    return st.session_state.rag_memory


def get_session_rag_chain(video_info):
    retriever = get_session_retriever()
    if retriever is None:
        return None

    llm = ChatGoogleGenerativeAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-2.0-flash",
        temperature=0.2,
    )

    memory = get_session_memory()

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
        }
        | RAG_PROMPT.partial(
            title=video_info["title"],
            description=video_info["description"],
        )
        | llm
    )


    return chain
