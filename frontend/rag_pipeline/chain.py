import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain.memory import ConversationBufferMemory
# from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_classic.memory import ConversationSummaryMemory
from rag_pipeline.prompt import RAG_PROMPT, DIRECT_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()


def get_session_retrievers(top_k=4):
    comment_vs = st.session_state.get("comment_vectorstore")
    transcript_vs = st.session_state.get("transcript_vectorstore")

    if comment_vs is None or transcript_vs is None:
        return None, None

    comment_retriever = comment_vs.as_retriever(
        search_kwargs={"k": top_k}
    )

    transcript_retriever = transcript_vs.as_retriever(
        search_kwargs={"k": top_k}
    )

    return transcript_retriever, comment_retriever

def get_session_memory():
    if "rag_memory" not in st.session_state:
        llm = ChatGoogleGenerativeAI(
            api_key=os.environ["GOOGLE_API_KEY"],
            model="gemini-2.5-flash",
            temperature=0.2,
        )
        st.session_state.rag_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=False
        )
    return st.session_state.rag_memory


def get_session_rag_chain(video_info, video_summary=None):
    transcript_retriever, comment_retriever = get_session_retrievers()
    if transcript_retriever is None:
        return None
    if comment_retriever is None:
        return None

    llm = ChatGoogleGenerativeAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-2.0-flash",
        temperature=0.2,
    )

    memory = get_session_memory()

    summary_text = (
        video_summary
        if video_summary is not None
        else "NOT_PROVIDED"
    )

    chain = (
        {
            "transcript_context": transcript_retriever,
            "comment_context": comment_retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
        }
        | RAG_PROMPT.partial(
            title=video_info["title"],
            description=video_info["description"],
            video_summary=summary_text
        )
        | llm
    )


    return chain

def get_session_direct_chain(video_info, video_summary=None):
    llm = ChatGoogleGenerativeAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        model="gemini-2.0-flash",
        temperature=0.2,
    )

    memory = get_session_memory()

    summary_text = (
        video_summary
        if video_summary is not None
        else "NOT_PROVIDED"
    )

    chain = (
        {
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
        }
        | DIRECT_PROMPT.partial(
            title=video_info["title"],
            description=video_info["description"],
            video_summary=summary_text
        )
        | llm
    )


    return chain
