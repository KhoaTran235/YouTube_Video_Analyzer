import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from rag_pipeline.prompt import RAG_PROMPT


def get_session_retriever(top_k=6):
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return None

    return vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )


def get_session_rag_chain(video_info):
    retriever = get_session_retriever()
    if retriever is None:
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT.partial(
            title=video_info["title"],
            description=video_info["description"],
        )
        | llm
    )


    return chain
