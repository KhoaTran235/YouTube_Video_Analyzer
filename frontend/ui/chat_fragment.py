import streamlit as st
from rag_pipeline.router import semantic_router
from rag_pipeline.chain import (
    get_session_rag_chain,
    get_session_direct_chain
)

@st.fragment
def render_chat(info):
    st.subheader("🤖 Ask AI about this video (Powered by Google Gemini)")

    if st.session_state.video_summary:
        st.success("Chatbot can answer based on video's comments, info, transcript, and AI-generated summary")
    else:
        st.info("Chatbot can only answer based on video's comments, info and transcript; video's AI-generated summary is not available")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ===== CHAT BOX =====
    chat_box = st.container(height=480)
    with chat_box:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["user"])
            with st.chat_message("assistant"):
                st.markdown(chat["assistant"])

    # ===== RESET =====
    if st.session_state.chat_history:
        if st.button("🗑 Reset conversation", key="reset_chat"):
            st.session_state.chat_history = []
            if "rag_memory" in st.session_state:
                del st.session_state.rag_memory
            st.rerun()

    # ===== INPUT =====
    user_query = st.chat_input(
        "Ask a question about comments, sentiment, or audience opinion",
        disabled=not st.session_state.analysis_done
    )

    if user_query:
        st.session_state.chat_history.append({
            "user": user_query,
            "assistant": "Thinking..."
        })
        st.session_state.pending_user_query = user_query
        st.rerun()

    # ===== PROCESS QUERY =====
    if st.session_state.pending_user_query:
        user_query = st.session_state.pending_user_query

        if st.session_state.embedder is None:
            st.error("Please analyze the video first.")
            st.session_state.pending_user_query = None
            st.stop()

        route = semantic_router(user_query, embedder=st.session_state.embedder)

        if route == "NO_RETRIEVAL":
            qa_chain = get_session_direct_chain(
                info,
                video_summary=st.session_state.video_summary
            )
        else:
            qa_chain = get_session_rag_chain(
                info,
                video_summary=st.session_state.video_summary
            )

        if qa_chain is None:
            answer = "Please analyze the video first."
        else:
            with st.spinner():
                result = qa_chain.invoke(user_query)
                answer = result.content

                st.session_state.rag_memory.save_context(
                    {"input": user_query},
                    {"output": answer}
                )

        st.session_state.chat_history[-1]["assistant"] = answer
        st.session_state.pending_user_query = None
        st.rerun()