import streamlit as st
from ui.chat_fragment import render_chat
from ui.video_info import render_video_info
from ui.sentiment_view import render_sentiment

import streamlit as st
import matplotlib.pyplot as plt
from utils import extract_video_id
from services.comment_sentiment import analyze_sentiment
from services.yt_service import get_video_info, get_video_comments, get_video_transcript
from services.video_summarize import summarize_video
from utils import merge_comments_with_sentiment, sentiment_statistics, merge_transcript_by_time
from rag_pipeline.build_vectorstore import build_comment_vectorstore, build_transcript_vectorstore
from rag_pipeline.chain import get_session_rag_chain, get_session_direct_chain
from rag_pipeline.router import semantic_router
from rag_pipeline.gemini_embedding import GeminiEmbedding

# =========================
# Define cached functions
# =========================
@st.cache_data(show_spinner=False)
def cached_get_video_comments(video_id, max_results):
    return get_video_comments(video_id, max_results)

@st.cache_data(show_spinner=False)
def cached_analyze_sentiment(texts):
    return analyze_sentiment(texts)

@st.cache_data(show_spinner=False)
def cached_get_video_transcript(video_id):
    return get_video_transcript(video_id)

@st.cache_data(show_spinner=False)
def cached_summarize_video(video_url):
    return summarize_video(video_url)

# =========================
# Initialize session state
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "stats" not in st.session_state:
    st.session_state.stats = None

if "last_video_id" not in st.session_state:
    st.session_state.last_video_id = None

if "video_summary" not in st.session_state:
    st.session_state.video_summary = None

if "comment_vectorstore" not in st.session_state:
    st.session_state.comment_vectorstore = None

if "transcript_vectorstore" not in st.session_state:
    st.session_state.transcript_vectorstore = None

if "pending_user_query" not in st.session_state:
    st.session_state.pending_user_query = None

if "comment" not in st.session_state:
    st.session_state.comment = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "chat_enabled" not in st.session_state:
    st.session_state.chat_enabled = False

# =========================
# Define constants
MAX_COMMENTS = 200


# =========================
# Main UI
# =========================
st.set_page_config(page_title="YouTube Video Analyzer", layout="wide")

st.title("🎥 YouTube Video Analyzer")
st.info("Analyze YouTube video based on video's information and comment sentiments. Help content creators understand their audience better.")

url = st.text_input("Input your YouTube video URL:")


if url:
    status = st.empty()

    status.info("Processing URL...")

    video_id = extract_video_id(url)

    if not video_id:
        status.empty()
        st.error("❌ URL is not valid. Please enter a valid YouTube video URL.")
    else:
        status.empty()
        st.success("✅ URL processed successfully.")
        
        info = get_video_info(video_id)
        if info:
            st.divider()
            render_video_info(info, cached_summarize_video, url)

        if video_id != st.session_state.last_video_id:
            # Core data
            st.session_state.analysis_done = False
            st.session_state.stats = None
            st.session_state.last_video_id = video_id

            # Reset RAG-related state
            st.session_state.comment_vectorstore = None
            st.session_state.transcript_vectorstore = None
            st.session_state.video_summary = None
            st.session_state.chat_history = []
            if "rag_memory" in st.session_state:
                del st.session_state.rag_memory
            st.session_state.pending_user_query = None

            # Reset UI options
            st.session_state.use_comment_likes = False

            st.rerun()

        st.divider()

        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            if st.button(
                "Analyze Video",
                disabled=st.session_state.analysis_done,
                help="Could take a few minutes for videos with many comments."):

                st.write(f"Maximum {MAX_COMMENTS} comments are analyzed.")
                with st.spinner("🔄 Fetching comments & analyzing sentiment..."):
                    
                    comments = cached_get_video_comments(video_id, max_results=MAX_COMMENTS)
                    texts = [{"text": c["text"]} for c in comments]

                    predictions = cached_analyze_sentiment(texts)
                    st.session_state.comment = merge_comments_with_sentiment(comments, predictions)

                    st.session_state.stats = sentiment_statistics(st.session_state.comment)

                    st.session_state.analysis_done = True

            if st.session_state.analysis_done:
                render_sentiment(st.session_state.stats)

        with right_col:
            if st.session_state.analysis_done:
                if st.button("Chat with AI about this video",
                            disabled=st.session_state.chat_enabled):
                    with st.spinner("⚙️ Setting up chat..."):
                        st.session_state.embedder = GeminiEmbedding()
                        st.session_state.comment_vectorstore = build_comment_vectorstore(st.session_state.comment, embeddings=st.session_state.embedder)

                        transcript = cached_get_video_transcript(video_id)
                        merged_transcript = merge_transcript_by_time(transcript, max_duration=90.0)

                        st.session_state.transcript_vectorstore = build_transcript_vectorstore(merged_transcript, embeddings=st.session_state.embedder)
                        
                        st.session_state.chat_enabled = True
                        st.rerun()
                if st.session_state.chat_enabled:
                    chat_box = st.container(height=800, border=False)
                    with chat_box:
                        render_chat(info)