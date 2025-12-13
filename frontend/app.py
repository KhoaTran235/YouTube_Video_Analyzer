# app.py
import streamlit as st

from utils import extract_video_id, clean_text, translate_text
from services.comment_sentiment import analyze_sentiment
from services.yt_service import get_video_info, get_video_comments, get_video_transcript
from utils import merge_comments_with_sentiment, sentiment_statistics

# =========================
# Define cached functions
# =========================
@st.cache_data(show_spinner=False)
def cached_get_video_comments(video_id, max_results):
    return get_video_comments(video_id, max_results)

@st.cache_data(show_spinner=False)
def cached_analyze_sentiment(texts):
    return analyze_sentiment(texts)

# =========================
# Session state init
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "stats" not in st.session_state:
    st.session_state.stats = None

if "last_video_id" not in st.session_state:
    st.session_state.last_video_id = None

# =========================
# Main UI
# =========================
st.set_page_config(page_title="YouTube Video Analyzer", layout="wide")

st.title("🎥 YouTube Video Analyzer")
st.info("Analyze YouTube video based on video's information and comment sentiments.")

url = st.text_input("Input YouTube video URL:")


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
        
        info = get_video_info(video_id) # Fetch video info
        if info:
            st.subheader("📊 Video Information:")
            st.write(f"**Title:** {info['title']}")
            st.write(f"**Description:** {info['description']}")
            st.write(f"👁️ {info['views']:,} views | 👍 {info['likes']:,} likes | 💬 {info['comments']:,} comments")

        if video_id != st.session_state.last_video_id:      # New video URL entered (reset state)
            st.session_state.analysis_done = False
            st.session_state.stats = None
            st.session_state.last_video_id = video_id

        if st.button("Analyze Video"):
            with st.spinner("🔄 Fetching comments & analyzing sentiment..."):
                MAX_COMMENTS = 500
                comments = cached_get_video_comments(video_id, max_results=MAX_COMMENTS)
                texts = [{"text": c["text"]} for c in comments]

                predictions = cached_analyze_sentiment(texts)
                merged = merge_comments_with_sentiment(comments, predictions)

                stats = sentiment_statistics(merged)

                st.session_state.stats = stats
                st.session_state.analysis_done = True

        if st.session_state.analysis_done:
            st.divider()
            left_col, right_col = st.columns([1.2, 1])
            
            with left_col:
                # --- Summary ---
                st.subheader("📊 Comment Sentiment Overview")

                use_comment_likes = st.checkbox(
                    "Analyze based on comment likes",
                    value=True,
                    help=(
                        "✔ Unchecked: Each comment counts as 1\n\n"
                        "✔ Checked: Comments with more likes contribute more weight\n"
                        "(likeCount is used as sentiment weight - meaning popular comments have more influence on the final sentiment distribution)"
                    )
                )

                stats = st.session_state.stats
                mode = "weighted" if use_comment_likes else "raw"
                dist = stats[mode]["distribution"]

                st.markdown(
                    f"""
                    **Total comments:** {stats["real_total"]}  
                    """
                )

                # st.divider()

                # --- Sentiment rows ---
                col1, col2, col3 = st.columns(3)

                def render_metric(col, label, emoji):
                    data = dist[label]

                    comment_text = f'{data["comment_count"]} comments'
                    if use_comment_likes:
                        comment_text += f' (👍 {data["like_weight"]} likes)'

                    with col:
                        st.metric(
                            label=f"{emoji} {label.capitalize()}",
                            value=f'{data["percentage"]}%',
                            delta=comment_text
                        )

                render_metric(col1, "negative", "🔴")
                render_metric(col2, "neutral", "🟡")
                render_metric(col3, "positive", "🟢")

                st.divider()

                # --- Explanation ---
                st.caption(
                    "ℹ️ **How sentiment is calculated:**\n"
                    "- When *Analyze based on comment likes* is OFF: each comment counts as 1\n"
                    "- When ON: comments with higher likes have more influence on sentiment distribution\n"
                    "- Percentages are always normalized over the selected method"
                )

            with right_col:
                st.subheader("🤖 Ask AI about this video (Powered by Google Gemini)")

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []


                # ===== CHAT BOX (SCROLLABLE) =====
                chat_box = st.container(height=400)  # chỉnh chiều cao tùy ý

                with chat_box:
                    for chat in st.session_state.chat_history:
                        with st.chat_message("user"):
                            st.write(chat["user"])
                        with st.chat_message("assistant"):
                            st.write(chat["assistant"])

                # ===== RESET BUTTON =====
                if st.session_state.chat_history:
                    if st.button("🗑 Reset conversation"):
                        st.session_state.chat_history = []
                        st.rerun()

                st.divider()

                # ===== INPUT LUÔN Ở DƯỚI =====
                user_query = st.chat_input(
                    "Ask a question about comments, sentiment, or audience opinion"
                )

                if user_query:
                    with st.spinner("Thinking..."):
                        answer = "You are noob."
                        st.session_state.chat_history.append({
                            "user": user_query,
                            "assistant": answer
                        })
                    st.rerun()

