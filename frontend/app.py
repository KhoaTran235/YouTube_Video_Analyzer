# app.py
import streamlit as st
import matplotlib.pyplot as plt
from utils import extract_video_id
from services.comment_sentiment import analyze_sentiment
from services.yt_service import get_video_info, get_video_comments, get_video_transcript
from services.video_summarize import summarize_video
from utils import merge_comments_with_sentiment, sentiment_statistics, merge_transcript_by_time
from rag_pipeline.build_vectorstore import build_comment_vectorstore, build_transcript_vectorstore
from rag_pipeline.chain import get_session_rag_chain

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
# Session state init
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



MAX_COMMENTS = 10
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
            st.divider()
            left_col, right_col = st.columns([1.2, 1])
            with left_col:
                st.subheader("📊 Video Information:")
                st.write(f"**Title:** {info['title']}")
                st.write(f"**Description:** {info['description']}")
                st.write(f"👁️ {info['views']:,} views | 👍 {info['likes']:,} likes | 💬 {info['comments']:,} comments")
            with right_col:
                if st.button("Get Video Summary by AI"):
                    st.session_state.video_summary = cached_summarize_video(url)

                if st.session_state.video_summary:
                    st.write(f"**AI-generated Summary:** {st.session_state.video_summary}")

        if video_id != st.session_state.last_video_id:      # New video URL entered (reset state)
            st.session_state.analysis_done = False
            st.session_state.stats = None
            st.session_state.last_video_id = video_id
            st.session_state.comment_vectorstore = None
            st.session_state.transcript_vectorstore = None
            st.session_state.video_summary = None

        st.divider()

        if st.button("Analyze Video"):
            st.write(f"Maximum {MAX_COMMENTS} comments are analyzed.")
            with st.spinner("🔄 Fetching comments & analyzing sentiment..."):

                comments = cached_get_video_comments(video_id, max_results=MAX_COMMENTS)
                texts = [{"text": c["text"]} for c in comments]

                predictions = cached_analyze_sentiment(texts)
                merged_comment = merge_comments_with_sentiment(comments, predictions)

                stats = sentiment_statistics(merged_comment)
                st.session_state.comment_vectorstore = build_comment_vectorstore(merged_comment)

                transcript = cached_get_video_transcript(video_id)
                merged_transcript = merge_transcript_by_time(transcript, max_duration=90.0)

                st.session_state.transcript_vectorstore = build_transcript_vectorstore(merged_transcript)
                
                # if st.session_state.video_summary is None:
                #     st.session_state.video_summary = cached_summarize_video(url)


                st.session_state.stats = stats
                st.session_state.analysis_done = True

        if st.session_state.analysis_done:
            
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

                labels = ["Negative", "Neutral", "Positive"]
                sizes = [
                    dist["negative"]["percentage"],
                    dist["neutral"]["percentage"],
                    dist["positive"]["percentage"],
                ]

                colors = ["#ff4d4d", "#ffd966", "#4CAF50"]

                fig, ax = plt.subplots(figsize=(5, 2))
                ax.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.2f%%",
                    startangle=90,
                    colors=colors,
                    wedgeprops={"edgecolor": "white"},
                )

                ax.axis("equal")  # hình tròn chuẩn
                ax.set_title("Sentiment Distribution", pad=20)

                st.pyplot(fig)


                # --- Explanation ---
                st.caption(
                    "ℹ️ **How sentiment is calculated:**\n"
                    "- When *Analyze based on comment likes* is OFF: each comment counts as 1\n"
                    "- When ON: comments with higher likes have more influence on sentiment distribution\n"
                    "- Percentages are always normalized over the selected method"
                )

            with right_col:
                st.subheader("🤖 Ask AI about this video (Powered by Google Gemini)")
                if st.session_state.video_summary:
                    st.success("Video summary is enabled for chatbot")
                else:
                    st.info("Chatbot is using transcript and comments only (no summary)")
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []


                # ===== CHAT BOX (SCROLLABLE) =====
                chat_box = st.container(height=482)  # chỉnh chiều cao tùy ý

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
                        if "rag_memory" in st.session_state:
                            del st.session_state.rag_memory
                        st.rerun()


                # ===== INPUT LUÔN Ở DƯỚI =====
                user_query = st.chat_input(
                    "Ask a question about comments, sentiment, or audience opinion"
                )

                if user_query:
                    with st.spinner("Thinking..."):
                        qa_chain = get_session_rag_chain(
                                        info,
                                        video_summary=st.session_state.video_summary
                                    )

                        if qa_chain is None:
                            answer = "Please analyze the video first."
                        else:
                            result = qa_chain.invoke(user_query)
                            answer = result.content

                            st.session_state.rag_memory.save_context(
                                {"input": user_query},
                                {"output": answer}
                            )
                        
                        st.session_state.chat_history.append({
                            "user": user_query,
                            "assistant": answer
                        })

                    st.rerun()



