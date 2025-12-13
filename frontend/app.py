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
st.set_page_config(page_title="YouTube Video Analyzer", layout="centered")

st.title("🎥 YouTube Video Analyzer")
st.info("Analyze YouTube video based on video stats and comment sentiments.")

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
        if video_id != st.session_state.last_video_id:      # New video URL entered (reset state)
            st.session_state.analysis_done = False
            st.session_state.stats = None
            st.session_state.last_video_id = video_id

        if st.button("Analyze Video"):
            with st.spinner("🔄 Fetching comments & analyzing sentiment..."):
                MAX_COMMENTS = 100
                comments = cached_get_video_comments(video_id, max_results=MAX_COMMENTS)
                texts = [{"text": c["text"]} for c in comments]

                predictions = cached_analyze_sentiment(texts)
                merged = merge_comments_with_sentiment(comments, predictions)

                stats = sentiment_statistics(merged)

                st.session_state.stats = stats
                st.session_state.analysis_done = True
                
        if st.session_state.analysis_done:
            st.divider()
    
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

            # --- Summary ---
            st.subheader("📊 Comment Sentiment Overview")

            st.markdown(
                f"""
                **Total comments:** {stats["real_total"]}  
                """
            )

            st.divider()

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
#         with st.spinner("🔍 Đang lấy dữ liệu video..."):
#             info = get_video_info(video_id)
#             comments = get_comments(video_id, max_results=MAX_COMMENTS)

#         if info:
#             st.subheader("📊 Thông tin video:")
#             st.write(f"**Tiêu đề:** {info['title']}")
#             st.write(f"👁️ {info['views']:,} lượt xem | 👍 {info['likes']:,} like | 💬 {info['comments']:,} bình luận")

#             if comments:
#                 st.info(f"Đã thu được {len(comments)} bình luận. Đang phân tích sentiment...")

#                 try:
#                     sentiments = analyze_sentiment(comments)
#                 except Exception as e:
#                     st.error(f"Lỗi khi gọi API sentiment: {e}")
#                     st.stop()

#                 score, pos, neg = compute_weighted_score(info, comments, sentiments, use_weight)

#                 df = pd.DataFrame({
#                     "comment": [c["text"] for c in comments],
#                     "likes": [c["likeCount"] for c in comments],
#                     "sentiment": sentiments
#                 })
#                 st.dataframe(df)

#                 st.metric("⭐ Điểm tổng quan", f"{score}/100")
#                 st.progress(score / 100)
#                 st.write(f"✅ Bình luận tích cực: **{pos*100:.1f}%**")
#                 st.write(f"❌ Bình luận tiêu cực: **{neg*100:.1f}%**")

#             else:
#                 st.warning("Không lấy được bình luận cho video này.")
#         else:
#             st.error("Không thể truy xuất thông tin video.")
