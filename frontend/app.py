# app.py
import streamlit as st
import pandas as pd

from utils import extract_video_id, clean_text, translate_text
from services.yt_service import get_video_info, get_comments
from services.comment_sentiment import analyze_sentiment

# =========================
# Main UI
# =========================
st.set_page_config(page_title="YouTube Video Analyzer", layout="centered")
st.title("🎥 YouTube Video Analyzer")
st.info("Analyze YouTube video based on video stats and comment sentiments.")
url = st.text_input("Input YouTube video URL:")
use_video_likes = st.checkbox("Include video like ratio in score", value=True)
use_comment_likes = st.checkbox("Analyze based on comment likes", value=True)


if url:
    st.info("Processing URL...")
    video_id = extract_video_id(url)
    st.success("✅ URL processed.")
    st.write("**Video ID:**", video_id)
#     if not video_id:
#         st.error("❌ Không thể nhận dạng video ID từ URL.")
#     else:
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
