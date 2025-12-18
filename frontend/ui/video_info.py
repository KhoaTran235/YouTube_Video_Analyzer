import streamlit as st

def render_video_summary(summarize_fn, video_url):
    if st.button("Get Video Summary by AI"):
        with st.spinner("This may take a while..."):
            st.session_state.video_summary = summarize_fn(video_url)
            st.rerun()

    if st.session_state.video_summary:
        st.write(f"**AI-generated Summary:** {st.session_state.video_summary}")

def render_video_info(info, summarize_fn, video_url):
    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        st.subheader("📊 Video Information")

        st.write(f"**Title:** {info['title']}")
        st.write(f"**Description:** {info['description']}")
        st.write(
            f"👁️ {info['views']:,} views | "
            f"👍 {info['likes']:,} likes | "
            f"💬 {info['comments']:,} comments"
        )

    with right_col:
        render_video_summary(summarize_fn, video_url)
