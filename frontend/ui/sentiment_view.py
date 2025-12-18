import streamlit as st
import matplotlib.pyplot as plt

@st.fragment
def render_sentiment(stats):
    st.subheader("📊 Comment Sentiment Overview")

    use_likes = st.checkbox(
        "Analyze based on comment likes",
        value=False,
        help=(
            "Unchecked: each comment counts as 1\n\n"
            "Checked: popular comments have more influence"
        )
    )

    mode = "weighted" if use_likes else "raw"
    dist = stats[mode]["distribution"]

    cols = st.columns(3)

    def metric(col, label, emoji):
        data = dist[label]
        delta = f'{data["comment_count"]} comments'
        if use_likes:
            delta += f' (👍 {data["like_weight"]} likes)'
        col.metric(
            f"{emoji} {label.capitalize()}",
            f'{data["percentage"]}%',
            delta
        )

    metric(cols[0], "positive", "🟢")
    metric(cols[1], "neutral", "🟡")
    metric(cols[2], "negative", "🔴")

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.pie(
        [dist["positive"]["percentage"],
        dist["neutral"]["percentage"],
        dist["negative"]["percentage"]],
        labels=["Positive", "Neutral", "Negative"],
        autopct="%1.2f%%",
        startangle=90,
        counterclock=False,
        colors=["#66bb66", "#ffdd66", "#ff6666"]
    )
    ax.axis("equal")

    st.pyplot(fig)

    st.caption(
                    "ℹ️ **How sentiment is calculated:**\n"
                    "- When *Analyze based on comment likes* is OFF: each comment counts as 1\n"
                    "- When ON: comments with higher likes have more influence on sentiment distribution\n"
                    "- Percentages are always normalized over the selected method"
                )
