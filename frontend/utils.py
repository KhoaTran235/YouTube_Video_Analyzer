import re
from collections import Counter

def extract_video_id(url: str):
    pattern = r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def clean_text(text: str):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    return text.strip()

def translate_text(text: str, target_language: str = "en"):
    pass

SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def merge_comments_with_sentiment(comments, predictions):

    merged = []

    results = predictions["results"]

    for i, cmt in enumerate(comments):
        sent = results[i]

        merged.append({
            "author": cmt["author"],
            "text": cmt["text"],
            "likeCount": cmt["likeCount"],
            "sentiment": SENTIMENT_MAP[sent["predicted_class"]]
        })

    return merged


def sentiment_statistics(merged):
    comment_counter = Counter()
    like_counter = Counter()
    weight_counter = Counter()

    for item in merged:
        label = item["sentiment"]
        likes = max(item.get("likeCount", 0), 0)

        comment_counter[label] += 1
        like_counter[label] += likes
        weight_counter[label] += 1 + likes

    total_comments = sum(comment_counter.values())
    total_weight = sum(weight_counter.values())

    def build_distribution(comment_ctr, weight_ctr, like_ctr, total, use_like=False):
        dist = {}
        for label in ["negative", "neutral", "positive"]:
            base = weight_ctr[label] if use_like else comment_ctr[label]
            percentage = round(base / total * 100, 2) if total > 0 else 0.0

            dist[label] = {
                "comment_count": comment_ctr[label],
                "like_weight": like_ctr[label],
                "percentage": percentage
            }
        return dist

    return {
        "real_total": total_comments,

        "raw": {
            "total": total_comments,
            "distribution": build_distribution(
                comment_counter,
                comment_counter,
                like_counter,
                total_comments,
                use_like=False
            )
        },

        "weighted": {
            "total": total_weight,
            "distribution": build_distribution(
                comment_counter,
                weight_counter,
                like_counter,
                total_weight,
                use_like=True
            )
        }
    }

def merge_transcript_by_time(snippets, max_duration=60.0):
    merged = []
    buffer = []
    start_time = None
    total_duration = 0.0

    for s in snippets:
        if start_time is None:
            start_time = s.start

        buffer.append(s.text)
        total_duration += s.duration

        if total_duration >= max_duration:
            merged.append({
                "text": " ".join(buffer),
                "start": start_time,
                "duration": total_duration
            })
            buffer = []
            start_time = None
            total_duration = 0.0

    if buffer:
        merged.append({
            "text": " ".join(buffer),
            "start": start_time,
            "duration": total_duration
        })

    return merged

def split_sentences(text: str):
    # Split by . ? !
    return re.split(r'(?<=[.!?])\s+', text)
