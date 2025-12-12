import re
import math

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
    return text


def compute_weighted_score(info, comments, sentiments, use_weight=True):
    if not sentiments:
        return 0, 0, 0

    weights = []
    for c in comments:
        if use_weight:
            w = 1 + math.log10(1 + c["likeCount"])
        else:
            w = 1
        weights.append(w)

    total_w = sum(weights)
    pos = sum(w for w, s in zip(weights, sentiments) if s == "positive") / total_w
    neg = sum(w for w, s in zip(weights, sentiments) if s == "negative") / total_w

    like_ratio = info["likes"] / max(info["views"], 1)
    score = 100 * (0.5 * pos + 0.3 * like_ratio + 0.2 * (1 - neg))
    return round(score, 2), pos, neg
