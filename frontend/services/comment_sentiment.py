import requests

def analyze_sentiment(comments):
    texts = [c["text"] for c in comments]
    payload = {"texts": texts}
    response = requests.post(SENTIMENT_API_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("predictions", [])
    else:
        raise RuntimeError(f"Sentiment API error: {response.text}")
