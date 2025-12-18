import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
from utils import clean_text

load_dotenv()
SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL")

def send_batch(chunk):
    payload = {"texts": chunk}
    try:
        response = requests.post(
            f"{SENTIMENT_API_URL}/predict",
            json=payload,
            timeout=100
        )
        response.raise_for_status()  # bắt 4xx / 5xx

        return response.json()

    except requests.exceptions.Timeout:
        raise RuntimeError("Sentiment API timeout")

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Sentiment API connection error")

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Sentiment API HTTP error {response.status_code}: {response.text}"
        ) from e

    except Exception as e:
        raise RuntimeError(f"Unexpected Sentiment API error: {str(e)}") from e
    
def analyze_sentiment(comments):
    texts = [c["text"] for c in comments]
    texts = [clean_text(t) for t in texts]
    sentiments = []
    if len(texts) > 128:
        chunks = [texts[i:i+128] for i in range(0, len(texts), 128)]


        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(send_batch, c) for c in chunks]

            for f in as_completed(futures):
                sentiments.append(f.result())

    else:
        sentiments.append(send_batch(texts))
    
    if len(sentiments) == 1:
        sentiments = sentiments[0]

    elif len(sentiments) > 1:
        batch_size = 0
        results = []
        for batch in sentiments:
            batch_size += batch['batch_size']
            results.extend(batch['results'])
        sentiments = {"batch_size": batch_size, "results": results}

    return sentiments