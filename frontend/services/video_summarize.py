"""Summarize YouTube videos using Google Gemini API."""
import google.generativeai as genai
import os
from dotenv import load_dotenv

# https://aistudio.google.com/app/apikey
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

def summarize_video(video_url):
    prompt = "Summarize the main content of this video, highlighting the topic, speaker, and key points. List multiple sections in chronological order."
    response = model.generate_content(
        [
            {"text": prompt},
            {"file_data": {"mime_type": "video/mp4", "file_uri": video_url}},
        ]
    )
    return response.text

print(summarize_video("https://www.youtube.com/watch?v=sQcrZHvrEnU"))