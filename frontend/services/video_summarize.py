"""Summarize YouTube videos using Google Gemini API."""
import google.generativeai as genai
import os
from dotenv import load_dotenv

# https://aistudio.google.com/app/apikey
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def summarize_video(video_url):
    prompt = """
    You are summarizing a YouTube video to provide background context for a retrieval-augmented QA system.

    Requirements:
    - Maximum 150 words
    - High-level and neutral summary
    - Describe the main topic, speaker intent, and overall structure
    - Do NOT include technical details or step-by-step explanations
    - Do NOT include viewer opinions or comments
    - Do NOT quote sentences verbatim
    - Do NOT introduce interpretations beyond what is clearly stated

    Important:
    - This summary is for background understanding only
    - It must NOT be treated as a source of factual evidence for answering questions

    Write in clear, factual language.
    """
    response = model.generate_content(
        [
            {"text": prompt},
            {"file_data": {"mime_type": "video/mp4", "file_uri": video_url}},
        ]
    )
    return response.text