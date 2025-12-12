from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def comment_summarizer(comments):
    comments_text = "\n".join([f"- {c['text']}" for c in comments])

    # Prompt hướng dẫn mô hình
    system_prompt = (
        "You are an expert at summarizing user comments on YouTube videos "
        "by sentiment group. Given the following comments, provide a concise "
        "summary highlighting the main points expressed by the users."
    )

    human_prompt = f"""
    Comments:
    {comments_text}

    Summary:
    """

    # Khởi tạo model
    chat = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    # Gọi model
    response = chat.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    print("Summary: ", response.content)

# Test
comments = [
    {"text": "I absolutely love this video! It made my day."},
    {"text": "This was so helpful, thank you!"},
    {"text": "I find this useful."},
    {"text": "Amazing content, keep it up!"},
    {"text": "Not what I expected, but still good."}
]

comment_summarizer(comments)