import os
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv


# from config import YOUTUBE_API_KEY
load_dotenv()
YOUTUBE_API_KEY = os.getenv("GOOGLE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
ytt_api = YouTubeTranscriptApi()
# response = youtube.commentThreads().list(
#     part="snippet,replies",
#     videoId="sQcrZHvrEnU"
# ).execute()
# print(response["items"][0]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
# print(response["items"][0]["replies"]["comments"][0]["snippet"]["textOriginal"])
# print(response["items"][0]["replies"]["comments"][1]["snippet"]["textOriginal"])

# for item in response["items"]:
#     s = item["snippet"]["topLevelComment"]["snippet"]
#     author = s.get("authorDisplayName", "")
#     text = html.unescape(s.get("textDisplay", ""))  
#     likes = s.get("likeCount", 0)
#     print(f"{author}: {text} ({likes} likes)")


def get_video_info(video_id):
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id,

    ).execute()

    if not response["items"]:
        return None
    item = response["items"][0]
    return {
        "title": item["snippet"]["title"],
        "views": int(item["statistics"].get("viewCount", 0)),
        "likes": int(item["statistics"].get("likeCount", 0)),
        "comments": int(item["statistics"].get("commentCount", 0)),
    }

def get_video_transcript(video_id):
    fetched_transcript = ytt_api.fetch(video_id)

    # is iterable
    for snippet in fetched_transcript:
        print(snippet.text)

    # indexable
    last_snippet = fetched_transcript[-1]

    # provides a length
    snippet_count = len(fetched_transcript)

def get_video_comments(video_id, max_results=1000):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=min(max_results, 100)
    )
    response = request.execute()

    while response:
        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet["textOriginal"],
                "likeCount": snippet.get("likeCount", 0)
            })

        if "nextPageToken" in response and len(comments) < max_results:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response["nextPageToken"],
                textFormat="plainText",
                maxResults=min(max_results - len(comments), 100)
            )
            response = request.execute()
        else:
            break

    return comments

video_id = "8V82hbiVzfE"
info = get_video_info(video_id)
comments = get_video_comments(video_id, max_results=100)
print("=== VIDEO INFO ===")
print(info)
print("=== COMMENTS ===")
for c in comments:
    print(c)
print(f"Total comments fetched: {len(comments)}")
get_video_transcript(video_id)