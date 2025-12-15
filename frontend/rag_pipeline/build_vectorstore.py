from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag_pipeline.gemini_embedding import GeminiEmbedding
from utils import split_sentences


def build_comment_vectorstore(comments):
    """
    Build a FAISS vector store from video's comments and transcript.
    comments: list of dict, each dict has "text", "author", "likes", "sentiment" fields
    """
    embeddings = GeminiEmbedding()
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "type": "comment",
                "author": c["author"],
                "sentiment": c["sentiment"],
                "likeCount": c.get("likeCount", 0),
            }
        )
        for c in comments
    ]

    return FAISS.from_documents(docs, embeddings)


def chunk_transcript(
    merged_records,
    max_words=250,
    overlap_sentences=2
):
    """
    merged_records: [
        {
            "text": "...",
            "start": float,
            "duration": float
        }
    ]
    """

    chunks = []

    for record in merged_records:
        # text = clean_text(record["text"])
        sentences = split_sentences(record["text"])

        buffer = []
        buffer_word_count = 0
        start_time = record["start"]

        for i, sent in enumerate(sentences):
            words = sent.split()
            buffer.append(sent)
            buffer_word_count += len(words)

            if buffer_word_count >= max_words:
                chunks.append({
                    "text": " ".join(buffer),
                    "start": start_time,
                    "end": record["start"] + record["duration"]
                })

                # overlap: giữ lại vài câu cuối
                buffer = buffer[-overlap_sentences:]
                buffer_word_count = sum(len(s.split()) for s in buffer)

        # phần còn lại
        if buffer:
            chunks.append({
                "text": " ".join(buffer),
                "start": start_time,
                "end": record["start"] + record["duration"]
            })

    return chunks

def build_transcript_vectorstore(transcript):
    """
    Build a FAISS vector store from transcript chunks.

    transcript: list of dict
        {
            "text": str,
            "start": float,
            "duration": float
        }
    """
    embeddings = GeminiEmbedding()

    docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "type": "transcript",
                "start": chunk["start"],
                "end": chunk["end"]
            }
        )
        for chunk in chunk_transcript(transcript)
        if chunk["text"].strip()
    ]

    return FAISS.from_documents(docs, embeddings)