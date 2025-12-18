from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag_pipeline.gemini_embedding import GeminiEmbedding
from utils import split_sentences

MAX_BATCH = 100
MIN_TRANSCRIPT_LEN = 5

def build_comment_vectorstore(comments, embeddings=GeminiEmbedding()):
    docs = []
    for c in comments:
        text = c.get("text", "")
        if not text or not text.strip():
            continue

        docs.append(
            Document(
                page_content=(
                    f"[COMMENT]\n"
                    f"Sentiment: {c['sentiment']}\n"
                    f"Number of likes: {c.get('likeCount')}\n"
                    f"Text: {text.strip()}\n"
                ),
                metadata={
                    "type": "comment",
                    "author": c.get("author", "unknown"),
                    "sentiment": c.get("sentiment", "unknown"),
                    "likeCount": c.get("likeCount", 0),
                }
            )
        )

    if not docs:
        return None  # hoặc raise warning
    # print(docs)
    vectorstore = None
    for i in range(0, len(docs), MAX_BATCH):
        batch = docs[i:i + MAX_BATCH]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    return vectorstore



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

def build_transcript_vectorstore(transcript, embeddings=GeminiEmbedding()):
    """
    Build a FAISS vector store from transcript chunks.

    transcript: list of dict
        {
            "text": str,
            "start": float,
            "duration": float
        }
    """
    docs = []

    for chunk in chunk_transcript(transcript):
        text = chunk.get("text", "")

        if not text or not text.strip():
            continue

        text = text.strip()

        if len(text) < MIN_TRANSCRIPT_LEN:
            continue
    docs.append(
        Document(
            page_content=chunk["text"],
            metadata={
                "type": "transcript",
                "start": chunk["start"],
                "end": chunk["end"]
            }
        )
    )

    if not docs:
        return None  # hoặc raise warning/log

    vectorstore = None
    for i in range(0, len(docs), MAX_BATCH):
        batch = docs[i:i + MAX_BATCH]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    return vectorstore