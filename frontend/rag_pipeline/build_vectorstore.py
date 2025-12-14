from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag_pipeline.gemini_embedding import GeminiEmbedding


def build_vectorstore(comments, transcript=None):
    """
    Build a FAISS vector store from video's comments and transcript.
    comments: list of dict, each dict has "text", "author", "likes", "sentiment" fields
    transcript: string
    """
    embeddings = GeminiEmbedding()
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "author": c["author"],
                "sentiment": c["sentiment"],
                "likeCount": c.get("likeCount", 0),
            }
        )
        for c in comments
    ]

    return FAISS.from_documents(docs, embeddings)