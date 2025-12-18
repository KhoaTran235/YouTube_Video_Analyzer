import numpy as np
from rag_pipeline.gemini_embedding import GeminiEmbedding

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_router(query: str, embedder=GeminiEmbedding()) -> str:
    if embedder is None:
        return "NO_RETRIEVAL"
    routes = {
        "NO_RETRIEVAL": "greeting, casual chat, general question, small talk",
        "RAG": (
            "question about video content, transcript, comments, sentiment, "
            "audience opinion, summary, statistics"
        ),
    }
    q_vec = embedder.embed_query(query)

    sims = {}
    for route, desc in routes.items():
        route_vec = embedder.embed_query(desc)
        sims[route] = cosine_sim(q_vec, route_vec)

    best_route, score = max(sims.items(), key=lambda x: x[1])

    # for debugging 
    print("🔀 Router scores:", sims)

    # if score < 0.65:
    #     return "LLM_ROUTER"

    return best_route