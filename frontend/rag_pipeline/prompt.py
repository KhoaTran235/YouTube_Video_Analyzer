from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
    You are an AI assistant analyzing a YouTube video for content creators.

    Video title:
    {title}

    Video description:
    {description}

    Conversation so far:
    empty

    Rules:
    - Answer strictly based on the retrieved context
    - If information is insufficient, say so clearly
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """
        Context:
        {context}

        Question:
        {question}
    """)
])