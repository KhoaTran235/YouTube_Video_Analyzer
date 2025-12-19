# YouTube Video Analyzer

A simple web application that helps YouTube Content Creators understand their audience better.

[*Demo*](https://youtubevideoanalyzer.streamlit.app/)
---
## Features

- Fetch video metadata (title, views, likes, etc.)
- Collect comments, predict comment sentiment (negative, neutral, positive)
- Create video summary by Google Gemini
- Generate basic statistics and sentiment analysis
- RAG (Retrieval-Augmented Generation) chatbot for Q&A on metadata, comments, transcript, summary

---
## System Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture.

- YouTube data (metadata, comments, transcript) is collected via API
- Comments are analyzed using a Sentiment Analysis API
- Video is summarized using Google Gemini
- Comments with sentiment and chunked transcripts are embedded into a vector store
- The chatbot retrieves relevant context and generates answers using Gemini LLM

```mermaid
flowchart LR
    U[User] --> |Input Video URL|UI[Streamlit UI]

    UI --> GS[Gemini LLM]
    UI --> Y_API[YouTube Data API]
    UI --> T_API[YouTube Transcript API]
    
    Y_API --> M[Metadata]
    Y_API --> C[Comments]
    
    GS --> VS[Video Summary]
    
    T_API --> T[Transcript]

    C --> P[Text Pre-processing]
    P --> S[Sentiment API]
    S --> CS[Comments + Sentiment]
    
    
    VS --> PRT
    M --> PRT[Build System Prompt]

    CS --> E[Gemini Embedding Model]
    T --> E[Gemini Embedding Model]
    E --> V[(FAISS Vector Store)]

    UI --> Q[User Query]
    
    Q --> SM{Semantic Router}
    PRT --> SM

    SM --> |No retrieve|L[Gemini LLM]
    SM --> |RAG|R

    V --> R[RAG Pipeline]
    
    R --> L

    L --> A[Answer]
    A --> UI
```

---
## RAG Chatbot

Instead of directly answering user questions, the chatbot:

1. Retrieves relevant information from comments, transcript
2. Injects the retrieved context, metadata, summary into the prompt
3. Generates grounded and explainable answers using Gemini LLM

This approach reduces hallucination and improves factual accuracy.

---
## Installation, Configuration and Usage


### Clone Repository
```bash
git clone https://github.com/KhoaTran235/YouTube_Video_Analyzer.git
cd YouTube_Video_Analyzer
```

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port ${PORT}
```

### Frontend (Streamlit)

1. Create **.env** file in the `frontend` folder:
```bash
GOOGLE_API_KEY=<YOUR_API_KEY>
SENTIMENT_API_URL=<YOUR_BACKEND_API_URL>
```
2. Install dependencies and run the app:
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
```
---
## Example
![Demo](screenshot/example_1.png)
![Demo](screenshot/example_2.png)
