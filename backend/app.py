from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import asyncio
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import time
from fastapi import FastAPI, HTTPException
from models.sentiment import TextsRequest, PredictionResponse

# MODEL_PATH = "onnx_lora_bert/model.onnx"
# TOKENIZER_PATH = "onnx_lora_bert"
BATCH_LIMIT = 16   # batch limit for each inference call
MAX_LENGTH = 128   # max token length
MAX_REQUEST_SAMPLES = 128  # max samples per request
NUM_WORKERS = 4    # number of concurrent threads
REQUEST_TIMEOUT = 90.0  # seconds

if "CUDAExecutionProvider" in ort.get_available_providers():
    provider = "CUDAExecutionProvider"
else:
    provider = "CPUExecutionProvider"

HF_REPO = "khoa-tran-hcmut/sentiment_lora_bert"

MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO,
    filename="model.onnx",
    subfolder="onnx_lora_bert"
)
tokenizer = AutoTokenizer.from_pretrained(
    HF_REPO,
    subfolder="onnx_lora_bert"
)

print("Loaded from HuggingFace:", MODEL_PATH)

session = ort.InferenceSession(MODEL_PATH, providers=[provider])
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"Model loaded on: {session.get_providers()[0]}")

executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

app = FastAPI(title="Sentiment Analysis API", version="1.0")


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    # print(f"{request.method} {request.url.path} completed in {duration:.3f}s")
    response.headers["X-Execution-Time"] = f"{duration:.3f}s"
    return response

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"detail": "Request timed out."})


# Inference functions
def run_inference_batch(texts: list[str]) -> list[int]:
    inputs = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])),
        }
    )

    logits = outputs[0]
    predicted_class_ids = np.argmax(logits, axis=-1).tolist()
    return predicted_class_ids

async def run_inference_async(texts: list[str]) -> list[int]:
    tasks = []
    for i in range(0, len(texts), BATCH_LIMIT):
        chunk = texts[i:i + BATCH_LIMIT]
        # Chạy inference trong thread pool để không block event loop
        task = asyncio.get_event_loop().run_in_executor(executor, run_inference_batch, chunk)
        tasks.append(task)

    # Chờ tất cả hoàn thành
    all_preds = await asyncio.gather(*tasks)
    return [p for chunk in all_preds for p in chunk]


# API Endpoints
@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use the /predict endpoint to analyze sentiment."}    

semaphore = asyncio.Semaphore(NUM_WORKERS)

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(text_request: TextsRequest):
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=20)
    except asyncio.TimeoutError:
        raise HTTPException(429, "Too many concurrent requests")

    try:
        texts = text_request.texts

        if not texts:
            raise HTTPException(status_code=400, detail="Empty input list.")
        
        if len(texts) > MAX_REQUEST_SAMPLES:
            raise HTTPException(status_code=413, detail=f"Batch too large (max {MAX_REQUEST_SAMPLES} samples).")

        preds = await run_inference_async(texts)
        
        results = [
            {"text": text, "predicted_class": int(pred)}
            for text, pred in zip(texts, preds)
        ]

        return {"batch_size": len(texts),
                "results": results}
    
    finally:
        semaphore.release()