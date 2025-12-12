from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import asyncio
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import time
import os

MODEL_PATH = "onnx_lora_bert/model.onnx"
TOKENIZER_PATH = "onnx_lora_bert"
BATCH_LIMIT = 16   # batch limit for each inference call
MAX_LENGTH = 128   # max token length
NUM_WORKERS = 4    # number of concurrent threads
REQUEST_TIMEOUT = 15.0  # seconds

if "CUDAExecutionProvider" in ort.get_available_providers():
    provider = "CUDAExecutionProvider"
else:
    provider = "CPUExecutionProvider"

session = ort.InferenceSession(MODEL_PATH, providers=[provider])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"Model loaded on: {session.get_providers()[0]}")

executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

app = FastAPI(title="Sentiment Analysis API", version="1.0")

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

class TextsRequest(BaseModel):
    texts: list[str]

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

@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use the /predict endpoint to analyze sentiment."}    

@app.post("/predict")
async def predict_sentiment(text_request: TextsRequest):
    texts = text_request.texts
    if not texts:
        raise HTTPException(status_code=400, detail="Empty input list.")
    
    if len(texts) > 128:
        raise HTTPException(status_code=413, detail="Batch too large (max 128 samples).")

    preds = await run_inference_async(texts)
    results = [
        {"text": text, "predicted_class": int(pred)}
        for text, pred in zip(texts, preds)
    ]

    return {"batch_size": len(texts),
            "results": results}