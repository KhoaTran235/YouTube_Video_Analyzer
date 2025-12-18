# ==============================
# 1️⃣ Base image có CUDA + cuDNN
# ==============================
FROM python:3.10-slim


# ==============================
# 2️⃣ Cài đặt cơ bản
# ==============================
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/.hf_cache
ENV TRANSFORMERS_CACHE=/app/.hf_cache

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==============================
# 3️⃣ Copy và cài requirements
# ==============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HF_REPO = "khoa-tran-hcmut/sentiment_lora_bert"

hf_hub_download(
    repo_id=HF_REPO,
    filename="model.onnx",
    subfolder="onnx_lora_bert"
)

AutoTokenizer.from_pretrained(
    HF_REPO,
    subfolder="onnx_lora_bert"
)
print("✅ Model & tokenizer cached")
EOF
# ==============================
# 4️⃣ Copy source code và model
# ==============================
COPY . .

# ==============================
# 5️⃣ Expose port và chạy server
# ==============================
EXPOSE 8000

# Khởi động bằng uvicorn, auto reload tắt trong production
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]