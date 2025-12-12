# ==============================
# 1️⃣ Base image có CUDA + cuDNN
# ==============================
FROM python:3.10-slim


# ==============================
# 2️⃣ Cài đặt cơ bản
# ==============================
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==============================
# 3️⃣ Copy và cài requirements
# ==============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==============================
# 4️⃣ Copy source code và model
# ==============================
COPY . .

# ==============================
# 5️⃣ Expose port và chạy server
# ==============================
EXPOSE 8000

# Khởi động bằng uvicorn, auto reload tắt trong production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]