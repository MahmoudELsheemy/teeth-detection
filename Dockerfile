FROM python:3.9-slim

WORKDIR /app

# تثبيت المتطلبات النظامية
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملفات المشروع
COPY requirements.txt .

# تثبيت مكتبات Python - نسخة متوافقة تماماً
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.15.0 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.36.0 \
    Pillow==10.1.0 \
    numpy==1.24.3 \
    gdown==5.1.0

# نسخ باقي الملفات
COPY . .

# تنزيل الموديلات من Google Drive
RUN mkdir -p models && \
    gdown "1--o19x7wPyCu5rQBfxIhH3gYUJwwQNXA" -O models/model_2.keras && \
    gdown "1JAcY_T_x16OHNUj-XKSjUMEUuepB1IFY" -O models/LAST_model_efficient.h5

# إعدادات البيئة
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
ENV PORT=10000

# فتح المنفذ
EXPOSE 10000

# تشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
