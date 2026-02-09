FROM python:3.10-slim

WORKDIR /app

# تثبيت متطلبات نظامية خفيفة
RUN apt-get update && apt-get install -y wget curl && rm -rf /var/lib/apt/lists/*

# نثبت الإصدارات اللي اتدربت عليها (ثبّت Keras/TF المطابقة للموديل)
COPY requirements.txt .
# ضع في requirements.txt:
# tensorflow==2.19.0
# keras==3.10.0
# fastapi==0.104.1
# uvicorn[standard]==0.24.0
# python-multipart==0.0.6
# torch==2.1.0
# torchvision==0.16.0
# transformers==4.36.0
# Pillow==10.1.0
# numpy==1.24.3
# gdown==5.1.0

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
