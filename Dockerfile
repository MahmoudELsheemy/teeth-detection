FROM python:3.10-slim

WORKDIR /app

# تثبيت المتطلبات النظامية
RUN apt-get update && apt-get install -y wget curl && rm -rf /var/lib/apt/lists/*

# نسخ ملفات المشروع
COPY requirements.txt .
COPY . .

# تثبيت المكتبات
RUN pip install --no-cache-dir -r requirements.txt

# تنزيل الموديلات من Google Drive
RUN python -c "
import gdown
import os

os.makedirs('models', exist_ok=True)

# رابط model_2.keras
gdown.download('https://drive.google.com/uc?id=1--o19x7wPyCu5rQBfxIhH3gYUJwwQNXA', 'models/model_2.keras', quiet=False)

# رابط LAST_model_efficient.h5  
gdown.download('https://drive.google.com/uc?id=1JAcY_T_x16OHNUj-XKSjUMEUuepB1IFY', 'models/LAST_model_efficient.h5', quiet=False)

print('✅ تم تنزيل الموديلات بنجاح')
"

EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
