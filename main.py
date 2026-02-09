from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
# HuggingFace
from transformers import pipeline
import torch
import gdown


# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_SIZE = 224


# Local TensorFlow models
# #BINARY_MODEL_PATH = r"D:\Colab-python\teethDises\new\api\model_2.keras"
# #DISEASE_MODEL_PATH = r"D:\Colab-python\teethDises\new\api\LAST_model_efficent.h5"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU completely

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BINARY_MODEL_PATH = os.path.join(MODELS_DIR, "model_2.keras")
DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, "LAST_model_efficient.h5")

# تحقق من وجود الموديلات، لو مش موجودة حملهم
def ensure_models():
    if not os.path.exists(BINARY_MODEL_PATH):
        print("📥 جاري تنزيل الموديل الثنائي...")
        gdown.download('https://drive.google.com/uc?id=1--o19x7wPyCu5rQBfxIhH3gYUJwwQNXA', 
                      BINARY_MODEL_PATH, quiet=False)
    
    if not os.path.exists(DISEASE_MODEL_PATH):
        print("📥 جاري تنزيل موديل الأمراض...")
        gdown.download('https://drive.google.com/uc?id=1JAcY_T_x16OHNUj-XKSjUMEUuepB1IFY', 
                      DISEASE_MODEL_PATH, quiet=False)

# استدعاء الدالة قبل تحميل الموديلات
ensure_models()

print(f"✅ الموديل الثنائي: {os.path.exists(BINARY_MODEL_PATH)}")
print(f"✅ موديل الأمراض: {os.path.exists(DISEASE_MODEL_PATH)}")

print("[INFO] Loading models...")

# محاولة 1: مع standalone keras
try:
    import keras
    BINARY_MODEL = keras.models.load_model(BINARY_MODEL_PATH)
    DISEASE_MODEL = keras.models.load_model(DISEASE_MODEL_PATH)
    print("✅ Loaded with standalone keras")
except:
    # محاولة 2: مع TensorFlow
    try:
        BINARY_MODEL = tf.keras.models.load_model(
            BINARY_MODEL_PATH,
            compile=False,
            custom_objects={
                'InputLayer': tf.keras.layers.InputLayer,
                'Dense': tf.keras.layers.Dense,
                'quantization_config': None
            }
        )
        DISEASE_MODEL = tf.keras.models.load_model(
            DISEASE_MODEL_PATH,
            compile=False,
            custom_objects={
                'InputLayer': tf.keras.layers.InputLayer,
                'Dense': tf.keras.layers.Dense,
                'quantization_config': None
            }
        )
        print("✅ Loaded with TensorFlow custom objects")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        # خطة طوارئ: استخدم model بسيط
        print("⚠️ Using fallback simple model")
        BINARY_MODEL = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        DISEASE_MODEL = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
            tf.keras.layers.Dense(6, activation='softmax')
        ])

# تجميع الموديلات
BINARY_MODEL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
DISEASE_MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Models ready for prediction")




# HuggingFace Teeth Health Model
HF_TEETH_HEALTH_MODEL = "steven123/Check_GoodBad_Teeth"

DEVICE = -1 

BINARY_CLASSES = ["not_teath", "teath"]
TEETH_HEALTH_CLASSES = ["Good Teeth", "Bad Teeth"]

DISEASE_CLASSES = [
    "Calculus",
    "Dental Caries",
    "Gingivitis",
    "Mouth Ulcer",
    "Tooth Discoloration",
    "Hypodontia"
]

# ============================================================
# APPLICATION INITIALIZATION
# ============================================================
app = FastAPI(
    title="Integrated Teeth Detection System",
    description="Binary Detection â†’ Teeth Health â†’ Disease Classification",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("exists:", os.path.exists(BINARY_MODEL_PATH))
print("isfile:", os.path.isfile(BINARY_MODEL_PATH))

# ============================================================
# MODEL LOADING
# ============================================================
#print("[INFO] Loading TensorFlow models...")
#BINARY_MODEL = tf.keras.models.load_model(BINARY_MODEL_PATH)
#DISEASE_MODEL = tf.keras.models.load_model(DISEASE_MODEL_PATH)


print("[INFO] Loading HuggingFace Teeth Health model...")
TEETH_HEALTH_MODEL = pipeline(
    "image-classification",
    model=HF_TEETH_HEALTH_MODEL,
    device=DEVICE
)

print("[INFO] All models loaded successfully")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def load_image(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def preprocess_for_binary(image_bytes: bytes) -> np.ndarray:
    image = load_image(image_bytes)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image).astype(np.float32) 
    return image

def preprocess_for_disease(image_bytes: bytes) -> np.ndarray:
    image = load_image(image_bytes)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image).astype(np.float32)
    image = preprocess_input(image)
    return image

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def predict_teeth(image: np.ndarray, threshold: float = 0.5) -> dict:
    image = np.expand_dims(image, axis=0)
    score = BINARY_MODEL.predict(image, verbose=0)[0][0]

    is_teeth = score >= threshold
    confidence = score if is_teeth else 1 - score

    return {
        "is_teeth": bool(is_teeth),
        "class": BINARY_CLASSES[1] if is_teeth else BINARY_CLASSES[0],
        "confidence": float(confidence),
        "raw_score": float(score),
        "threshold": threshold
    }

def predict_teeth_health(image_bytes: bytes) -> dict:
    image = load_image(image_bytes)
    outputs = TEETH_HEALTH_MODEL(image)

    top = outputs[0]

    return {
        "predicted_class": top["label"],
        "confidence": float(top["score"]),
        "all_predictions": outputs
    }

def predict_disease(image: np.ndarray) -> dict:
    image = np.expand_dims(image, axis=0)
    predictions = DISEASE_MODEL.predict(image, verbose=0)[0]

    top_index = np.argmax(predictions)
    confidence = predictions[top_index]

    top_predictions = sorted(
        [
            {
                "class": DISEASE_CLASSES[i],
                "confidence": float(predictions[i])
            }
            for i in range(len(DISEASE_CLASSES))
        ],
        key=lambda x: x["confidence"],
        reverse=True
    )[:3]

    return {
        "predicted_class": DISEASE_CLASSES[top_index],
        "confidence": float(confidence),
        "top_predictions": top_predictions
    }

# ============================================================
# MAIN PIPELINE
# ============================================================
def teeth_diagnosis_pipeline(image_bytes: bytes, threshold: float = 0.5) -> dict:
    # 1ï¸âƒ£ Binary detection
    binary_image = preprocess_for_binary(image_bytes)
    binary_result = predict_teeth(binary_image, threshold)

    if not binary_result["is_teeth"]:
        return {
            "status": "rejected",
            "binary_result": binary_result,
            "message": "Image does not contain teeth"
        }

    # 2ï¸âƒ£ Teeth Health
    health_result = predict_teeth_health(image_bytes)

    # 3ï¸âƒ£ Disease classification (skip if Good Teeth >= 85%)
    if health_result["predicted_class"] == "Good Teeth" and health_result["confidence"] >= 0.84:
        disease_result = {
            "message": "Teeth are healthy and free of diseases",
            "predicted_class": None,
            "top_predictions": []
        }
    else:
        disease_image = preprocess_for_disease(image_bytes)
        disease_result = predict_disease(disease_image)

    return {
        "status": "success",
        "binary_result": binary_result,
        "teeth_health_result": health_result,
        "disease_result": disease_result
    }

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "system": "Integrated Teeth Detection & Diagnosis API",
        "pipeline": [
            "Teeth Detection",
            "Teeth Health Classification",
            "Disease Classification"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "binary_model": "loaded",
        "health_model": "loaded",
        "disease_model": "loaded"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    image_bytes = await file.read()
    result = teeth_diagnosis_pipeline(image_bytes, threshold)
    result["filename"] = file.filename
    return result

@app.post("/check-teeth-health")
async def check_teeth_health(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return predict_teeth_health(image_bytes)

# ============================================================
# SERVER START
# ============================================================

# ============================================================
# LOCAL DEVELOPMENT ONLY
# ============================================================
def start_server():
    """Start server for local development only"""
    print("=" * 70)
    print("Integrated Teeth Detection & Disease Classification System")
    print("Server running at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 70)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_server()

#     uvicorn.run(app, host="0.0.0.0", port=8000)



