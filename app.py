from fastapi import FastAPI, UploadFile, File
import tempfile
import torch
import numpy as np
from typing import Dict
from nemo.collections.asr.models import EncDecClassificationModel

app = FastAPI()

# اسم ملف المودل
MODEL_PATH = "final_emotion_model.nemo"

print(f"🔁 Loading NeMo model from: {MODEL_PATH}")
model = EncDecClassificationModel.restore_from(MODEL_PATH)
model.eval()
model = model.to("cpu")  # لاحقاً ممكن نخليه "cuda" إذا فعلنا الـ GPU

# عدّلي الترتيب حسب التدريب لو مختلف
EMOTION_LABELS = ["angry", "happy", "sad"]


@app.get("/")
def root():
    return {"status": "Emotion API is running ✅"}


@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    # 1) نحفظ الصوت مؤقتًا كـ wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_bytes = await file.read()
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # 2) نشغّل المودل
    with torch.no_grad():
        preds = model.transcribe(paths2audio_files=[tmp_path])

    print("🔍 RAW preds from NeMo:", preds)

    # نحضر مخرجات منسّقة
    top_emotion = {}
    raw_result = {}

    if not preds:
        return {
            "emotion": top_emotion,
            "raw_result": raw_result,
        }

    first = preds[0]

    # ✳️ حالة: NeMo يرجّع tensor([class_index])
    if isinstance(first, torch.Tensor):
        class_idx = int(first.item())
        print("🔢 class_idx:", class_idx)

        if 0 <= class_idx < len(EMOTION_LABELS):
            label = EMOTION_LABELS[class_idx]
            top_emotion = {label: 1.0}
            raw_result = {lbl: (1.0 if i == class_idx else 0.0)
                          for i, lbl in enumerate(EMOTION_LABELS)}
        else:
            top_emotion = {"unknown": 1.0}
            raw_result = {"index": class_idx}

    # ✳️ حالة: ترجع نص جاهز مثل "happy"
    elif isinstance(first, str):
        top_emotion = {first: 1.0}
        raw_result = {first: 1.0}

    # ✳️ حالة: ترجع dict فيه لابيل/احتمالات
    elif isinstance(first, dict):
        print("🔍 First dict result:", first)
        if "pred_label" in first:
            label = first["pred_label"]
            top_emotion = {label: 1.0}
            raw_result = {label: 1.0}
        else:
            top_emotion = first
            raw_result = first

    else:
        # fallback
        top_emotion = {"unknown": 1.0}
        raw_result = {"raw": str(first)}

    return {
        "emotion": top_emotion,
        "raw_result": raw_result,
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
