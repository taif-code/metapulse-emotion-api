from fastapi import FastAPI, UploadFile, File
import tempfile
from pathlib import Path
from typing import Dict

import torch
from nemo.collections.asr.models import EncDecClassificationModel

app = FastAPI()


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "emotion_model.nemo"

print(f" Loading NeMo model from: {MODEL_PATH}")
model = EncDecClassificationModel.restore_from(str(MODEL_PATH), map_location="cpu")
model.eval()
model = model.to("cpu") 


EMOTION_LABELS = ["angry", "happy", "sad"]


@app.get("/")
def root():
    return {"status": "Emotion API is running"}


@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)) -> Dict:
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_bytes = await file.read()
        tmp.write(audio_bytes)
        tmp_path = tmp.name

  
    try:
        with torch.no_grad():
            preds = model.transcribe(
                paths2audio_files=[tmp_path],
                logprobs=True, 
            )

        first = preds[0]
        logits_t = torch.tensor(first)
        probs = torch.softmax(logits_t, dim=-1)

        
        pred_idx = int(torch.argmax(probs))
        top_label = EMOTION_LABELS[pred_idx]

        scores = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}

    except TypeError:
       
        with torch.no_grad():
            label_str = model.transcribe(paths2audio_files=[tmp_path])[0]

        top_label = label_str
        scores = {
            lbl: (1.0 if lbl == label_str else 0.0)
            for lbl in EMOTION_LABELS
        }

    return {
        "emotion": {
            "label": top_label,
            "scores": scores,  
        },
        "raw_result": scores,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
