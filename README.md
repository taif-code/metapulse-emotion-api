# MetaPulse Emotion API
**The Game That Feels You**

MetaPulse Emotion API is a **stateless speech emotion recognition backend service** that enables applications, games, and AI agents to understand and react to human emotions through voice.

It is designed to integrate seamlessly with:
- Real-time agents
- WebSocket pipelines
- REST-based game engines
- Future **AaaS (Agent-as-a-Service)** systems

---

## 🧠 Model Details

- **Model Type:** EncDecClassificationModel  
- **Framework:** NVIDIA NeMo  
- **Input:** Single-channel `.wav` audio  
- **Sample Rate:** 16 kHz  
- **Classes:** 3  

### Supported Emotions
- `happy`
- `sad`
- `angry`

---

## 🏗 Architecture Summary

- Mel-spectrogram preprocessing  
- Convolutional ASR encoder  
- Classification decoder with average pooling  
- Softmax output for probability estimation  

The model is restored from a `.nemo` checkpoint and runs in **CPU mode** for portability.

---

## 🔌 API Endpoints

### GET `/`
Health check endpoint.

```json
{
  "status": "Emotion API is running"
}
```
## POST /predict_emotion

Uploads an audio file and returns emotion prediction results.
# Request

- Content-Type: multipart/form-data
- Field name: file
- File type: .wav

# Response
```json
{
  "emotion": {
    "label": "happy",
    "scores": {
      "angry": 0.12,
      "happy": 0.73,
      "sad": 0.15
    }
  },
  "raw_result": {
    "angry": 0.12,
    "happy": 0.73,
    "sad": 0.15
  }
}
```
## 🎮 Demo — MetaPulse MVP

A live demo showcasing **emotion-aware gameplay**, where the game world reacts instantly to the player’s voice emotion.

🔗 **Demo Video:**  
https://drive.google.com/file/d/1iABExnpwns0PEnh-qMg7ZFjZ9g5EtNvu/view

The demo demonstrates how this API integrates with a **Unity-based game** to dynamically alter the environment based on the player’s emotional state.
