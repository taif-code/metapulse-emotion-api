import requests

url = "http://127.0.0.1:8000/predict_emotion"
files = {"file": open("test_audio.wav", "rb")}

res = requests.post(url, files=files)
print(res.json())
