import base64
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from .services.predictor_service import PredictorService

ROOT = Path(__file__).resolve().parents[1]
HAMSTERS_DIR = ROOT / "docs/media/img/hamsters"
STATIC_DIR = ROOT / "app/static"

app = FastAPI()
app.mount("/hamsters", StaticFiles(directory=str(HAMSTERS_DIR)), name="hamsters")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
predictor = PredictorService()

@app.get("/")
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())

def landmarks_to_array(face_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)

def decode_image(data_url: str):
    s = data_url.split(",", 1)[-1]
    b = base64.b64decode(s)
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def hamster_url_for_label(label: str):
    names = predictor.canon_to_names.get(label, [label])
    for name in names:
        p = HAMSTERS_DIR / f"{name}.png"
        if p.exists():
            return f"/hamsters/{name}.png"
    return "/hamsters/neutral.png"

@app.post("/predict")
def predict(image: str = Body(..., embed=True)):
    bgr = decode_image(image)
    if bgr is None:
        idx = 0
        label = predictor.label_for_id(idx)
        return {"id": idx, "label": label, "hamster_url": hamster_url_for_label(label)}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        fl = res.multi_face_landmarks[0]
        coords = landmarks_to_array(fl)
        idx = predictor.predict_id(coords)
    else:
        idx = 0
    label = predictor.label_for_id(idx)
    return {"id": idx, "label": label, "hamster_url": hamster_url_for_label(label)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)