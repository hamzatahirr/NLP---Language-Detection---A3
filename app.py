from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import pickle
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ----- Paths -----
BASE_DIR = os.path.dirname(__file__)
SAVED_DIR = os.path.join(BASE_DIR, "saved")

VECTORIZER_PATH = os.path.join(SAVED_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(SAVED_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(SAVED_DIR, "best_model.h5") 

# ----- Load artifacts -----
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

model = joblib.load(MODEL_PATH)

# ----- Routes -----
@app.get("/", response_class=HTMLResponse)
def home():
    return templates.TemplateResponse("index.html", {"request": {}})

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextIn):
    sent = payload.text.strip()

    # Input validations
    if not sent:
        return JSONResponse({"error": "Empty input"})
    if len(sent) > 1000:
        return JSONResponse({"error": "Input too long, max 1000 characters"})
    if not any(c.isalpha() for c in sent):
        return JSONResponse({"error": "Input must contain alphabetic characters"})

    X = vectorizer.transform([sent])

    # Prediction
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        top_idx = int(np.argmax(probs, axis=1)[0])
        top_prob = float(np.max(probs))
    else:
        top_idx = int(model.predict(X)[0])
        top_prob = None

    label = label_encoder.inverse_transform([top_idx])[0]

    response = {"prediction": label}
    if top_prob is not None:
        response["confidence"] = round(top_prob * 100, 2)
    else:
        response["confidence"] = "N/A"

    return JSONResponse(response)
