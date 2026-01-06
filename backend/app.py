from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import pickle
import numpy as np

# -----------------------------
# Path Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -----------------------------
# App init
# -----------------------------
app = FastAPI(
    title="CineSense AI",
    description="Movie Genre Classification using TF-IDF + Logistic Regression",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Mount Static Files (Frontend)
# -----------------------------
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

# --------------------------------------------------
# Load Vectorizer & Model (PKL)
# --------------------------------------------------
try:
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "movie_genre_model.pkl"), "rb") as f:
        model = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"❌ Model loading failed: {e}")

# -----------------------------
# Request schema
# -----------------------------
class MoviePlot(BaseModel):
    plot: str

# -----------------------------
# Health check / Frontend
# -----------------------------
@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_genre(data: MoviePlot):

    if not data.plot or len(data.plot.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Plot text must be at least 20 characters long"
        )

    text = data.plot.lower().strip()
    features = tfidf.transform([text])

    # Get probabilities
    probs = model.predict_proba(features)[0]
    classes = model.classes_

    # Create list of dicts
    predictions = [
        {"genre": str(cls), "confidence": float(prob)}
        for cls, prob in zip(classes, probs)
    ]
    
    # Sort by confidence desc
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # Return top 5
    return {"top_predictions": predictions[:5]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
