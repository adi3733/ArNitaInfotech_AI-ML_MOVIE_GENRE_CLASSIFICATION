from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import base64
import gzip
import json
import math
import re
import sys
from array import array

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
# Load Vectorizer & Model (Runtime Artifacts)
# --------------------------------------------------
def _load_f32_b64(payload: str) -> array:
    raw = base64.b64decode(payload.encode("ascii"))
    a = array("f")
    a.frombytes(raw)
    # Export uses little-endian floats; Vercel Linux is little-endian.
    if sys.byteorder != "little":
        a.byteswap()
    return a


def _sigmoid(z: float) -> float:
    # Numerically-stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


class _RuntimeTfidfVectorizer:
    def __init__(
        self,
        vocabulary: dict,
        idf: array,
        token_pattern: str,
        lowercase: bool,
        ngram_range: tuple,
        sublinear_tf: bool,
        norm: str,
    ) -> None:
        self.vocabulary = vocabulary
        self.idf = idf
        self.token_re = re.compile(token_pattern)
        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.norm = norm

    def transform_one(self, text: str) -> dict:
        if self.lowercase:
            text = text.lower()

        tokens = self.token_re.findall(text)
        if not tokens:
            return {}

        min_n, max_n = self.ngram_range
        terms = []

        if min_n <= 1 <= max_n:
            terms.extend(tokens)

        if min_n <= 2 <= max_n and len(tokens) >= 2:
            terms.extend(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))

        counts: dict[int, int] = {}
        vocab = self.vocabulary
        for term in terms:
            idx = vocab.get(term)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0) + 1

        if not counts:
            return {}

        features: dict[int, float] = {}
        idf = self.idf
        l2 = 0.0
        for idx, count in counts.items():
            tf = (1.0 + math.log(count)) if self.sublinear_tf else float(count)
            val = tf * float(idf[idx])
            features[idx] = val
            l2 += val * val

        if self.norm == "l2" and l2 > 0:
            inv = 1.0 / math.sqrt(l2)
            for k in list(features.keys()):
                features[k] *= inv

        return features


class _RuntimeLogisticRegression:
    def __init__(
        self,
        classes: list,
        coef: array,
        intercept: array,
        n_features: int,
        is_multinomial: bool,
    ) -> None:
        self.classes = classes
        self.coef = coef
        self.intercept = intercept
        self.n_features = n_features
        self.is_multinomial = is_multinomial

    def predict_proba_one(self, features: dict) -> list:
        n_classes = len(self.classes)
        n_features = self.n_features
        coef = self.coef
        intercept = self.intercept

        logits = [0.0] * n_classes
        for c in range(n_classes):
            base = c * n_features
            z = float(intercept[c])
            for idx, val in features.items():
                z += float(coef[base + idx]) * float(val)
            logits[c] = z

        if self.is_multinomial:
            m = max(logits)
            exps = [math.exp(z - m) for z in logits]
            s = sum(exps)
            if s <= 0:
                return [1.0 / n_classes] * n_classes
            return [e / s for e in exps]

        # OVR-style normalization (rare here): sigmoid then normalize
        probs = [_sigmoid(z) for z in logits]
        s = sum(probs)
        if s <= 0:
            return [1.0 / n_classes] * n_classes
        return [p / s for p in probs]


try:
    runtime_dir = os.path.join(MODEL_DIR, "runtime")
    vec_path = os.path.join(runtime_dir, "vectorizer.json.gz")
    model_path = os.path.join(runtime_dir, "model.json.gz")

    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing runtime artifacts. Run backend/tools/export_runtime_artifacts.py "
            "to generate backend/model/runtime/vectorizer.json.gz and model.json.gz"
        )

    with gzip.open(vec_path, "rt", encoding="utf-8") as f:
        vec_payload = json.load(f)

    with gzip.open(model_path, "rt", encoding="utf-8") as f:
        model_payload = json.load(f)

    tfidf = _RuntimeTfidfVectorizer(
        vocabulary=vec_payload["vocabulary"],
        idf=_load_f32_b64(vec_payload["idf_f32_b64"]),
        token_pattern=vec_payload.get("token_pattern", r"(?u)\b\w\w+\b"),
        lowercase=bool(vec_payload.get("lowercase", True)),
        ngram_range=tuple(vec_payload.get("ngram_range", [1, 1])),
        sublinear_tf=bool(vec_payload.get("sublinear_tf", False)),
        norm=str(vec_payload.get("norm", "l2")),
    )

    model = _RuntimeLogisticRegression(
        classes=model_payload["classes"],
        coef=_load_f32_b64(model_payload["coef_f32_b64"]),
        intercept=_load_f32_b64(model_payload["intercept_f32_b64"]),
        n_features=int(model_payload["n_features"]),
        is_multinomial=bool(model_payload.get("is_multinomial", True)),
    )

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

    features = tfidf.transform_one(text)
    probs = model.predict_proba_one(features)
    classes = model.classes

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
