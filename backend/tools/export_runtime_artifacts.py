import base64
import gzip
import json
import math
import os
import pickle
from array import array
from typing import Any, Dict, List, Tuple


def _to_f32_le_b64(values: List[float]) -> str:
    # array('f') uses native endianness; on typical x86 it's little-endian.
    # Vercel Linux is little-endian, so this is safe for runtime.
    a = array("f", (float(v) for v in values))
    return base64.b64encode(a.tobytes()).decode("ascii")


def _dump_gz_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(here)
    model_dir = os.path.join(backend_dir, "model")

    vec_pkl = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    mod_pkl = os.path.join(model_dir, "movie_genre_model.pkl")

    if not os.path.exists(vec_pkl) or not os.path.exists(mod_pkl):
        raise SystemExit("Missing backend/model/*.pkl files; cannot export.")

    with open(vec_pkl, "rb") as f:
        vec = pickle.load(f)

    with open(mod_pkl, "rb") as f:
        model = pickle.load(f)

    vocab_raw = getattr(vec, "vocabulary_", None)
    if not isinstance(vocab_raw, dict) or not vocab_raw:
        raise SystemExit("Vectorizer missing vocabulary_.")

    vocab: Dict[str, int] = {str(k): int(v) for k, v in vocab_raw.items()}

    idf = getattr(vec, "idf_", None)
    if idf is None or len(idf) != len(vocab):
        raise SystemExit("Vectorizer missing idf_ or length mismatch.")

    n_features = len(vocab)

    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    classes = getattr(model, "classes_", None)

    if coef is None or intercept is None or classes is None:
        raise SystemExit("Model missing coef_/intercept_/classes_.")

    n_classes = len(classes)
    if coef.shape != (n_classes, n_features):
        raise SystemExit(f"Unexpected coef shape {coef.shape}; expected {(n_classes, n_features)}")

    if intercept.shape != (n_classes,):
        raise SystemExit(f"Unexpected intercept shape {intercept.shape}; expected {(n_classes,)}")

    # Heuristic matching sklearn LogisticRegression 'auto' behavior (3+ classes):
    # lbfgs/newton-cg/sag/saga => multinomial softmax; liblinear => ovr
    solver = getattr(model, "solver", "lbfgs")
    is_multinomial = (n_classes > 2) and (solver != "liblinear")

    runtime_dir = os.path.join(model_dir, "runtime")

    vectorizer_payload = {
        "version": 1,
        "n_features": n_features,
        "vocabulary": vocab,
        "idf_f32_b64": _to_f32_le_b64([float(x) for x in idf.tolist()]),
        "token_pattern": getattr(vec, "token_pattern", r"(?u)\\b\\w\\w+\\b"),
        "lowercase": bool(getattr(vec, "lowercase", True)),
        "ngram_range": list(getattr(vec, "ngram_range", (1, 1))),
        "sublinear_tf": bool(getattr(vec, "sublinear_tf", False)),
        "norm": getattr(vec, "norm", "l2"),
    }

    # Flatten coef row-major
    flat_coef: List[float] = [float(x) for x in coef.reshape(-1).tolist()]
    model_payload = {
        "version": 1,
        "n_features": n_features,
        "n_classes": n_classes,
        "classes": [str(c) for c in classes.tolist()],
        "coef_f32_b64": _to_f32_le_b64(flat_coef),
        "intercept_f32_b64": _to_f32_le_b64([float(x) for x in intercept.tolist()]),
        "solver": str(solver),
        "is_multinomial": bool(is_multinomial),
    }

    _dump_gz_json(os.path.join(runtime_dir, "vectorizer.json.gz"), vectorizer_payload)
    _dump_gz_json(os.path.join(runtime_dir, "model.json.gz"), model_payload)

    print("Exported runtime artifacts:")
    print("-", os.path.join(runtime_dir, "vectorizer.json.gz"))
    print("-", os.path.join(runtime_dir, "model.json.gz"))


if __name__ == "__main__":
    main()
