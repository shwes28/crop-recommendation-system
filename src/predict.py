"""
predict.py — Prediction helpers for loading saved models and making inferences.
"""

import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "SVM":           "svm.pkl",
    "Naive Bayes":   "naive_bayes.pkl",
}


def load_model(name: str):
    """Load a saved model by friendly name."""
    path = MODELS_DIR / MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. Run train_models.py first."
        )
    return joblib.load(path)


def load_all_models() -> dict:
    """Load all three saved models."""
    return {name: load_model(name) for name in MODEL_FILES}


def load_scaler():
    path = MODELS_DIR / "scaler.pkl"
    if not path.exists():
        raise FileNotFoundError("Scaler not found. Run train_models.py first.")
    return joblib.load(path)


def load_encoder():
    path = MODELS_DIR / "label_encoder.pkl"
    if not path.exists():
        raise FileNotFoundError("LabelEncoder not found. Run train_models.py first.")
    return joblib.load(path)


def predict_single(
    N: float, P: float, K: float,
    temperature: float, humidity: float, ph: float, rainfall: float,
) -> dict:
    """
    Make predictions from all three models for a single input.

    Returns
    -------
    dict: {model_name: {"crop": str, "confidence": float, "probabilities": dict}}
    """
    scaler  = load_scaler()
    le      = load_encoder()
    models  = load_all_models()

    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    X_scaled = scaler.transform(X)

    output = {}
    for name, model in models.items():
        pred_idx = model.predict(X_scaled)[0]
        crop     = le.inverse_transform([pred_idx])[0]

        if hasattr(model, "predict_proba"):
            proba  = model.predict_proba(X_scaled)[0]
            conf   = proba[pred_idx]
            prob_d = {le.classes_[i]: float(p) for i, p in enumerate(proba)}
        else:
            conf   = 1.0
            prob_d = {crop: 1.0}

        output[name] = {
            "crop":          crop,
            "confidence":    float(conf),
            "probabilities": prob_d,
        }

    return output
