# -*- coding: utf-8 -*-
"""
train_models.py -- Main entry point: generate dataset, train all models, save artefacts.

Usage
-----
    python train_models.py
"""

import sys
import os
import joblib
import numpy as np
from pathlib import Path

# ── Make src importable ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_data, preprocess, FEATURE_COLS
from src.train       import train_and_evaluate, plot_model_comparison, plot_confusion_matrix

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Generate dataset if not present ───────────────────────────────────────
DATA_PATH = ROOT / "data" / "crop_recommendation.csv"
if not DATA_PATH.exists():
    print("[INFO] Generating dataset ...")
    exec(open(ROOT / "generate_dataset.py").read())

# ── 2. Load & preprocess ─────────────────────────────────────────────────────
print("\n[INFO] Loading dataset ...")
df = load_data(DATA_PATH)
print(f"   Shape  : {df.shape}")
print(f"   Crops  : {df['label'].nunique()}")
print(f"   Sample :\n{df.head(3)}\n")

(X_train, X_test,
 y_train, y_test,
 X_train_raw, X_test_raw,
 scaler, le, classes) = preprocess(df)

print(f"   Train  : {X_train.shape[0]} samples")
print(f"   Test   : {X_test.shape[0]} samples")
print(f"   Classes: {classes}\n")

# ── 3. Train & evaluate ───────────────────────────────────────────────────────
results = train_and_evaluate(
    X_train, X_test, y_train, y_test,
    classes=classes, save=True,
)

# ── 4. Save scaler & label encoder ───────────────────────────────────────────
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(le,     MODELS_DIR / "label_encoder.pkl")
print("\n[OK] Scaler and LabelEncoder saved.")

# -- 5. Summary table ---------------------------------------------------------
print("\n" + "="*55)
print("  MODEL COMPARISON SUMMARY")
print("="*55)
print(f"  {'Model':<18} {'Test Acc':>10} {'CV Acc':>10} {'CV Std':>8}")
print("  " + "-"*51)
for name, res in results.items():
    print(f"  {name:<18} {res['accuracy']*100:>9.2f}% "
          f"{res['cv_mean']*100:>9.2f}% {res['cv_std']*100:>7.2f}%")
print("="*55)

# -- 6. Feature importances (RF) -----------------------------------------------
if "feature_importances" in results["Random Forest"]:
    fi = results["Random Forest"]["feature_importances"]
    print("\n  Random Forest - Feature Importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, fi), key=lambda x: -x[1]):
        bar = "|" * int(imp * 50)
        print(f"    {feat:<12} {imp*100:>5.1f}%  {bar}")

print("\n[DONE] Training complete! Run: streamlit run app/app.py")
