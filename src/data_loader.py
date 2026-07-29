"""
data_loader.py — Load, preprocess and split the crop recommendation dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "crop_recommendation.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset."""
    df = pd.read_csv(path)
    return df


def get_feature_names() -> list[str]:
    return FEATURE_COLS


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train/test sets and return scaled features.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (scaled)
    X_train_raw, X_test_raw          : unscaled DataFrames
    scaler                            : fitted StandardScaler
    le                                : fitted LabelEncoder
    classes                           : list of class names
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    return (
        X_train, X_test,
        y_train, y_test,
        X_train_raw, X_test_raw,
        scaler, le,
        list(le.classes_)
    )
