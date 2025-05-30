"""XGBoost model training for match outcome prediction."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_data import fetch
from preprocess import FeatureEngine, FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "xgb_model.json"
META_PATH = MODELS_DIR / "model_meta.json"


def train() -> None:
    data_path = fetch()

    try:
        df = pd.read_csv(data_path)
    except pd.errors.ParserError as e:
        print(f"Error: could not parse {data_path} — {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} matches")

    engine = FeatureEngine()
    X, y = engine.build_training_data(df, min_date="2000-01-01")
    print(pd.Series(y).value_counts(normalize=True))
    print(pd.Series(y).value_counts())

    if X.shape[0] == 0:
        print("Error: no valid training samples")
        sys.exit(1)

    if X.shape[1] != len(FEATURE_COLUMNS):
        print(f"Error: expected {len(FEATURE_COLUMNS)} features, got {X.shape[1]}")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print("\nTop features:")

    for name, score in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{name:<35} {score:.4f}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"Test accuracy : {acc:.4f}")
    print(f"Test log-loss : {ll:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    meta = {
        "features": FEATURE_COLUMNS,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    train()
