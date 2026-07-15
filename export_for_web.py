"""
export_for_web.py
==================
Runs the final production model on your data and outputs a simple JSON
file (predictions_export.json) ready to upload into a Lovable web app
for the calendar-style view.

Run this from the same project folder as app.py:
    python export_for_web.py

This will generate: data/predictions_export.json
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import joblib

from feature_engineering import build_features

MODELS_DIR = "models"
DATA_PATH = "data/features_data.csv"   # change this if your file name is different
OUTPUT_PATH = "data/predictions_export.json"

# Risk thresholds (must match alert_system.py)
WARNING_THRESHOLD = 0.4
CRITICAL_THRESHOLD = 0.7


def classify(p):
    if p >= CRITICAL_THRESHOLD:
        return "critical"
    elif p >= WARNING_THRESHOLD:
        return "warning"
    return "normal"


# Simple plain-language explanations for non-technical users, based on the main contributing sensor
REASON_TEXT = {
    "smoke_level": "a noticeable rise in smoke levels",
    "temperature": "temperature rising above normal",
    "power_load": "a significant increase in electrical power load",
    "voltage_fluctuation": "abnormal fluctuation in voltage",
    "network_traffic": "unusual congestion in network traffic",
}

LEVEL_TEXT = {
    "normal": "Everything is normal, no risk indicators at this time.",
    "warning": "Some indicators need monitoring, but there is no immediate danger.",
    "critical": "High probability of a failure or fire — immediate action required.",
}


def main():
    meta = joblib.load(f"{MODELS_DIR}/production_meta.pkl")
    model = joblib.load(f"{MODELS_DIR}/production_model.pkl")
    feature_cols = meta["feature_cols"]

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = build_features(df, horizon_steps=2)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    df["risk_probability"] = model.predict_proba(df[feature_cols])[:, 1]
    df["risk_level"] = df["risk_probability"].apply(classify)

    # Compute the main contributing factor (highest relative reading among key sensors) for each row
    reason_cols = [c for c in REASON_TEXT if c in df.columns]
    norm = (df[reason_cols] - df[reason_cols].min()) / (df[reason_cols].max() - df[reason_cols].min() + 1e-9)
    df["main_reason_col"] = norm.idxmax(axis=1)

    records = []
    for _, row in df.iterrows():
        level = row["risk_level"]
        reason = REASON_TEXT.get(row["main_reason_col"], "") if level != "normal" else ""
        explanation = LEVEL_TEXT[level] + (f" Main reason: {reason}." if reason else "")
        records.append({
            "date": row["timestamp"].strftime("%Y-%m-%d"),
            "hour": int(row["timestamp"].hour),
            "risk_probability": round(float(row["risk_probability"]), 3),
            "risk_level": level,
            "explanation": explanation,
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_PATH} — total records: {len(records)}")


if __name__ == "__main__":
    main()
