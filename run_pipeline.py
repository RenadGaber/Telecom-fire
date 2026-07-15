"""
run_pipeline.py — تشغيل خط الأنابيب كاملاً بأمر واحد:
1) توليد/تحميل البيانات  2) Preprocessing  3) Feature Engineering
4) تدريب ML (RF + XGBoost)  5) تدريب LSTM (اختياري)

الاستخدام:
    python run_pipeline.py                 # يشمل كل شيء ما عدا LSTM
    python run_pipeline.py --with-lstm      # يشمل تدريب LSTM أيضًا (أبطأ)
    python run_pipeline.py --raw-data path/to/your_real_data.csv
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from generate_synthetic_data import generate_synthetic_data
from preprocessing import preprocess_pipeline
from feature_engineering import build_features
from train_ml_models import run_training_pipeline
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=str, default=None, help="مسار بيانات حقيقية (اختياري)")
    parser.add_argument("--with-lstm", action="store_true", help="تدريب نموذج LSTM أيضًا")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    raw_path = args.raw_data or "data/central_sensors_data.csv"

    print("\n========== [1/5] البيانات ==========")
    if args.raw_data and os.path.exists(args.raw_data):
        print(f"استخدام بيانات حقيقية من: {args.raw_data}")
    else:
        print("لا توجد بيانات حقيقية → توليد بيانات صناعية واقعية...")
        generate_synthetic_data(out_path=raw_path)

    print("\n========== [2/5] Preprocessing ==========")
    df = preprocess_pipeline(raw_path)

    print("\n========== [3/5] Feature Engineering ==========")
    df_feat = build_features(df)
    df_feat.to_csv("data/features_data.csv", index=False)
    print(f"✅ تم حفظ data/features_data.csv | الشكل: {df_feat.shape}")

    print("\n========== [4/5] تدريب نماذج ML (RandomForest + XGBoost) ==========")
    run_training_pipeline("data/features_data.csv")

    if args.with_lstm:
        print("\n========== [5/5] تدريب LSTM ==========")
        from train_lstm import run_lstm_pipeline
        run_lstm_pipeline("data/features_data.csv")
    else:
        print("\n(تم تخطي تدريب LSTM — استخدم --with-lstm لتفعيله)")

    print("\n✅ اكتمل الـ Pipeline بالكامل. شغّل الداشبورد الآن:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
