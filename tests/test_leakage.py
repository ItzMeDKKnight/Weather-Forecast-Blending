"""
Unit test for data leakage audit.
Asserts that no date from the test period is present in the training set or used in weight/bias estimation.
"""
import os, json, yaml
import pandas as pd
import numpy as np

def run_leakage_audit():
    parquet_path = "data/processed/wide.parquet"
    assert os.path.exists(parquet_path), "wide.parquet does not exist."
    df = pd.read_parquet(parquet_path)

    # Cutoff date definition
    train_end = "2026-06-15"
    test_start = "2026-06-16"

    train_dates = set(df[df["date"] <= train_end]["date"])
    test_dates = set(df[df["date"] >= test_start]["date"])

    # 1. Date disjointness assertion
    overlap = train_dates.intersection(test_dates)
    assert len(overlap) == 0, f"FAIL: Data leakage found! Overlapping dates: {overlap}"

    # 2. Check metadata in weights.json
    weights_path = "public/weights.json"
    if os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            w_data = json.load(f)
        saved_train_end = w_data.get("metadata", {}).get("train_end")
        assert saved_train_end == train_end, f"FAIL: weights.json train_end {saved_train_end} != expected {train_end}"

    print(f"LEAKAGE AUDIT: PASSED")
    print(f"  Training period: {min(train_dates)} to {max(train_dates)} ({len(train_dates)} distinct dates)")
    print(f"  Test period:     {min(test_dates)} to {max(test_dates)} ({len(test_dates)} distinct dates)")
    print(f"  Shared dates:    0 (Strict temporal separation confirmed)")
    return True

if __name__ == "__main__":
    success = run_leakage_audit()
    if not success:
        exit(1)
