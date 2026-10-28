"""
Lab 4: Data pipeline and quality
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Task 1 & 2: Ingestion step reads raw data without modifying it.
Log row count 48842 and column count 15.
Problem definition and metadata.
"""

import pandas as pd
from sklearn.datasets import fetch_openml


def load_raw_data() -> pd.DataFrame:
    """
    Step 1 & 2: Load UCI Adult dataset via fetch_openml(name='adult', version=2, as_frame=True).
    Reads raw data without modifying it.
    Logs row count and column count.
    """
    print("\n[Step 2] Ingesting raw UCI Adult dataset from OpenML...")
    dataset = fetch_openml(name='adult', version=2, as_frame=True)
    raw_df = dataset.frame.copy()

    # If target is named 'class', rename to 'income' for semantic clarity if needed, or maintain alias
    if 'class' in raw_df.columns and 'income' not in raw_df.columns:
        raw_df.rename(columns={'class': 'income'}, inplace=True)

    rows, cols = raw_df.shape
    print(f"  Ingested raw data shape: {rows} rows, {cols} columns")
    assert rows == 48842, f"Expected 48842 rows, got {rows}"
    assert cols == 15, f"Expected 15 columns, got {cols}"
    print(f"  [PASS] Verified exact row count: {rows} and column count: {cols}")

    return raw_df


if __name__ == "__main__":
    df = load_raw_data()
    print("Columns:", list(df.columns))
    print("Sample:\n", df.head(3))
