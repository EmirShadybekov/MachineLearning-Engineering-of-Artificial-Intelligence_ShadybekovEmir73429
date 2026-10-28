"""
Lab 4: Data pipeline and quality - Main Runner
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Executes complete leakage-free pipeline in one command:
1. Problem definition & ingestion (UCI Adult, 48842 rows, 15 columns)
2. Schema, range, and missing-value validation (results/validation_report.txt)
3. 70/30 stratified train/test split, imputer fitting, outlier clipping, deduplication
4. Feature engineering (capital_net, hours_per_age) & LogisticRegression benchmarking
5. Label quality audit & duplicate conflict detection (results/conflict_audit.csv)
6. SHA256 dataset hashing and config YAML export (results/config.yaml)
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ingestion import load_raw_data
from validation import validate_dataset
from pipeline import build_and_evaluate_pipeline
from quality_audit import audit_label_quality, compute_dataset_hash_and_config


def print_problem_definition():
    print("=" * 80)
    print(" LAB 4: DATA PIPELINE AND QUALITY")
    print(" Course: Machine Learning [CII5SE039CI], Group 03 - Vistula University")
    print(" Author: Emir Shadybekov (Album: 73429)")
    print("=" * 80)
    print("""
[Step 1] Half-Page Problem Definition:
--------------------------------------------------------------------------------
1. Problem Context & Goal:
   Predict whether an individual's annual income exceeds $50,000 (binary classification)
   based on 1994 US Census demographic, occupational, and financial attributes.

2. Inputs:
   14 mixed tabular features (6 continuous/integer: age, fnlwgt, education-num,
   capital-gain, capital-loss, hours-per-week; 8 categorical: workclass, education,
   marital-status, occupation, relationship, race, sex, native-country).

3. Output:
   Binary label: income >50K (positive class: 1) versus <=50K (negative class: 0).

4. Success Metrics:
   - Primary metric: F1-score (macro & binary) due to significant class imbalance (~76% <=50K vs ~24% >50K).
   - Secondary metric: Overall Classification Accuracy.

5. Critical Engineering Constraints:
   - Zero Data Leakage: All preprocessing transformations (imputer medians/modes,
     outlier percentiles, encoders, scalers) must be fit exclusively on training data.
   - Non-Destructive Ingestion: Raw source data is logged immutable; missing values
     represented as '?' are audited without silently dropping rows.
   - Reproducibility & Provenance: The processed feature representation must be
     cryptographically verified via SHA256 hashing and reproducible via one command.
--------------------------------------------------------------------------------
""")


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Problem Definition
    print_problem_definition()

    # 2. Ingestion
    raw_df = load_raw_data()

    # 3. Validation
    val_report_path = os.path.join(results_dir, "validation_report.txt")
    validate_dataset(raw_df, report_save_path=val_report_path)

    # 4 & 5. Cleaning, Transformation, Feature Engineering & Baseline Model
    pipe_results = build_and_evaluate_pipeline(raw_df, results_dir=results_dir)

    # 6. Label-Quality & Conflict Audit
    audit_label_quality(raw_df, results_dir=results_dir)

    # 7. Dataset Hashing & Config Export
    compute_dataset_hash_and_config(pipe_results['X_tr_with_fe'], results_dir=results_dir)

    print("\n" + "=" * 80)
    print(" LAB 4 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f" All graded evidence stored in: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
