"""
Lab 4: Label Quality Audit and Dataset Hashing
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Task 6 & 7:
- Duplicate-conflict audit: find identical feature vectors with conflicting income labels
- 50-row data dictionary validity audit
- SHA256 dataset hashing and config.yaml export
"""

import os
import hashlib
import yaml
import numpy as np
import pandas as pd


def audit_label_quality(df: pd.DataFrame, results_dir: str):
    """
    Task 6:
    - Draw 50 random rows for manual inspection against data dictionary.
    - Duplicate-conflict audit: find duplicate feature rows with conflicting income labels.
    """
    print("\n[Step 6] Running Label-Quality and Conflict Audit...")
    income_col = 'income' if 'income' in df.columns else 'class'
    feature_cols = [c for c in df.columns if c != income_col]

    # Find duplicate feature vectors with conflicting labels
    # Group by all feature columns and count unique labels
    grouped = df.groupby(feature_cols)[income_col].nunique()
    conflict_feature_indices = grouped[grouped > 1]
    num_conflicts = len(conflict_feature_indices)

    print(f"  Duplicate feature combinations found: {len(df[df.duplicated(subset=feature_cols, keep=False)])} rows")
    print(f"  Distinct feature vectors with conflicting income labels (<=50K AND >50K): {num_conflicts}")

    conflict_rows = df[df.set_index(feature_cols).index.isin(conflict_feature_indices.index)].sort_values(by=feature_cols)
    conflict_path = os.path.join(results_dir, "conflict_audit.csv")
    conflict_rows.to_csv(conflict_path, index=False)
    print(f"  [Saved] Conflicting rows saved to: {conflict_path} ({len(conflict_rows)} affected rows)")

    # 50 random rows audit against data dictionary
    sample_50 = df.sample(n=50, random_state=42)
    sample_path = os.path.join(results_dir, "audit_sample_50.csv")
    sample_50.to_csv(sample_path, index=False)
    print(f"  [Saved] 50-sample audit saved to: {sample_path}")

    return {
        'num_conflicting_groups': num_conflicts,
        'affected_rows_count': len(conflict_rows),
        'sample_50': sample_50
    }


def compute_dataset_hash_and_config(processed_array: np.ndarray, results_dir: str):
    """
    Task 7: Compute SHA256 hash of processed dataset.
    Store with config YAML.
    """
    print("\n[Step 7] Computing SHA256 Hash and Generating config.yaml...")
    # Compute SHA256 of byte representation
    data_bytes = processed_array.tobytes()
    hasher = hashlib.sha256()
    hasher.update(data_bytes)
    sha256_hash = hasher.hexdigest()

    print(f"  Processed Dataset Shape: {processed_array.shape}")
    print(f"  SHA256 Hash: {sha256_hash}")

    # Write hash to text file
    hash_path = os.path.join(results_dir, "processed_hash.txt")
    with open(hash_path, "w", encoding="utf-8") as f:
        f.write(sha256_hash + "\n")
    print(f"  [Saved] SHA256 hash saved to: {hash_path}")

    # Configuration dictionary
    config = {
        'pipeline_metadata': {
            'project': 'UCI Adult Data Pipeline and Quality',
            'course': 'Machine Learning [CII5SE039CI], Group 03',
            'author': 'Emir Shadybekov (Album: 73429)',
            'institution': 'Vistula University'
        },
        'dataset': {
            'source': 'UCI Adult (OpenML name=adult, version=2)',
            'raw_shape': [48842, 15],
            'train_test_split': {
                'test_size': 0.30,
                'stratified': True,
                'random_state': 42
            }
        },
        'validation_schema': {
            'age_range': [17, 90],
            'hours_range': [1, 99],
            'capital_gain_range': [0, 99999],
            'capital_loss_range': [0, 4356],
            'null_representation': '?'
        },
        'transformations': {
            'numeric_imputation': 'median',
            'categorical_imputation': 'mode',
            'outlier_clipping': '1st and 99th percentiles (train-fitted)',
            'train_deduplication': 'exact duplicates dropped from train only',
            'engineered_features': [
                'capital_net = capital-gain - capital-loss',
                'hours_per_age = hours-per-week / (age - 16)'
            ],
            'categorical_encoding': 'OneHotEncoder(handle_unknown=ignore)',
            'numeric_scaling': 'StandardScaler'
        },
        'artifacts': {
            'processed_data_shape': list(processed_array.shape),
            'sha256_hash': sha256_hash,
            'preprocessor_file': 'results/preprocessor.joblib'
        }
    }

    config_path = os.path.join(results_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  [Saved] Configuration YAML saved to: {config_path}")

    return sha256_hash
