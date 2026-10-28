"""
Lab 4: Leakage-free Data Pipeline & Feature Engineering
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Task 4 & 5:
- 70/30 Stratified split before cleaning
- Fit imputer on train only (mode for categorical, median for numeric)
- Outlier clipping to 1st/99th percentiles from train only
- Drop duplicates from train only (keep test untouched)
- Feature engineering: capital_net and hours_per_age
- Evaluate LogisticRegression with and without engineered features
- Save preprocessor with joblib.dump
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class LeakageFreePreprocessor:
    """
    Leakage-free transformer fitted strictly on training data:
    1. Replaces '?' with NaN and imputes median (numeric) / mode (categorical).
    2. Clips numeric features to 1st and 99th percentiles derived from train.
    3. Computes engineered features:
       - capital_net = capital-gain - capital-loss
       - hours_per_age = hours-per-week / (age - 16)
    4. One-hot encodes categoricals and scales numerics.
    """
    def __init__(self, include_engineered_features: bool = True):
        self.include_engineered_features = include_engineered_features
        self.numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        self.cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        
        self.impute_values = {}
        self.clip_bounds = {}
        self.scaler = None
        self.encoder = None
        self.fitted = False

    def _clean_missing_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = df.copy()
        for col in self.cat_cols:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype(str).str.strip()
                clean_df[col] = clean_df[col].replace({'?': np.nan, '': np.nan})
        return clean_df

    def fit(self, X_train: pd.DataFrame):
        X = self._clean_missing_symbols(X_train)
        
        # 1. Compute imputation statistics on train only
        for col in self.numeric_cols:
            self.impute_values[col] = float(X[col].median())
        for col in self.cat_cols:
            mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'Missing'
            self.impute_values[col] = mode_val

        # 2. Impute training set temporarily to compute percentiles
        X_imp = X.copy()
        for col in self.numeric_cols:
            X_imp[col] = X_imp[col].fillna(self.impute_values[col])

        # 3. Compute outlier clipping bounds (1st and 99th percentiles) on train only
        for col in self.numeric_cols:
            p1 = float(np.percentile(X_imp[col], 1))
            p99 = float(np.percentile(X_imp[col], 99))
            self.clip_bounds[col] = (p1, p99)

        # 4. Fit OneHotEncoder and Scaler
        X_trans = self.transform(X_train, for_fit=True)
        self.fitted = True
        return self

    def transform(self, X_in: pd.DataFrame, for_fit: bool = False) -> np.ndarray:
        X = self._clean_missing_symbols(X_in)

        # 1. Impute using train statistics
        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.impute_values[col])
        for col in self.cat_cols:
            X[col] = X[col].fillna(self.impute_values[col])

        # 2. Clip outliers using train bounds
        for col in self.numeric_cols:
            p1, p99 = self.clip_bounds[col]
            X[col] = np.clip(X[col], p1, p99)

        # 3. Engineer features if enabled
        num_feature_cols = list(self.numeric_cols)
        if self.include_engineered_features:
            X['capital_net'] = X['capital-gain'] - X['capital-loss']
            denom = np.maximum(1.0, X['age'] - 16.0)
            X['hours_per_age'] = X['hours-per-week'] / denom
            num_feature_cols.extend(['capital_net', 'hours_per_age'])

        # 4. Scale numerics
        if for_fit:
            self.scaler = StandardScaler()
            scaled_num = self.scaler.fit_transform(X[num_feature_cols])
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_cat = self.encoder.fit_transform(X[self.cat_cols])
        else:
            scaled_num = self.scaler.transform(X[num_feature_cols])
            encoded_cat = self.encoder.transform(X[self.cat_cols])

        return np.hstack([scaled_num, encoded_cat])


def build_and_evaluate_pipeline(df: pd.DataFrame, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    
    # Target encoding: >50K is positive class 1
    income_col = 'income' if 'income' in df.columns else 'class'
    target = (df[income_col].astype(str).str.contains('>50K')).astype(int)
    features = df.drop(columns=[income_col])

    print("\n[Step 4] Splitting Train/Test 70/30 stratified before cleaning...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.30, stratify=target, random_state=42
    )
    print(f"  Train samples: {len(X_train)} (70%), Test samples: {len(X_test)} (30%)")

    # Drop exact duplicates from train only (keep test untouched)
    train_combined = pd.concat([X_train, y_train], axis=1)
    train_before = len(train_combined)
    train_dedup = train_combined.drop_duplicates()
    train_after = len(train_dedup)
    dropped_dupes = train_before - train_after
    print(f"  Exact duplicates dropped from train: {dropped_dupes} (Train size: {train_before} -> {train_after})")
    print(f"  Test set preserved completely untouched: {len(X_test)} rows")

    X_train_clean = train_dedup.drop(columns=[y_train.name])
    y_train_clean = train_dedup[y_train.name].values
    y_test_clean = y_test.values

    # 1. Pipeline WITHOUT engineered features
    print("\n[Step 5] Evaluating baseline WITHOUT engineered features...")
    prep_no_fe = LeakageFreePreprocessor(include_engineered_features=False)
    X_tr_no_fe = prep_no_fe.fit(X_train_clean).transform(X_train_clean)
    X_te_no_fe = prep_no_fe.transform(X_test)

    clf_no_fe = LogisticRegression(max_iter=1000, random_state=42)
    clf_no_fe.fit(X_tr_no_fe, y_train_clean)
    preds_no_fe = clf_no_fe.predict(X_te_no_fe)

    acc_no_fe = accuracy_score(y_test_clean, preds_no_fe)
    f1_no_fe = f1_score(y_test_clean, preds_no_fe)
    f1_macro_no_fe = f1_score(y_test_clean, preds_no_fe, average='macro')
    print(f"  Without FE -> Accuracy: {acc_no_fe*100:.2f}%, F1 (binary): {f1_no_fe:.4f}, F1 (macro): {f1_macro_no_fe:.4f}")

    # 2. Pipeline WITH engineered features (capital_net, hours_per_age)
    print("\n[Step 5] Evaluating baseline WITH engineered features (capital_net, hours_per_age)...")
    prep_with_fe = LeakageFreePreprocessor(include_engineered_features=True)
    X_tr_with_fe = prep_with_fe.fit(X_train_clean).transform(X_train_clean)
    X_te_with_fe = prep_with_fe.transform(X_test)

    clf_with_fe = LogisticRegression(max_iter=1000, random_state=42)
    clf_with_fe.fit(X_tr_with_fe, y_train_clean)
    preds_with_fe = clf_with_fe.predict(X_te_with_fe)

    acc_with_fe = accuracy_score(y_test_clean, preds_with_fe)
    f1_with_fe = f1_score(y_test_clean, preds_with_fe)
    f1_macro_with_fe = f1_score(y_test_clean, preds_with_fe, average='macro')
    print(f"  With FE    -> Accuracy: {acc_with_fe*100:.2f}%, F1 (binary): {f1_with_fe:.4f}, F1 (macro): {f1_macro_with_fe:.4f}")

    # Save fitted preprocessor
    preprocessor_path = os.path.join(results_dir, "preprocessor.joblib")
    joblib.dump(prep_with_fe, preprocessor_path)
    print(f"\n[Saved] Fitted transformer saved to: {preprocessor_path}")

    # Save baseline metrics
    metrics = [
        {
            "Pipeline Configuration": "Baseline (Without Derived Features)",
            "Accuracy": round(acc_no_fe, 4),
            "F1 Score (Binary)": round(f1_no_fe, 4),
            "F1 Score (Macro)": round(f1_macro_no_fe, 4)
        },
        {
            "Pipeline Configuration": "Feature Engineered (capital_net, hours_per_age)",
            "Accuracy": round(acc_with_fe, 4),
            "F1 Score (Binary)": round(f1_with_fe, 4),
            "F1 Score (Macro)": round(f1_macro_with_fe, 4)
        }
    ]
    metrics_df = pd.DataFrame(metrics)
    csv_path = os.path.join(results_dir, "baseline_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[Saved] Comparison metrics saved to: {csv_path}")

    return {
        'metrics_df': metrics_df,
        'preprocessor': prep_with_fe,
        'X_train_clean': X_train_clean,
        'X_test': X_test,
        'y_train_clean': y_train_clean,
        'y_test_clean': y_test_clean,
        'X_tr_with_fe': X_tr_with_fe,
        'X_te_with_fe': X_te_with_fe
    }
