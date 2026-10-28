# Lab 4: Data Pipeline and Quality

**Course:** Machine Learning [CII5SE039CI], Group 03  
**Institution:** Faculty of Computer Science, Graphics and Architecture, Akademia Finansów i Biznesu Vistula w Warszawie  
**Instructor:** mgr. inż. Kumar Nalinaksh  
**Student:** Emir Shadybekov (Album: 73429)  
**Prescribed Reading:** Vijay Janapa Reddi, *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems*, Chapter 5 AI Workflow (§§5.2–5.5) and Chapter 6 Data Engineering (§§6.2–6.8).

---

## 1. Goal

The objective of Laboratory 4 is to architect and execute a production-grade, reproducible, leakage-free data engineering pipeline for mixed numerical and categorical data using the UCI Adult (Census Income) benchmark. In machine learning systems engineering, subtle data leakage (such as fitting scalers, imputers, or outlier boundaries on pooled data before partitioning) severely inflates offline validation metrics while degrading online generalization. 

This laboratory establishes rigorous data quality controls, including:
1. Non-destructive raw ingestion and structural validation against strict data dictionary schema and range constraints.
2. Strict train/test isolation prior to data cleaning, median/mode imputation, and empirical percentile outlier clipping.
3. Training-set-only deduplication while preserving the test partition intact.
4. Domain-driven feature engineering (`capital_net` and `hours_per_age`) evaluated via a Logistic Regression baseline.
5. Systematic label-quality audits to identify duplicate feature vectors with conflicting ground-truth targets.
6. Cryptographic lineage tracking via SHA-256 dataset hashing and complete dataset documentation via `data_card.md`.

---

## 2. Method

### 2.1 Problem Definition & Ingestion Integrity
The binary classification task predicts whether an individual earns $>50\text{K}$ annually based on 14 census attributes (6 numerical, 8 categorical). Raw data was ingested via `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)` without mutation, confirming exactly 48,842 rows and 15 columns.

### 2.2 Schema and Range Validation
A declarative schema was enforced across all 15 variables:
- **Numerical Features (int):** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- **Categorical Features:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`, `income`.
- **Domain Range Checks:** `age` $\in [17, 90]$, `hours-per-week` $\in [1, 99]$, `capital-gain` $\in [0, 99999]$, and `capital-loss` $\in [0, 4356]$.
- **Null Policy Audit:** Missing values encoded as `'?'` and `NaN` were systematically audited without silently dropping rows.

### 2.3 Leakage-Free Cleaning & Transformation Pipeline
To prevent data snooping and information leakage:
1. **Stratified Split:** Data was partitioned 70/30 (34,189 train / 14,653 test) stratified by `income` prior to any cleaning operations (`random_state=42`).
2. **Train-Only Imputation:** Numerical features were imputed using training medians; categorical missing indicators (`'?'` and `NaN`) were imputed using training modes (`workclass`: *Private*, `occupation`: *Prof-specialty*, `native-country`: *United-States*).
3. **Outlier Mitigation:** Training-set 1st and 99th percentiles were computed and applied as clipping boundaries to both splits.
4. **Partition-Isolated Deduplication:** 24 exact duplicate rows were excised strictly from the training partition ($34,189 \to 34,165$ rows). The test partition was kept 100% untouched to evaluate true unseen operational distributions.
5. **Feature Engineering:**
   - $\text{capital\_net} = \text{capital-gain} - \text{capital-loss}$
   - $\text{hours\_per\_age} = \frac{\text{hours-per-week}}{\max(1.0, \text{age} - 16)}$
6. **Encoding & Standardization:** Categorical variables were transformed via `OneHotEncoder(handle_unknown='ignore')` and numerical features were standardized using `StandardScaler` (both fit strictly on training splits).

### 2.4 Label-Quality Audit & Hashing
A label-quality audit was executed across all feature vectors to detect duplicate feature rows mapped to conflicting income labels. In addition, 50 random samples were drawn and validated against domain dictionaries. The processed training tensor was cryptographically hashed using SHA-256 and registered with `config.yaml`.

---

## 3. Results Table and Quality Metrics

### 3.1 Data Validation Summary

| Feature Column | Expected Type | Actual Ingested Type | Allowed Domain Range | Observed Range | Status / Violations |
|:---|:---:|:---:|:---:|:---:|:---:|
| `age` | `int` | `int64` | $[17, 90]$ | $[17, 90]$ | Valid (0 violations) |
| `workclass` | `category` | `category` | 9 categories + `'?'` | 9 levels | 2,799 missing ($5.73\%$) |
| `fnlwgt` | `int` | `int64` | $\ge 0$ | $[12285, 1490400]$ | Valid |
| `education` | `category` | `category` | 16 categories | 16 levels | Valid (0 missing) |
| `education-num` | `int` | `int64` | $[1, 16]$ | $[1, 16]$ | Valid (0 violations) |
| `marital-status` | `category` | `category` | 7 categories | 7 levels | Valid (0 missing) |
| `occupation` | `category` | `category` | 14 categories + `'?'` | 14 levels | 2,809 missing ($5.75\%$) |
| `relationship` | `category` | `category` | 6 categories | 6 levels | Valid (0 missing) |
| `race` | `category` | `category` | 5 categories | 5 levels | Valid (0 missing) |
| `sex` | `category` | `category` | 2 categories | 2 levels | Valid (0 missing) |
| `capital-gain` | `int` | `int64` | $[0, 99999]$ | $[0, 99999]$ | Valid (0 violations) |
| `capital-loss` | `int` | `int64` | $[0, 4356]$ | $[0, 4356]$ | Valid (0 violations) |
| `hours-per-week` | `int` | `int64` | $[1, 99]$ | $[1, 99]$ | Valid (0 violations) |
| `native-country` | `category` | `category` | 41 categories + `'?'` | 41 levels | 857 missing ($1.75\%$) |
| `income` | `category` | `category` | $\{<=50\text{K}, >50\text{K}\}$ | 2 levels | Valid ($76.1\%$ vs $23.9\%$) |

*Logged in `results/validation_report.txt`.*

### 3.2 Feature Engineering Baseline Impact (Logistic Regression)
Performance on the isolated 14,653-sample test set:

| Pipeline Configuration | Accuracy | F1-Score (Binary $>50\text{K}$) | F1-Score (Macro Average) | Impact Assessment |
|:---|:---:|:---:|:---:|:---|
| **Baseline (Without Derived Features)** | 85.40% | 0.6601 | 0.7835 | Baseline standard features |
| **With Engineered Features (`capital_net`, `hours_per_age`)** | **85.78%** | **0.6696** | **0.7895** | **$+0.38\%$ Acc, $+0.0095$ F1** |

*Logged in `results/baseline_metrics.csv`.*

### 3.3 Label-Quality & Cryptographic Lineage Audit
- **Conflicting Label Feature Rows:** Identified **5 distinct feature vectors** encompassing **10 total rows** where identical personal profiles mapped to contradictory target values ($<=50\text{K}$ and $>50\text{K}$). Exported to `results/conflict_audit.csv`.
- **Deduplication:** 24 exact duplicate records removed from train only.
- **Processed Dataset Dimensionality:** $(34,165 \times 107)$ dense matrix.
- **Cryptographic SHA-256 Hash:** `c2eceb3535a5efb4f2628934b40480f5e48097fef271bd4d45eec084fb6eceb5`.
- **Configuration & Data Card:** Stored in `results/config.yaml` and `data_card.md`.

---

## 4. Three Conclusions in Your Own Words

1. **Partition Timing Dictates Data Integrity (The Zero-Leakage Imperative):**  
   Performing train/test splitting before any imputation, outlier clipping, or scaling is non-negotiable in ML systems engineering. Had median imputation or 99th-percentile clipping been computed over the entire 48,842 rows, test-set distributions would have contaminated training transformations. By fitting the preprocessor strictly on the 34,189 training records, we guaranteed that the test set remained a true unobserved benchmark. Furthermore, dropping duplicates strictly from the training split preserved the genuine real-world test distribution.

2. **Domain-Specific Feature Synthesis Significantly Enhances Linear Model Expressivity:**  
   The introduction of two derived features—`capital_net` (capturing net financial liquidity) and `hours_per_age` (capturing workforce commitment relative to adult career maturity)—measurably boosted Logistic Regression performance, increasing binary F1-score from $0.6601$ to $0.6696$ and test accuracy to $85.78\%$. Simple linear models cannot naturally capture ratio-based or differential non-linear interactions; explicit feature engineering injects critical domain signals directly into the linear decision boundary.

3. **Real-World Tabular Benchmarks Contain Inherent Label Noise and Ambiguity:**  
   Our label-quality conflict audit uncovered 5 distinct demographic profiles that simultaneously contained contradictory ground-truth labels ($<=50\text{K}$ and $>50\text{K}$). This confirms that even canonical benchmarks like UCI Adult possess epistemic label noise stemming from unobserved latent factors (e.g., geographic cost-of-living differences, unrecorded spousal assets, or survey data entry error). Production pipelines must systematically audit label conflicts rather than assuming that ground-truth labels are infallible ground truth.
