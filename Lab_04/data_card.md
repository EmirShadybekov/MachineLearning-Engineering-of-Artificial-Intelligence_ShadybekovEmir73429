# Data Card: UCI Adult (Census Income) Dataset

**Course:** Machine Learning [CII5SE039CI], Group 03  
**Institution:** Faculty of Computer Science, Graphics and Architecture, Akademia Finansów i Biznesu Vistula w Warszawie  
**Author:** Emir Shadybekov (Album: 73429)  
**Lab Assignment:** Lab 4: Data Pipeline and Quality  
**Framework:** Standard Data Documentation Card (Reddi Chapters 5–6 / Gebru et al.)

---

## 1. Dataset Overview & Source

- **Dataset Name:** UCI Adult Dataset (also known as the "Census Income" dataset).
- **Curators:** Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics).
- **Source Repository:** UCI Machine Learning Repository / OpenML (`name='adult'`, version=2, ID: 1590).
- **Ingestion Protocol:** `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)`.
- **Extraction Origin:** 1994 United States Census Bureau database (Current Population Survey).

---

## 2. Licensing & Terms of Use

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) / Public Domain.
- **Permitted Uses:** Academic research, educational benchmarking, and non-commercial machine learning systems engineering.
- **Attribution:** Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid*. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining.

---

## 3. Dataset Composition and Dimensionality

- **Total Record Count:** Exactly 48,842 instances (combining historical 32,561 train and 16,281 test splits).
- **Total Feature Count:** 15 columns (14 input predictor features + 1 binary target label).
- **Target Variable:** `income` (`class` in OpenML representation) indicating whether an individual's annual income exceeds $50,000 USD:
  - `>50K` (Positive class, $\approx 23.93\%$)
  - `<=50K` (Negative class, $\approx 76.07\%$)
- **Feature Schema Breakdown:**
  - **Numerical (6 features):** `age` (17–90), `fnlwgt` (final sampling weight), `education-num` (1–16), `capital-gain` (0–99,999), `capital-loss` (0–4,356), `hours-per-week` (1–99).
  - **Categorical (8 features):** `workclass` (9 levels), `education` (16 levels), `marital-status` (7 levels), `occupation` (15 levels), `relationship` (6 levels), `race` (5 levels), `sex` (2 levels), `native-country` (42 levels).
- **Engineered Derived Features:**
  - `capital_net = capital-gain - capital-loss`
  - `hours_per_age = hours-per-week / (age - 16)`

---

## 4. Known Biases and Distributional Skews

1. **Historical Socioeconomic Imbalance:**  
   The data reflects 1994 US societal wage patterns, exhibiting severe demographic disparities:
   - **Gender Bias:** Approximately $66.8\%$ of records represent male heads-of-household or workers, with males disproportionately represented in the $>50\text{K}$ income bracket ($\approx 30.4\%$ male $>50\text{K}$ vs $\approx 10.9\%$ female $>50\text{K}$).
   - **Racial Skew:** Over $85.5\%$ of records represent White individuals, leading to significant under-representation of minority demographics.
2. **Capital Gain Ceiling Effect:**  
   `capital-gain` possesses an artificial recording threshold capped at \$99,999, creating an extreme point mass at the boundary.
3. **Class Imbalance:**  
   The binary target exhibits an approximately $3.18:1$ majority skew toward the $<=\$50\text{K}$ bracket, mandating the use of macro-averaged F1 and stratified partitioning.

---

## 5. Privacy, Anonymity, and Ethical Concerns

- **PII Protection:** Direct personal identifiers (Social Security Numbers, names, exact addresses) were stripped by the US Census Bureau prior to release.
- **Sampling Weights (`fnlwgt`):** Represents an estimated count of socioeconomic units in the population matching that demographic profile, mitigating individual re-identification risk.
- **Fairness & Regulatory Considerations:** Using sensitive protected attributes (`race`, `sex`, `native-country`) directly in predictive employment or credit underwriting models raises severe Equal Credit Opportunity Act (ECOA) and GDPR compliance issues. In production systems, such attributes must be audited for disparate impact and counterfactual fairness.

---

## 6. Provenance and Cryptographic Integrity

- **Pre-cleaning Data Ingestion Check:** Row count: 48,842 | Column count: 15.
- **Train/Test Partitioning:** 70% Train (34,189 rows), 30% Test (14,653 rows), stratified by `income`.
- **Deduplication:** 24 exact duplicates dropped strictly from the training partition; test partition preserved completely intact.
- **Cryptographic Hash:** The processed dataset array is verified using SHA-256 and stored in `results/config.yaml` and `results/processed_hash.txt` to guarantee reproducible data lineage across pipeline runs.
