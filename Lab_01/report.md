# Lab 1: Environment and First System Measurements

**Course:** Machine Learning [CII5SE039CI], Group 03  
**Institution:** Faculty of Computer Science, Graphics and Architecture, Akademia Finansów i Biznesu Vistula w Warszawie  
**Instructor:** mgr. inż. Kumar Nalinaksh  
**Student:** Emir Shadybekov (Album: 73429)  
**Prescribed Textbook:** Vijay Janapa Reddi, *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems* (Chapters 1, 2, and 12).

---

## 1. Goal

The objective of Laboratory 1 is to establish a rigorous, reproducible Machine Learning Systems engineering environment and perform baseline empirical system measurements across competing model paradigms. Rather than treating machine learning strictly as an abstract statistical optimization task, this lab evaluates models through an engineering lens—quantifying training duration, single-sample inference latency, storage footprint, and memory consumption. Finally, empirical measurements are evaluated against standard industrial deployment budgets spanning Cloud, Edge, Mobile, and TinyML tiers to determine real-world architectural feasibility.

---

## 2. Method

### 2.1 Environment Setup and Package Verification
An isolated Python environment was created with pinned dependencies adhering to the course specification. Environment and package versions were introspected and exported to `results/versions.txt` via `src/print_versions.py`.

### 2.2 Dataset and Reproducible Partitioning
The diagnostic Breast Cancer Wisconsin dataset was loaded via `sklearn.datasets.load_breast_cancer(return_X_y=True)`. The dataset contains 569 patient records across 30 real-valued clinical features. To ensure exact reproducibility and preserve class representation, the dataset was partitioned into a 70% training set (398 samples) and a 30% test set (171 samples) using a stratified split with `random_state=42`.

### 2.3 Model Architecture Selection
Two representative paradigms were evaluated:
- **Linear Parametric Baseline:** `LogisticRegression(max_iter=1000, random_state=42)`
- **Non-Parametric Ensemble Baseline:** `RandomForestClassifier(n_estimators=100, random_state=42)`

Both models were fitted on the training split, and generalisation performance was evaluated on the unseen test split. Test accuracy was recorded to four decimal places in `results/baseline_accuracy.csv`.

### 2.4 Empirical Measurement Protocol
To eliminate timing anomalies caused by operating system scheduling, memory paging, and cache warmup, the following protocols were enforced:
- **Training Time:** 1 initial warmup fit followed by 5 consecutive runs. The median elapsed time measured using high-resolution monotonic clock (`time.perf_counter()`) was recorded.
- **Single-Sample Inference Latency:** 5 warmup inferences followed by 100 individual single-sample prediction cycles. The median latency across the 100 iterations was computed.
- **Model Storage Footprint:** Trained model artifacts were serialized to disk using `joblib.dump` and measured down to the exact byte and kibibyte (`KB`).
- **Peak Memory:** Process resident set size (`RSS`) was continuously monitored throughout execution using `psutil.Process().memory_info().rss`.

### 2.5 Deployment Budget Definitions
Measurements were mapped against four standard system tiers:
- **Cloud:** Memory $\ge 1\text{ GB}$, Latency $\le 100\text{ ms}$, Model Size $\le 500\text{ MB}$
- **Edge:** Memory $256 - 1024\text{ MB}$, Latency $\le 50\text{ ms}$, Model Size $\le 50\text{ MB}$
- **Mobile:** Memory $64 - 256\text{ MB}$, Latency $\le 20\text{ ms}$, Model Size $\le 10\text{ MB}$
- **TinyML:** Memory $\le 256\text{ KB}$, Latency $\le 10\text{ ms}$, Model Size $\le 100\text{ KB}$

---

## 3. Results Table and Deployment Analysis

### 3.1 Baseline Predictive Accuracy
Test accuracy evaluated on the stratified test split (171 samples):

| Model | Test Accuracy (4 decimals) | Percentage |
|:---|:---:|:---:|
| **Logistic Regression** | `0.9415` | 94.15% |
| **Random Forest Classifier** | `0.9357` | 93.57% |

*Recorded in `results/baseline_accuracy.csv`.*

### 3.2 System Cost Measurements Table
Detailed empirical measurements collected across training, inference, disk storage, and memory:

| System Metric | Metric Protocol | Logistic Regression | Random Forest Classifier | Delta / Ratio |
|:---|:---|:---:|:---:|:---:|
| **Test Accuracy** | Stratified 70/30 split | **0.9415** | **0.9357** | $+0.58\%$ (LogReg) |
| **Training Time** | Median of 5 runs (1 warmup) | 149.88 ms | 89.09 ms | $1.68\times$ faster (RF) |
| **Inference Latency** | Median of 100 single-sample runs | **0.0763 ms** (76.3 µs) | **3.1691 ms** (3169.1 µs) | **$41.5\times$ faster (LogReg)** |
| **Model Size (Bytes)** | `joblib.dump` binary on disk | **1,103 bytes** | **290,889 bytes** | **$263.7\times$ smaller (LogReg)** |
| **Model Size (KB)** | Disk footprint | **1.08 KB** | **284.07 KB** | $263.7\times$ smaller (LogReg) |
| **Peak Memory (MB)** | Process RSS peak | 170.80 MB | 171.93 MB | Comparable (~171 MB) |

*Recorded in `results/system_measurements.csv` and visualised in `results/system_measurements.png`.*

### 3.3 Deployment Suitability Matrix

| Deployment Tier | Resource Constraints (Memory / Latency / Size) | Logistic Regression Fit | Random Forest Fit | Technical Justification |
|:---|:---|:---:|:---:|:---|
| **Cloud** | $\ge 1\text{ GB}$ / $\le 100\text{ ms}$ / $\le 500\text{ MB}$ | **YES** | **YES** | Both models require negligible fractions of server resources and comfortably satisfy latency constraints. |
| **Edge** | $256-1024\text{ MB}$ / $\le 50\text{ ms}$ / $\le 50\text{ MB}$ | **YES** | **YES** | Both models easily meet the 50 ms latency threshold and 50 MB storage envelope of edge gateways and embedded boards. |
| **Mobile** | $64-256\text{ MB}$ / $\le 20\text{ ms}$ / $\le 10\text{ MB}$ | **YES** | **YES** | Logistic Regression (0.076 ms) and Random Forest (3.17 ms) operate well below the 20 ms interactive threshold; storage is under 1 MB. |
| **TinyML** | $\le 256\text{ KB}$ / $\le 10\text{ ms}$ / $\le 100\text{ KB}$ | **YES** | **NO** | **Logistic Regression fits:** 1.08 KB $\le$ 100 KB budget, and 0.076 ms $\le$ 10 ms.<br>**Random Forest fails:** Model footprint (284.07 KB) exceeds the 100 KB limit by $2.84\times$. |

---

## 4. Three Conclusions in Your Own Words

1. **Parametric Simplicity Grants Exponential Inference and Footprint Advantages:**  
   While Logistic Regression and Random Forest achieved comparable predictive classification accuracy on the test set (94.15% vs 93.57%), their operational profiles diverge dramatically. Logistic Regression executes single-sample inference in just 76.3 microseconds—over $41\times$ faster than Random Forest's 3.17 milliseconds. Furthermore, the linear model stores only 31 floating-point scalar weights and a bias (1.08 KB on disk), whereas the 100-tree ensemble stores extensive branch thresholds, feature indices, and leaf states totaling 284.07 KB ($263\times$ larger). When accuracy is parity, the simpler model offers vastly superior engineering economy.

2. **The Hardware Boundary Between Mobile and TinyML is Hard and Structural:**  
   Random Forest is fully qualified for Cloud, Edge, and modern Mobile devices, where megabytes of flash storage are abundant. However, the TinyML budget imposes an unforgiving ceiling: maximum 100 KB model size and 256 KB memory. Because Random Forest serialized at 284.07 KB, it is disqualified from running on microcontrollers (such as ARM Cortex-M class chips) without aggressive pruning, quantization, or tree-compression techniques. Logistic Regression, consuming barely 1.08 KB, easily fits into standard microcontroller SRAM/Flash architectures with substantial headroom.

3. **Production Model Selection Requires Holistic Systems Engineering Beyond Raw Metrics:**  
   In classical machine learning courses, models are often judged solely by validation accuracy or AUC-ROC. This lab demonstrates that in real-world artificial intelligence engineering, operational constraints (latency distribution, cold-start time, binary footprint, and target deployment tiers) dictate whether an algorithm is viable in production. System benchmarking must be integrated into the earliest phases of model development rather than treated as a post-hoc deployment afterthought.
