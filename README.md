# Machine Learning & Engineering of Artificial Intelligence

**Course:** Machine Learning [CII5SE039CI], Group 03  
**Institution:** Faculty of Computer Science, Graphics and Architecture, Akademia Finansów i Biznesu Vistula w Warszawie  
**Instructor:** mgr. inż. Kumar Nalinaksh  
**Student:** Emir Shadybekov (Album: 73429)  
**Prescribed Textbook:** *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems* by Vijay Janapa Reddi.

---

## Repository Structure

As required by the course laboratory manual and global rules:

```text
MachineLearning-Engineering-of-Artificial-Intelligence_ShadybekovEmir73429/
├── README.md                # Main repository documentation
├── requirements.txt         # Pinned course dependencies
├── .gitignore               # Environment, temporary files, and data/ exclusions
├── data/                    # Local datasets (ignored by git, downloaded via scripts)
├── Lab_01/                  # Lab 1: Environment and first system measurements
│   ├── report.md            # Lab 1 formal report (Goal, Method, Results, Conclusions)
│   ├── src/                 # Lab 1 source code scripts
│   │   ├── print_versions.py
│   │   ├── train_baselines.py
│   │   └── system_measurements.py
│   └── results/             # Output artifacts, tables, model dumps, versions
│       ├── versions.txt
│       ├── baseline_accuracy.csv
│       ├── system_measurements.csv
│       ├── model.joblib
│       ├── logistic_regression.joblib
│       └── random_forest.joblib
```

---

## Global Environment Setup

The course prescribes Python 3.11+ and an isolated virtual environment with pinned packages:

```bash
# Create isolated environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Laboratory Index

| Lab | Topic | Dataset | Status |
|:---:|:---|:---|:---:|
| **Lab 01** | Environment and First System Measurements | Breast Cancer Wisconsin (Diagnostic) | **Completed** |
| **Lab 02** | A Neural Network from Scratch (2 Hidden Layers) | Optical Recognition of Handwritten Digits | **Completed** |
| **Lab 03** | Architectures and Their Costs (MLP, CNN, Attention) | Fashion-MNIST | **Completed** |
| **Lab 04** | Data Pipeline and Quality (Leakage-Free, Validation, Data Card) | UCI Adult (Census Income) | **Completed** |
| **Lab 05–11** | Optimization, Deep Learning, Robustness & Fairness | Two Moons, Fashion-MNIST, UCI Adult | Upcoming |
| **Project** | End-to-end Machine Learning Systems Engineering | Fashion-MNIST / Approved Dataset | Upcoming |
