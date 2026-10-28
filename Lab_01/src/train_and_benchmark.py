"""
Lab 1: Environment and first system measurements
Course: Machine Learning [CII5SE039CI], Group 03
Institution: Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Tasks 2, 3, and 4:
- Load Breast Cancer Wisconsin dataset
- 70/30 stratified train/test split (random_state=42)
- Train LogisticRegression(max_iter=1000, random_state=42)
- Train RandomForestClassifier(n_estimators=100, random_state=42)
- Record test accuracy to 4 decimal places in results/baseline_accuracy.csv
- Measure system costs:
    * Training time (median of 5 runs after 1 warm-up)
    * Single-sample inference latency (median of 100 runs)
    * Model size with joblib.dump (bytes and KB)
    * Peak memory with psutil.Process().memory_info().rss
- Evaluate deployment suitability against Cloud, Edge, Mobile, TinyML budgets
"""

import os
import sys
import time
import warnings
import psutil
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def get_current_memory_mb():
    """Returns the RSS memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_training_time(model_cls, model_kwargs, X_train, y_train, warmups=1, repeats=5):
    """
    Measure training time:
    - 1 warm-up run
    - median of 5 runs
    """
    # Warm-up run
    for _ in range(warmups):
        warmup_model = model_cls(**model_kwargs)
        warmup_model.fit(X_train, y_train)

    times = []
    mem_before = get_current_memory_mb()
    peak_mem = mem_before

    for _ in range(repeats):
        model = model_cls(**model_kwargs)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        curr_mem = get_current_memory_mb()
        if curr_mem > peak_mem:
            peak_mem = curr_mem

    median_time_ms = float(np.median(times) * 1000.0)
    memory_delta_mb = max(0.0, peak_mem - mem_before)
    
    return median_time_ms, peak_mem, memory_delta_mb


def measure_single_sample_inference(model, sample_x, repeats=100):
    """
    Single-sample inference:
    - repeat 100 times
    - report median latency in milliseconds
    """
    # Warm-up inference
    for _ in range(5):
        _ = model.predict(sample_x)

    times = []
    mem_before = get_current_memory_mb()
    peak_mem = mem_before

    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model.predict(sample_x)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        curr_mem = get_current_memory_mb()
        if curr_mem > peak_mem:
            peak_mem = curr_mem

    median_latency_ms = float(np.median(times) * 1000.0)
    return median_latency_ms, peak_mem


def evaluate_deployment_fit(model_name, mem_mb, latency_ms, size_kb):
    """
    Evaluate deployment fit against exact budgets:
    - Cloud:  memory >= 1 GB,     latency <= 100 ms, model size <= 500 MB
    - Edge:   memory 256-1024 MB, latency <= 50 ms,  model size <= 50 MB
    - Mobile: memory 64-256 MB,   latency <= 20 ms,  model size <= 10 MB
    - TinyML: memory <= 256 KB,   latency <= 10 ms,  model size <= 100 KB
    """
    fits = {}
    size_mb = size_kb / 1024.0

    # Cloud budget: memory >= 1 GB (1024 MB), latency <= 100 ms, size <= 500 MB
    fits['Cloud'] = {
        'suitable': (latency_ms <= 100.0 and size_mb <= 500.0),
        'reason': f"Latency {latency_ms:.3f} ms <= 100 ms, size {size_mb:.3f} MB <= 500 MB. Fits easily."
    }

    # Edge budget: memory 256-1024 MB, latency <= 50 ms, size <= 50 MB
    fits['Edge'] = {
        'suitable': (latency_ms <= 50.0 and size_mb <= 50.0),
        'reason': f"Latency {latency_ms:.3f} ms <= 50 ms, size {size_mb:.3f} MB <= 50 MB. Fits comfortably."
    }

    # Mobile budget: memory 64-256 MB, latency <= 20 ms, size <= 10 MB
    fits['Mobile'] = {
        'suitable': (latency_ms <= 20.0 and size_mb <= 10.0),
        'reason': f"Latency {latency_ms:.3f} ms <= 20 ms, size {size_mb:.3f} MB <= 10 MB. Highly suitable."
    }

    # TinyML budget: memory <= 256 KB (0.25 MB), latency <= 10 ms, size <= 100 KB
    fits['TinyML'] = {
        'suitable': (latency_ms <= 10.0 and size_kb <= 100.0),
        'reason': f"Latency {latency_ms:.3f} ms <= 10 ms; Size {size_kb:.1f} KB {'<= 100 KB (Fits!)' if size_kb <= 100.0 else '> 100 KB (Exceeds budget!)'}"
    }

    return fits


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 80)
    print(" LAB 1: ENVIRONMENT AND FIRST SYSTEM MEASUREMENTS")
    print(" Course: Machine Learning [CII5SE039CI], Group 03 - Vistula University")
    print(" Author: Emir Shadybekov (Album: 73429)")
    print("=" * 80)

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dataset Loading and Splitting
    print("\n[Step 2] Loading Breast Cancer Wisconsin Dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    print(f"Dataset shape: {X.shape} (569 samples, 30 features)")

    # 70/30 Stratified Split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    print(f"Train samples: {X_train.shape[0]} (70%), Test samples: {X_test.shape[0]} (30%)")

    # Define models
    models_config = {
        "LogisticRegression": {
            "cls": LogisticRegression,
            "kwargs": {"max_iter": 1000, "random_state": 42},
            "filename": "logistic_regression.joblib"
        },
        "RandomForestClassifier": {
            "cls": RandomForestClassifier,
            "kwargs": {"n_estimators": 100, "random_state": 42},
            "filename": "random_forest.joblib"
        }
    }

    accuracy_records = []
    system_records = []

    print("\n[Step 2 & 3] Training Models and Measuring System Costs...")
    for name, cfg in models_config.items():
        print(f"\n--- Benchmarking {name} ---")
        
        # 1. Measure Training Time (1 warmup, 5 timed runs)
        median_train_ms, peak_train_mem, mem_delta = measure_training_time(
            cfg["cls"], cfg["kwargs"], X_train, y_train, warmups=1, repeats=5
        )
        print(f"  Training Time (median of 5 runs): {median_train_ms:.3f} ms")
        print(f"  Peak Memory during training:     {peak_train_mem:.2f} MB")

        # 2. Train final model for evaluation
        final_model = cfg["cls"](**cfg["kwargs"])
        final_model.fit(X_train, y_train)

        # 3. Test Accuracy (recorded to 4 decimal places)
        y_pred = final_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_rounded = round(acc, 4)
        print(f"  Test Accuracy:                    {acc_rounded:.4f} ({acc*100:.2f}%)")
        accuracy_records.append({
            "model": name,
            "test_accuracy": f"{acc_rounded:.4f}"
        })

        # 4. Measure Single-Sample Inference Latency (100 runs)
        single_sample = X_test[0:1]
        median_inf_latency_ms, peak_inf_mem = measure_single_sample_inference(
            final_model, single_sample, repeats=100
        )
        print(f"  Single-sample Inference Latency:  {median_inf_latency_ms:.4f} ms (median of 100 runs)")

        # 5. Save Model and Measure Model Size
        model_save_path = os.path.join(results_dir, cfg["filename"])
        joblib.dump(final_model, model_save_path)
        size_bytes = os.path.getsize(model_save_path)
        size_kb = size_bytes / 1024.0
        print(f"  Model Size:                       {size_bytes} bytes ({size_kb:.2f} KB)")

        # Also save as results/model.joblib for generic requirement compliance
        if name == "LogisticRegression":
            generic_model_path = os.path.join(results_dir, "model.joblib")
            joblib.dump(final_model, generic_model_path)

        # 6. Evaluate deployment fit
        fits = evaluate_deployment_fit(name, peak_inf_mem, median_inf_latency_ms, size_kb)

        system_records.append({
            "Model": name,
            "Accuracy": f"{acc_rounded:.4f}",
            "Train Time (ms)": round(median_train_ms, 3),
            "Inference Latency (ms)": round(median_inf_latency_ms, 4),
            "Model Size (Bytes)": size_bytes,
            "Model Size (KB)": round(size_kb, 2),
            "Peak Memory (MB)": round(peak_train_mem, 2),
            "Cloud Fit": "YES" if fits['Cloud']['suitable'] else "NO",
            "Edge Fit": "YES" if fits['Edge']['suitable'] else "NO",
            "Mobile Fit": "YES" if fits['Mobile']['suitable'] else "NO",
            "TinyML Fit": "YES" if fits['TinyML']['suitable'] else "NO"
        })

    # Save baseline accuracy to results/baseline_accuracy.csv
    acc_df = pd.DataFrame(accuracy_records)
    acc_csv_path = os.path.join(results_dir, "baseline_accuracy.csv")
    acc_df.to_csv(acc_csv_path, index=False)
    print(f"\n[Saved] Baseline accuracy saved to: {acc_csv_path}")

    # Save system measurements to results/system_measurements.csv
    sys_df = pd.DataFrame(system_records)
    sys_csv_path = os.path.join(results_dir, "system_measurements.csv")
    sys_df.to_csv(sys_csv_path, index=False)
    print(f"[Saved] System measurements saved to: {sys_csv_path}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print(" SYSTEM MEASUREMENT SUMMARY TABLE")
    print("=" * 80)
    print(sys_df.to_string(index=False))

    # Generate visualization plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    model_names = [r["Model"] for r in system_records]
    
    # 1. Accuracy vs Train Time
    ax1 = axes[0]
    train_times = [r["Train Time (ms)"] for r in system_records]
    ax1.bar(model_names, train_times, color=['#1f77b4', '#ff7f0e'])
    ax1.set_ylabel("Training Time (ms)")
    ax1.set_title("Training Time (Median of 5 runs)")
    for i, v in enumerate(train_times):
        ax1.text(i, v + 1.0, f"{v:.1f} ms", ha='center', fontweight='bold')

    # 2. Inference Latency
    ax2 = axes[1]
    latencies = [r["Inference Latency (ms)"] for r in system_records]
    ax2.bar(model_names, latencies, color=['#2ca02c', '#d62728'])
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Single-Sample Inference Latency (100 runs)")
    for i, v in enumerate(latencies):
        ax2.text(i, v + 0.005, f"{v:.4f} ms", ha='center', fontweight='bold')

    # 3. Model Size
    ax3 = axes[2]
    sizes_kb = [r["Model Size (KB)"] for r in system_records]
    ax3.bar(model_names, sizes_kb, color=['#9467bd', '#8c564b'])
    ax3.set_ylabel("Model Size (KB)")
    ax3.set_title("Model Footprint (joblib)")
    for i, v in enumerate(sizes_kb):
        ax3.text(i, v + 5.0, f"{v:.1f} KB", ha='center', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "system_measurements.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Saved] Visualization plot saved to: {plot_path}")

    print("\n[Complete] Lab 1 data collection and benchmarking finished successfully!")


if __name__ == "__main__":
    main()
