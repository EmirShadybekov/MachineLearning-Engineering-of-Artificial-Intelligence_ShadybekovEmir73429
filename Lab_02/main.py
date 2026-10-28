"""
Lab 2: A Neural Network from Scratch for Machine Learning [CII5SE039CI], Group 03
Faculty of Computer Science, Graphics and Architecture
Vistula University (Akademia Finansów i Biznesu Vistula w Warszawie)
Author: Emir Shadybekov (Album: 73429)

Full execution script for Lab 2:
1. Dataset: load_digits, scale / 16, one-hot encode, stratified 70/30 split (random_state=42).
2. Two-hidden-layer network design (W=16 and W=64, N(0, 0.01) weights).
3. Forward pass verification (output shape (batch, 10), row sums = 1 within 1e-6).
4. Cross-entropy loss sanity check (~ln(10) = 2.3026).
5. Chain rule backpropagation and numerical gradient check (eps=1e-5, rel_error < 1e-6).
6. Mini-batch training loop (mini-batch 32, 50 epochs, lr in [0.001, 0.01, 0.1], W in [16, 64]).
   Plot loss curves to results/loss_curves.png. Record final test accuracies.
7. Time forward and backward passes (median of 100 runs). Explain matrix multiplications.
"""

import os
import sys
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local src/ is accessible
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from neural_network import TwoHiddenLayerNN, cross_entropy_loss
from utils import (
    load_and_preprocess_data,
    train_model,
    plot_loss_curves,
    benchmark_forward_backward
)


def print_and_save_versions(results_dir: str):
    """Log environment versions for Lab 2 requirement."""
    lines = [
        "=" * 60,
        "LAB 2 ENVIRONMENT & LIBRARY VERSIONS",
        f"Student: Emir Shadybekov (Album: 73429)",
        f"Python: {platform.python_version()} ({platform.architecture()[0]})",
        f"OS: {platform.system()} {platform.release()}",
        "-" * 60
    ]
    for pkg in ["numpy", "scipy", "scikit-learn", "matplotlib", "pandas"]:
        try:
            ver = importlib.metadata.version(pkg)
        except Exception:
            ver = "N/A"
        lines.append(f"{pkg:<15}: {ver}")
    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(results_dir, "versions.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 80)
    print(" LAB 2: A NEURAL NETWORK FROM SCRATCH FOR MACHINE LEARNING [CII5SE039CI]")
    print(" Vistula University | Group 03 | Student: Emir Shadybekov (73429)")
    print("=" * 80)

    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 0. Versions log
    print_and_save_versions(results_dir)

    # 1. Dataset loading and split
    print("\n" + "=" * 80)
    print(" [Step 1] Dataset: load_digits (Scale / 16, One-Hot, Stratified 70/30 split)")
    print("=" * 80)
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_and_preprocess_data(random_state=42)
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape} (Range: [{X_train.min():.1f}, {X_train.max():.1f}])")
    print(f"X_test shape:  {X_test.shape}, Y_test shape:  {Y_test.shape}")
    print(f"Stratification: Train counts = {np.bincount(y_train)}, Test counts = {np.bincount(y_test)}")
    print("[PASS] Step 1 Complete.")

    # 2. Architecture design
    print("\n" + "=" * 80)
    print(" [Step 2] Architecture Dimensional Verification (W=16 and W=64)")
    print("=" * 80)
    for W in [16, 64]:
        nn = TwoHiddenLayerNN(input_dim=64, hidden_dim=W, output_dim=10, weight_scale=0.01, random_state=42)
        print(f"Width W={W}: W1={nn.W1.shape}, b1={nn.b1.shape}, W2={nn.W2.shape}, b2={nn.b2.shape}, W3={nn.W3.shape}, b3={nn.b3.shape}")
        assert nn.W1.shape == (64, W) and nn.b1.shape == (W,)
        assert nn.W2.shape == (W, W) and nn.b2.shape == (W,)
        assert nn.W3.shape == (W, 10) and nn.b3.shape == (10,)
    print("[PASS] Step 2 Complete: Weight and bias shapes are dimensionally correct.")

    # 3. Forward pass verification
    print("\n" + "=" * 80)
    print(" [Step 3] Forward Pass: Shape & Probability Normalization Checks")
    print("=" * 80)
    test_nn = TwoHiddenLayerNN(input_dim=64, hidden_dim=16, output_dim=10, random_state=42)
    for bs in [1, 16, 32, len(X_test)]:
        batch_out = test_nn.forward(X_test[:bs])
        row_sums = np.sum(batch_out, axis=1)
        max_dev = np.max(np.abs(row_sums - 1.0))
        print(f"  Batch {bs:4d}: Shape = {batch_out.shape}, Max deviation from 1.0 = {max_dev:.2e}")
        assert batch_out.shape == (bs, 10)
        assert max_dev < 1e-6, "Row sum exceeds 1e-6 tolerance!"
    print("[PASS] Step 3 Complete: Forward pass outputs valid probabilities summing to 1 within 1e-6.")

    # 4. Cross-entropy loss sanity check
    print("\n" + "=" * 80)
    print(" [Step 4] Cross-Entropy Loss & Untrained Sanity Check")
    print("=" * 80)
    expected_loss = np.log(10.0)  # ~2.302585
    untrained_nn = TwoHiddenLayerNN(input_dim=64, hidden_dim=64, output_dim=10, weight_scale=0.01, random_state=42)
    probs_untrained = untrained_nn.forward(X_train)
    loss_untrained = cross_entropy_loss(probs_untrained, Y_train)
    print(f"Theoretical uniform prior loss: ln(10) = {expected_loss:.6f}")
    print(f"Actual untrained network loss:           = {loss_untrained:.6f}")
    print(f"Difference:                              = {abs(loss_untrained - expected_loss):.6f}")
    assert abs(loss_untrained - expected_loss) < 0.05
    print("[PASS] Step 4 Complete: Untrained cross-entropy loss matches ln(10) = 2.3026.")

    # 5. Gradient checking
    print("\n" + "=" * 80)
    print(" [Step 5] Numerical Gradient Checking (Nudge ±1e-5, Relative Error < 1e-6)")
    print("=" * 80)
    grad_nn = TwoHiddenLayerNN(input_dim=64, hidden_dim=16, output_dim=10, weight_scale=0.01, random_state=42)
    grad_res = grad_nn.check_numerical_gradients(X_train[:32], Y_train[:32], num_checks=10, eps=1e-5, tol=1e-6, sample_seed=23)
    
    print(f"{'Param':<6} | {'Index':<12} | {'Analytic Slope':<18} | {'Numerical Slope':<18} | {'Rel Error':<12} | {'Status'}")
    print("-" * 80)
    for r in grad_res['results']:
        st = "PASS" if r['passed'] else "FAIL"
        print(f"{r['param']:<6} | {str(r['index']):<12} | {r['analytic']:<+18.10e} | {r['numeric']:<+18.10e} | {r['rel_error']:<12.4e} | [{st}]")
    print("-" * 80)
    print(f"Passed: {grad_res['passed_count']} of {grad_res['num_checks']} (Requirement: >= 9 of 10)")
    assert grad_res['all_passed_threshold'], f"Gradient check failed: {grad_res['passed_count']}/10 passed."
    print("[PASS] Step 5 Complete: Gradients successfully verified with numerical gradient checking!")

    # 6. Training loop across hyperparameters
    print("\n" + "=" * 80)
    print(" [Step 6] Model Training (6 configurations: W in [16, 64], lr in [0.001, 0.01, 0.1], 50 Epochs)")
    print("=" * 80)
    widths = [16, 64]
    learning_rates = [0.001, 0.01, 0.1]
    all_experiments = {}
    summary_rows = []

    for W in widths:
        for lr in learning_rates:
            config_name = f"W={W}_lr={lr}"
            model = TwoHiddenLayerNN(input_dim=64, hidden_dim=W, output_dim=10, weight_scale=0.01, random_state=42)
            hist = train_model(
                model=model,
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                lr=lr, batch_size=32, epochs=50,
                verbose=True,
                model_name=config_name
            )
            final_train_acc = hist['train_acc'][-1] * 100.0
            final_test_acc = hist['test_acc'][-1] * 100.0
            final_train_loss = hist['train_loss'][-1]
            final_test_loss = hist['test_loss'][-1]
            
            all_experiments[config_name] = {
                'model': model,
                'history': hist
            }
            summary_rows.append({
                "Hidden Width (W)": W,
                "Learning Rate (eta)": lr,
                "Train Loss": round(final_train_loss, 4),
                "Train Accuracy (%)": round(final_train_acc, 2),
                "Test Loss": round(final_test_loss, 4),
                "Test Accuracy (%)": round(final_test_acc, 2)
            })

    # Save loss curves plot
    curves_path = os.path.join(results_dir, "loss_curves.png")
    plot_loss_curves(all_experiments, save_path=curves_path)

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(results_dir, "training_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[Saved] Training summary table saved to: {summary_csv_path}")

    print("\n" + "=" * 80)
    print(" STEP 6 RESULTS TABLE: PERFORMANCE ACROSS HYPERPARAMETERS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("[PASS] Step 6 Complete.")

    # 7. Timing benchmark
    print("\n" + "=" * 80)
    print(" [Step 7] Timing Forward & Backward Passes (Median of 100 Runs, Batch Size 32)")
    print("=" * 80)
    timing_rows = []
    X_bench = X_train[:32]
    Y_bench = Y_train[:32]

    for W in widths:
        b_model = TwoHiddenLayerNN(input_dim=64, hidden_dim=W, output_dim=10, random_state=42)
        bench = benchmark_forward_backward(b_model, X_bench, Y_bench, num_runs=100)
        print(f"Width W={W}:")
        print(f"  Forward Pass:  {bench['median_forward_ms']:.4f} ms")
        print(f"  Backward Pass: {bench['median_backward_ms']:.4f} ms")
        print(f"  Ratio (Bwd/Fwd): {bench['ratio_backward_to_forward']:.2f}x")
        
        timing_rows.append({
            "Hidden Width (W)": W,
            "Median Forward Pass (ms)": round(bench['median_forward_ms'], 4),
            "Median Backward Pass (ms)": round(bench['median_backward_ms'], 4),
            "Backward/Forward Ratio": round(bench['ratio_backward_to_forward'], 2)
        })

    # Save timing benchmark table
    timing_df = pd.DataFrame(timing_rows)
    timing_csv_path = os.path.join(results_dir, "timing_benchmark.csv")
    timing_df.to_csv(timing_csv_path, index=False)
    print(f"\n[Saved] Timing benchmark saved to: {timing_csv_path}")

    # Plot timing bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(widths))
    width = 0.35
    fwd_vals = [r["Median Forward Pass (ms)"] for r in timing_rows]
    bwd_vals = [r["Median Backward Pass (ms)"] for r in timing_rows]
    ax.bar(x - width/2, fwd_vals, width, label='Forward Pass', color='#2ca02c')
    ax.bar(x + width/2, bwd_vals, width, label='Backward Pass', color='#d62728')
    for i, r in enumerate(timing_rows):
        ax.text(x[i] + width/2, bwd_vals[i] + 0.001, f"{r['Backward/Forward Ratio']:.2f}x",
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylabel("Median Execution Time (ms)")
    ax.set_title("Forward vs Backward Pass Latency (100 Runs, Batch Size 32)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W = {w}" for w in widths])
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.tight_layout()
    timing_plot_path = os.path.join(results_dir, "timing_breakdown.png")
    plt.savefig(timing_plot_path, dpi=300)
    plt.close()
    print(f"[Saved] Timing breakdown plot saved to: {timing_plot_path}")

    print("\n" + "=" * 80)
    print(" ALL LAB 2 EVIDENCE GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
