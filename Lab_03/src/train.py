"""
Lab 3: Training loop and Architectural Trade-off plots
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Trains MLP, CNN, and Attention models for 5 epochs on Fashion-MNIST using Adam (lr=1e-3, batch=128).
Plots accuracy vs. parameters and accuracy vs. latency.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from models import FashionMLP, FashionCNN, PatchAttentionModel
from dataset import get_dataloaders
from compute_costs import profile_models


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(model: nn.Module, train_loader, val_loader, test_loader, epochs: int = 5, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation evaluation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (logits.argmax(dim=-1) == y).sum().item()
                v_total += x.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Final test evaluation
    model.eval()
    t_loss, t_correct, t_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            t_loss += criterion(logits, y).item() * x.size(0)
            t_correct += (logits.argmax(dim=-1) == y).sum().item()
            t_total += x.size(0)

    test_acc = t_correct / t_total
    test_loss = t_loss / t_total
    print(f"  -> Final Test Accuracy: {test_acc*100:.2f}% (Test Loss: {test_loss:.4f})")
    return test_acc, test_loss, history


def plot_tradeoffs(summary_df: pd.DataFrame, save_path: str):
    """
    Step 5: Plot accuracy against parameters and against latency.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {'MLP': '#1f77b4', 'CNN': '#2ca02c', 'Attention': '#d62728'}
    markers = {'MLP': 'o', 'CNN': 's', 'Attention': '^'}

    # 1. Accuracy vs Parameters
    ax1 = axes[0]
    for _, row in summary_df.iterrows():
        m = row['Model']
        ax1.scatter(row['Parameters'] / 1e3, row['Test Accuracy (%)'],
                    color=colors[m], marker=markers[m], s=120, label=m)
        ax1.annotate(f"{m}\n({row['Parameters']//1000}k)",
                     (row['Parameters'] / 1e3, row['Test Accuracy (%)']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
    ax1.set_xlabel("Parameters (Thousands, k)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy vs. Parameter Count", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Accuracy vs Latency (Batch 1)
    ax2 = axes[1]
    for _, row in summary_df.iterrows():
        m = row['Model']
        ax2.scatter(row['Latency B1 (ms)'], row['Test Accuracy (%)'],
                    color=colors[m], marker=markers[m], s=120, label=m)
        ax2.annotate(f"{m}\n({row['Latency B1 (ms)']:.3f} ms)",
                     (row['Latency B1 (ms)'], row['Test Accuracy (%)']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
    ax2.set_xlabel("Single-Sample Latency (ms)", fontsize=11)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy vs. Batch 1 Latency", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Accuracy vs Latency (Batch 128)
    ax3 = axes[2]
    for _, row in summary_df.iterrows():
        m = row['Model']
        ax3.scatter(row['Latency B128 (ms)'], row['Test Accuracy (%)'],
                    color=colors[m], marker=markers[m], s=120, label=m)
        ax3.annotate(f"{m}\n({row['Latency B128 (ms)']:.2f} ms)",
                     (row['Latency B128 (ms)'], row['Test Accuracy (%)']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
    ax3.set_xlabel("Batch 128 Latency (ms)", fontsize=11)
    ax3.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax3.set_title("Accuracy vs. Batch 128 Latency", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Saved] Trade-off plots saved to: {save_path}")


def run_training_and_tradeoffs():
    set_seed(42)
    print("\n[Task 1] Preparing Fashion-MNIST DataLoaders (Batch size 128)...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, seed=42)

    # Profiling
    print("\n[Task 3] Profiling FLOPs, Parameters, and Inference Latencies...")
    cost_df = profile_models()

    # Training
    print("\n[Task 4] Training MLP, CNN, and Attention Models (5 Epochs, Adam lr=1e-3)...")
    models = {
        "MLP": FashionMLP(),
        "CNN": FashionCNN(),
        "Attention": PatchAttentionModel()
    }

    accuracies = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        test_acc, test_loss, _ = train_single_model(model, train_loader, val_loader, test_loader, epochs=5, lr=1e-3)
        accuracies[name] = round(test_acc * 100.0, 2)

    # Merge results
    cost_df["Test Accuracy (%)"] = cost_df["Model"].map(accuracies)

    # Save results table
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(curr_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "architecture_comparison.csv")
    cost_df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Comprehensive architectural comparison saved to: {csv_path}")

    # Plot trade-offs
    plot_path = os.path.join(results_dir, "tradeoffs.png")
    plot_tradeoffs(cost_df, save_path=plot_path)

    print("\n" + "=" * 80)
    print(" LAB 3 FINAL ARCHITECTURAL COMPARISON TABLE")
    print("=" * 80)
    print(cost_df.to_string(index=False))


if __name__ == "__main__":
    run_training_and_tradeoffs()
