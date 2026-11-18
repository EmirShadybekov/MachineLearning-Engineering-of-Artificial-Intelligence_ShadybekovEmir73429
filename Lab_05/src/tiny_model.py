"""
Lab 5: Tiny Model on make_moons (2 -> 8 -> 1, Sigmoid, BCE Loss)
Course: Machine Learning [CII5SE039CI], Group 03
Author: Emir Shadybekov (Album: 73429)

Task 5:
- Dataset: make_moons(n_samples=1000, noise=0.1, random_state=42)
- Architecture: 2 -> 8 -> 1, sigmoid, BCE loss
- Train custom Value model and PyTorch model
- Benchmark training speed: 1 warm-up, then median of 5 runs
- Compare speed and plot decision boundary
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import torch.nn as nn
import torch.optim as optim
from engine import Value


# --- Custom Value Model Components ---

class Neuron:
    def __init__(self, nin: int, nonlin: bool = True):
        # Xavier/He normal-like initialization
        scale = math.sqrt(2.0 / nin)
        self.w = [Value(random.gauss(0.0, scale)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, nonlin: bool = True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class TinyCustomMLP:
    """Architecture: 2 -> 8 (ReLU) -> 1 (Linear) -> Sigmoid"""
    def __init__(self):
        self.l1 = Layer(2, 8, nonlin=True)
        self.l2 = Layer(8, 1, nonlin=False)

    def __call__(self, x):
        h = self.l1(x)
        out = self.l2(h)
        return out.sigmoid()

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


# --- PyTorch Model Components ---

class TinyPyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# --- Dataset Generation ---

def get_moon_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    return X, y


# --- Training Routines ---

def train_custom_model(X, y, epochs=25, lr=0.1, batch_size=64):
    random.seed(42)
    model = TinyCustomMLP()
    N = len(X)

    for epoch in range(epochs):
        # Mini-batch shuffling
        indices = list(range(N))
        random.shuffle(indices)

        for start_idx in range(0, N, batch_size):
            batch_ids = indices[start_idx:min(start_idx + batch_size, N)]
            
            # Forward pass
            total_loss = Value(0.0)
            B = len(batch_ids)
            for i in batch_ids:
                xi = [Value(X[i][0]), Value(X[i][1])]
                yi = y[i]
                y_pred = model(xi)
                
                # Binary Cross-Entropy Loss: - (y * log(p) + (1 - y) * log(1 - p))
                loss_i = -(yi * y_pred.log() + (1.0 - yi) * (1.0 - y_pred).log())
                total_loss = total_loss + loss_i

            batch_loss = total_loss * (1.0 / B)

            # Zero grad
            for p in model.parameters():
                p.grad = 0.0

            # Backward pass
            batch_loss.backward()

            # SGD update
            for p in model.parameters():
                p.data -= lr * p.grad

    # Evaluate final loss and accuracy
    correct = 0
    total_loss_val = 0.0
    for i in range(N):
        xi = [Value(X[i][0]), Value(X[i][1])]
        prob = model(xi).data
        pred = 1 if prob >= 0.5 else 0
        if pred == y[i]:
            correct += 1
        loss_val = -(y[i] * math.log(max(prob, 1e-15)) + (1.0 - y[i]) * math.log(max(1.0 - prob, 1e-15)))
        total_loss_val += loss_val

    accuracy = correct / N
    avg_loss = total_loss_val / N
    return model, avg_loss, accuracy


def train_pytorch_model(X, y, epochs=25, lr=0.1, batch_size=64):
    torch.manual_seed(42)
    model = TinyPyTorchMLP()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    N = len(X)

    for epoch in range(epochs):
        perm = torch.randperm(N)
        for start_idx in range(0, N, batch_size):
            batch_ids = perm[start_idx:min(start_idx + batch_size, N)]
            bx = X_t[batch_ids]
            by = y_t[batch_ids]

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        all_preds = model(X_t)
        final_loss = criterion(all_preds, y_t).item()
        acc = ((all_preds >= 0.5).float() == y_t).float().mean().item()

    return model, final_loss, acc


# --- Timing Benchmark ---

def benchmark_tiny_models(results_dir: str = None):
    if results_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(curr_dir), "results")
    os.makedirs(results_dir, exist_ok=True)

    X, y = get_moon_data()
    epochs = 25
    batch_size = 64
    lr = 0.1

    print(f"Dataset: make_moons (N=1000, noise=0.1)")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # 1. Warm-up runs
    print("\nExecuting warm-up runs (1 run each)...")
    _ = train_custom_model(X, y, epochs=epochs, lr=lr, batch_size=batch_size)
    _ = train_pytorch_model(X, y, epochs=epochs, lr=lr, batch_size=batch_size)

    # 2. Benchmark 5 runs for Custom Model
    print("Benchmarking Custom Value Model (5 runs)...")
    custom_times = []
    custom_model, custom_loss, custom_acc = None, 0.0, 0.0
    for r in range(5):
        t0 = time.perf_counter()
        custom_model, custom_loss, custom_acc = train_custom_model(X, y, epochs=epochs, lr=lr, batch_size=batch_size)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        custom_times.append(elapsed)
        print(f"  Run {r+1}: {elapsed:.4f} s (Loss: {custom_loss:.4f}, Acc: {custom_acc*100:.2f}%)")

    # 3. Benchmark 5 runs for PyTorch Model
    print("Benchmarking PyTorch Model (5 runs)...")
    pytorch_times = []
    pytorch_model, pytorch_loss, pytorch_acc = None, 0.0, 0.0
    for r in range(5):
        t0 = time.perf_counter()
        pytorch_model, pytorch_loss, pytorch_acc = train_pytorch_model(X, y, epochs=epochs, lr=lr, batch_size=batch_size)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        pytorch_times.append(elapsed)
        print(f"  Run {r+1}: {elapsed:.4f} s (Loss: {pytorch_loss:.4f}, Acc: {pytorch_acc*100:.2f}%)")

    median_custom = float(np.median(custom_times))
    median_pytorch = float(np.median(pytorch_times))
    ratio = median_custom / median_pytorch

    print("-" * 80)
    print(f"Custom Model:  Median = {median_custom:.4f} s | Acc = {custom_acc*100:.2f}% | Loss = {custom_loss:.4f}")
    print(f"PyTorch Model: Median = {median_pytorch:.4f} s | Acc = {pytorch_acc*100:.2f}% | Loss = {pytorch_loss:.4f}")
    print(f"Speedup Factor: PyTorch is {ratio:.2f}x faster than pure-Python scalar autograd")

    # Save summary table
    summary_data = [
        {
            "Framework": "Custom Autograd (Value)",
            "Architecture": "2 -> 8 -> 1 (Sigmoid)",
            "Loss (BCE)": round(custom_loss, 4),
            "Accuracy (%)": round(custom_acc * 100.0, 2),
            "Median Time (s)": round(median_custom, 4),
            "Runs": "5 runs (median)",
            "Relative Speed": "1.00x (Baseline)"
        },
        {
            "Framework": "PyTorch (C++ Core)",
            "Architecture": "2 -> 8 -> 1 (Sigmoid)",
            "Loss (BCE)": round(pytorch_loss, 4),
            "Accuracy (%)": round(pytorch_acc * 100.0, 2),
            "Median Time (s)": round(median_pytorch, 4),
            "Runs": "5 runs (median)",
            "Relative Speed": f"{ratio:.2f}x faster"
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    csv_path = f"{results_dir}/tiny_model_benchmark.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"[Saved] Benchmark summary saved to: {csv_path}")

    # Plot decision boundary comparison
    plot_decision_boundaries(X, y, custom_model, pytorch_model, f"{results_dir}/decision_boundary.png")

    return df_summary, ratio


def plot_decision_boundaries(X, y, custom_model, pytorch_model, save_path):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Custom Model Decision Surface
    Z_custom = []
    for pt in grid_points:
        p = custom_model([Value(pt[0]), Value(pt[1])]).data
        Z_custom.append(p)
    Z_custom = np.array(Z_custom).reshape(xx.shape)

    ax1 = axes[0]
    ax1.contourf(xx, yy, Z_custom, levels=20, cmap=plt.cm.Spectral, alpha=0.8)
    ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral)
    ax1.set_title("Custom Value Autograd (2 -> 8 -> 1)\nDecision Boundary", fontsize=12, fontweight='bold')
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    # 2. PyTorch Model Decision Surface
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        Z_pytorch = pytorch_model(grid_tensor).numpy().reshape(xx.shape)

    ax2 = axes[1]
    ax2.contourf(xx, yy, Z_pytorch, levels=20, cmap=plt.cm.Spectral, alpha=0.8)
    ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral)
    ax2.set_title("PyTorch Autograd (2 -> 8 -> 1)\nDecision Boundary", fontsize=12, fontweight='bold')
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Decision boundary comparison plot saved to: {save_path}")


if __name__ == "__main__":
    benchmark_tiny_models()
