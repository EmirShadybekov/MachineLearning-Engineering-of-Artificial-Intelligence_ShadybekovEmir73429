"""
Utility functions for Lab 2: A Neural Network from Scratch
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels (N,) into one-hot encoded matrix (N, num_classes)."""
    N = y.shape[0]
    one_hot = np.zeros((N, num_classes), dtype=float)
    one_hot[np.arange(N), y] = 1.0
    return one_hot


def load_and_preprocess_data(random_state: int = 42):
    """
    Step 1: Dataset: load_digits.
    Scale inputs by dividing by 16.
    One-hot encode labels.
    Split 70/30 stratified with random_state=42.
    """
    digits = load_digits()
    X = digits.data.astype(float)  # shape (1797, 64)
    y = digits.target              # shape (1797,)
    
    # Scale inputs by dividing by 16
    X_scaled = X / 16.0
    
    # One-hot encode labels
    Y_onehot = one_hot_encode(y, num_classes=10)
    
    # Stratified 70/30 train/test split
    X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
        X_scaled, Y_onehot, y,
        test_size=0.30,
        stratify=y,
        random_state=random_state
    )
    
    return X_train, X_test, Y_train, Y_test, y_train, y_test


def train_model(model, X_train: np.ndarray, Y_train: np.ndarray,
                X_test: np.ndarray, Y_test: np.ndarray,
                lr: float = 0.01, batch_size: int = 32, epochs: int = 50,
                verbose: bool = True, model_name: str = "Model"):
    """
    Step 6: Training loop:
    - Shuffle training data at each epoch
    - Mini-batch size 32
    - 50 epochs
    - Per-epoch logging
    """
    N = X_train.shape[0]
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    if verbose:
        print(f"\nTraining: {model_name} | lr={lr} | batch_size={batch_size} | epochs={epochs}")
        
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch = Y_shuffled[start_idx:end_idx]
            
            model.forward(X_batch)
            grads = model.backward(Y_batch)
            model.update_params(grads, lr=lr)
            
        train_loss, train_acc = model.evaluate(X_train, Y_train)
        test_loss, test_acc = model.evaluate(X_test, Y_test)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"  Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.2f}%) | "
                  f"Test Loss: {test_loss:.4f} (Acc: {test_acc*100:.2f}%)")
            
    return history


def plot_loss_curves(all_histories: dict, save_path: str = "results/loss_curves.png"):
    """
    Step 6: Plot loss curves to results/loss_curves.png.
    Compares 6 configurations: learning rates [0.001, 0.01, 0.1] and widths [16, 64].
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    widths = [16, 64]
    learning_rates = [0.001, 0.01, 0.1]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    fig.suptitle("Lab 2: Training & Test Loss Curves Across Hyperparameters\n(Mini-batch size=32, 50 Epochs, Stratified 70/30 Split)",
                 fontsize=14, fontweight='bold', y=0.98)
    
    colors = {
        'train': '#1f77b4',
        'test': '#d62728'
    }
    
    for row_idx, w in enumerate(widths):
        for col_idx, lr in enumerate(learning_rates):
            ax = axes[row_idx, col_idx]
            key = f"W={w}_lr={lr}"
            
            if key in all_histories:
                hist = all_histories[key]['history']
                epochs = hist['epoch']
                
                ax.plot(epochs, hist['train_loss'], label='Train Loss', color=colors['train'], linewidth=2.0)
                ax.plot(epochs, hist['test_loss'], label='Test Loss', color=colors['test'], linewidth=2.0, linestyle='--')
                
                final_test_acc = hist['test_acc'][-1] * 100.0
                final_test_loss = hist['test_loss'][-1]
                
                ax.set_title(f"Hidden Width W={w} | Learning Rate $\\eta$={lr}\nFinal Test Acc: {final_test_acc:.2f}% (Loss: {final_test_loss:.3f})",
                             fontsize=11, fontweight='semibold')
                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper right', framealpha=0.9)
                ax.set_ylim(bottom=0.0)
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Saved] Loss curves saved to: {save_path}")


def benchmark_forward_backward(model, X_sample: np.ndarray, Y_sample: np.ndarray,
                               num_runs: int = 100) -> dict:
    """
    Step 7: Time forward and backward passes separately. Median of 100 runs.
    """
    for _ in range(10):
        model.forward(X_sample)
        model.backward(Y_sample)
        
    forward_times = []
    backward_times = []
    
    for _ in range(num_runs):
        t0 = time.perf_counter()
        model.forward(X_sample)
        t1 = time.perf_counter()
        forward_times.append(t1 - t0)
        
        t2 = time.perf_counter()
        model.backward(Y_sample)
        t3 = time.perf_counter()
        backward_times.append(t3 - t2)
        
    median_forward_ms = np.median(forward_times) * 1000.0
    median_backward_ms = np.median(backward_times) * 1000.0
    ratio = median_backward_ms / median_forward_ms if median_forward_ms > 0 else 0.0
    
    return {
        'median_forward_ms': median_forward_ms,
        'median_backward_ms': median_backward_ms,
        'mean_forward_ms': np.mean(forward_times) * 1000.0,
        'mean_backward_ms': np.mean(backward_times) * 1000.0,
        'std_forward_ms': np.std(forward_times) * 1000.0,
        'std_backward_ms': np.std(backward_times) * 1000.0,
        'ratio_backward_to_forward': ratio,
        'num_runs': num_runs
    }
