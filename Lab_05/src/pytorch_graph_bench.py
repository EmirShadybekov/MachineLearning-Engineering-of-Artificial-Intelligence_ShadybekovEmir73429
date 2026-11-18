"""
Lab 5: PyTorch Graph Inspection, Execution Modes, and Tensor Contiguity
Course: Machine Learning [CII5SE039CI], Group 03
Author: Emir Shadybekov (Album: 73429)

Task 6:
- Inspect computation graph in PyTorch (TorchScript IR / Torch FX).
- Compare eager execution vs torch.compile or torch.jit.trace after warm-up.
- Inspect tensor shape, dtype, strides, and contiguity.
- Measure how non-contiguous memory views degrade operation speed.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BenchmarkModel(nn.Module):
    """Small neural network representing 2 -> 8 -> 1 architecture."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


def inspect_model_graph(model, dummy_input):
    """Inspects computation graph via TorchScript JIT tracing and Torch FX."""
    print("\n--- 1. Computation Graph Inspection ---")
    traced = torch.jit.trace(model, dummy_input)
    graph_ir = str(traced.graph)
    graph_code = str(traced.code)

    print("TorchScript Graph IR:")
    print("-" * 50)
    for line in graph_ir.strip().split("\n")[:15]:
        print(" ", line)
    print("  ... [truncated for readability]")
    print("-" * 50)

    return graph_ir, graph_code


def benchmark_execution_modes(model, dummy_input, num_runs=100, warmup=10):
    """
    Compares eager execution against torch.compile or torch.jit.trace / torch.jit.script.
    Uses median of 100 runs after 10 warm-up runs.
    """
    print("\n--- 2. Eager vs. Compiled / Traced Execution ---")
    model.eval()

    # 1. Eager Mode Benchmark
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

        eager_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            eager_times.append((t1 - t0) * 1000.0)  # ms
    median_eager = float(np.median(eager_times))
    print(f"Eager Execution:        Median = {median_eager:.4f} ms (over {num_runs} runs)")

    # 2. TorchScript Traced Mode
    traced_model = torch.jit.trace(model, dummy_input)
    with torch.no_grad():
        for _ in range(warmup):
            _ = traced_model(dummy_input)

        traced_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = traced_model(dummy_input)
            t1 = time.perf_counter()
            traced_times.append((t1 - t0) * 1000.0)
    median_traced = float(np.median(traced_times))
    print(f"TorchScript (JIT Trace): Median = {median_traced:.4f} ms")

    # 3. TorchScript Scripted Mode
    scripted_model = torch.jit.script(model)
    with torch.no_grad():
        for _ in range(warmup):
            _ = scripted_model(dummy_input)

        scripted_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = scripted_model(dummy_input)
            t1 = time.perf_counter()
            scripted_times.append((t1 - t0) * 1000.0)
    median_scripted = float(np.median(scripted_times))
    print(f"TorchScript (JIT Script): Median = {median_scripted:.4f} ms")

    # 4. Attempt torch.compile (with graceful fallback if unsupported on Windows/CPU)
    compiled_status = "Unsupported on current environment"
    median_compiled = None
    try:
        if hasattr(torch, "compile"):
            compiled_model = torch.compile(model, mode="default")
            with torch.no_grad():
                for _ in range(warmup):
                    _ = compiled_model(dummy_input)
                comp_times = []
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    _ = compiled_model(dummy_input)
                    t1 = time.perf_counter()
                    comp_times.append((t1 - t0) * 1000.0)
            median_compiled = float(np.median(comp_times))
            compiled_status = f"{median_compiled:.4f} ms"
            print(f"torch.compile (Inductor): Median = {compiled_status}")
    except Exception as e:
        print(f"torch.compile not fully supported on this platform ({e.__class__.__name__}), using TorchScript as prescribed.")

    # Results table
    records = [
        {"Execution Mode": "PyTorch Eager", "Median Latency (ms)": round(median_eager, 4), "Speedup vs Eager": "1.00x (Baseline)"},
        {"Execution Mode": "TorchScript (JIT Trace)", "Median Latency (ms)": round(median_traced, 4), "Speedup vs Eager": f"{median_eager / median_traced:.2f}x"},
        {"Execution Mode": "TorchScript (JIT Script)", "Median Latency (ms)": round(median_scripted, 4), "Speedup vs Eager": f"{median_eager / median_scripted:.2f}x"},
    ]
    if median_compiled is not None:
        records.append({"Execution Mode": "torch.compile (Inductor)", "Median Latency (ms)": round(median_compiled, 4), "Speedup vs Eager": f"{median_eager / median_compiled:.2f}x"})

    df_modes = pd.DataFrame(records)
    return df_modes


def inspect_tensor_representation_and_contiguity(num_runs=100, warmup=10):
    """
    Inspects tensor shape, dtype, strides, contiguity,
    and measures how a non-contiguous view degrades operation speed.
    """
    print("\n--- 3. Tensor Representation & Contiguity Benchmarks ---")

    # Allocate large 2D matrix (2048 x 2048 = 4M elements, ~16 MB in memory)
    size = 2048
    A_contig = torch.randn(size, size, dtype=torch.float32)

    # Create non-contiguous tensor via transpose
    B_non_contig = A_contig.t()

    print(f"Contiguous Tensor A:")
    print(f"  Shape:       {tuple(A_contig.shape)}")
    print(f"  Dtype:       {A_contig.dtype}")
    print(f"  Strides:     {A_contig.stride()} (Row-major: row stride = {size}, col stride = 1)")
    print(f"  Contiguous:  {A_contig.is_contiguous()}")

    print(f"\nNon-Contiguous Tensor B (Transposed view A.t()):")
    print(f"  Shape:       {tuple(B_non_contig.shape)}")
    print(f"  Dtype:       {B_non_contig.dtype}")
    print(f"  Strides:     {B_non_contig.stride()} (Column-major memory layout: row stride = 1, col stride = {size})")
    print(f"  Contiguous:  {B_non_contig.is_contiguous()}")

    assert A_contig.is_contiguous() is True
    assert B_non_contig.is_contiguous() is False

    # Benchmark: Row-wise reduction sum(dim=1)
    # Contiguous: elements along dim=1 are adjacent in memory -> maximum L1/L2 cache-line hit rate.
    # Non-contiguous: elements along dim=1 have stride 2048 -> each read incurs a cache miss!
    with torch.no_grad():
        for _ in range(warmup):
            _ = torch.sum(A_contig, dim=1)
            _ = torch.sum(B_non_contig, dim=1)

        t_contig = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = torch.sum(A_contig, dim=1)
            t1 = time.perf_counter()
            t_contig.append((t1 - t0) * 1000.0)

        t_non_contig = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = torch.sum(B_non_contig, dim=1)
            t1 = time.perf_counter()
            t_non_contig.append((t1 - t0) * 1000.0)

    med_contig = float(np.median(t_contig))
    med_non_contig = float(np.median(t_non_contig))
    slowdown = med_non_contig / med_contig

    print("\nBenchmark Results: sum(dim=1) across 2048x2048 Matrix:")
    print(f"  Contiguous Layout:     {med_contig:.4f} ms")
    print(f"  Non-Contiguous Layout: {med_non_contig:.4f} ms")
    print(f"  Performance Penalty:   Non-contiguous is {slowdown:.2f}x slower due to CPU cache striding!")

    df_contig = pd.DataFrame([
        {
            "Tensor Layout": "Contiguous (Row-major, stride=(2048, 1))",
            "is_contiguous()": True,
            "Operation": "sum(dim=1)",
            "Median Latency (ms)": round(med_contig, 4),
            "Relative Speed": "1.00x (Baseline)"
        },
        {
            "Tensor Layout": "Non-Contiguous (Transposed, stride=(1, 2048))",
            "is_contiguous()": False,
            "Operation": "sum(dim=1)",
            "Median Latency (ms)": round(med_non_contig, 4),
            "Relative Speed": f"{slowdown:.2f}x slower"
        }
    ])

    return df_contig, med_contig, med_non_contig, slowdown


def run_graph_and_contiguity_benchmarks(results_dir: str = None):
    if results_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(os.path.dirname(curr_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 80)
    print(" TASK 6: PYTORCH GRAPH INSPECTION, EXECUTION MODES & CONTIGUITY")
    print("=" * 80)

    model = BenchmarkModel()
    dummy_input = torch.randn(128, 2)

    # 1. Graph inspection
    graph_ir, graph_code = inspect_model_graph(model, dummy_input)
    with open(f"{results_dir}/pytorch_graph_ir.txt", "w", encoding="utf-8") as f:
        f.write("=== TorchScript Graph IR ===\n" + graph_ir + "\n\n=== TorchScript Code ===\n" + graph_code)
    print(f"[Saved] Graph IR written to: {results_dir}/pytorch_graph_ir.txt")

    # 2. Execution modes
    df_modes = benchmark_execution_modes(model, dummy_input, num_runs=100, warmup=10)
    df_modes.to_csv(f"{results_dir}/eager_vs_compiled.csv", index=False)
    print(f"[Saved] Execution mode comparison saved to: {results_dir}/eager_vs_compiled.csv")

    # 3. Contiguity
    df_contig, med_c, med_nc, slowdown = inspect_tensor_representation_and_contiguity(num_runs=100, warmup=10)
    df_contig.to_csv(f"{results_dir}/contiguity_benchmark.csv", index=False)
    print(f"[Saved] Contiguity benchmark saved to: {results_dir}/contiguity_benchmark.csv")

    # 4. Plot Contiguity Benchmark
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Contiguous", "Non-Contiguous\n(Transposed)"], [med_c, med_nc], color=['#2ca02c', '#d62728'], width=0.5)
    ax.set_ylabel("Execution Time (ms)", fontsize=11)
    ax.set_title("Memory Contiguity Impact on sum(dim=1)\n(2048 x 2048 Matrix, Median of 100 Runs)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f} ms", ha='center', va='bottom', fontweight='bold')
    ax.text(1, med_nc * 0.5, f"{slowdown:.2f}x SLOWER", ha='center', color='white', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plot_path = f"{results_dir}/contiguity_benchmark.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Saved] Contiguity benchmark plot saved to: {plot_path}")

    return df_modes, df_contig


if __name__ == "__main__":
    run_graph_and_contiguity_benchmarks()
