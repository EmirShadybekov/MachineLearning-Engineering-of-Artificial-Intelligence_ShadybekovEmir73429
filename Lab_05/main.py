"""
Lab 5: Frameworks and automatic differentiation - Main Execution Runner
Course: Machine Learning [CII5SE039CI], Group 03
Institution: Faculty of Computer Science, Graphics and Architecture, Vistula University
Author: Emir Shadybekov (Album: 73429)

Executes all Lab 5 requirements:
1 & 2 & 3. Scalar autograd engine (Value node) with topological backward pass.
4. Verify on ((a*b + c)**2).relu().sum() vs PyTorch and numerical gradients (< 1e-6 diff).
5. Train 2 -> 8 -> 1 model on make_moons with BCE loss, comparing speed with PyTorch (median of 5 runs).
6. Inspect PyTorch computation graph, benchmark eager vs traced/compiled, and measure non-contiguous tensor penalty.
"""

import os
import sys

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from print_versions import log_versions
from verification import verify_autograd
from tiny_model import benchmark_tiny_models
from pytorch_graph_bench import run_graph_and_contiguity_benchmarks


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" EXECUTING LAB 5: FRAMEWORKS AND AUTOMATIC DIFFERENTIATION")
    print(" Course: Machine Learning [CII5SE039CI], Group 03 - Vistula University")
    print(" Author: Emir Shadybekov (Album: 73429)")
    print("=" * 80)

    # 1. Environment & Package Audit
    print("\n[Step 0] Logging Environment and Library Versions...")
    log_versions(results_dir)

    # 2. Task 4: Autograd Verification
    print("\n[Tasks 1, 2, 3, 4] Running Autograd Engine Verification...")
    df_verif, max_pt, max_num = verify_autograd()
    df_verif.to_csv(os.path.join(results_dir, "autograd_verification.csv"), index=False)
    print(f"[Saved] Verification table saved to: {results_dir}/autograd_verification.csv")

    # 3. Task 5: Tiny Model on make_moons
    print("\n[Task 5] Training Tiny Model (2 -> 8 -> 1) on make_moons & Speed Benchmark...")
    df_tiny, speedup = benchmark_tiny_models(results_dir)

    # 4. Task 6: PyTorch Graph, Execution Modes, and Contiguity
    print("\n[Task 6] Inspecting PyTorch Graph, Execution Modes, and Contiguity Penalty...")
    df_modes, df_contig = run_graph_and_contiguity_benchmarks(results_dir)

    print("\n" + "=" * 80)
    print(" ALL LAB 5 REQUIREMENTS COMPLETED SUCCESSFULLY!")
    print(f" All graded evidence and reports saved in: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
