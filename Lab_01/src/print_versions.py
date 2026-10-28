"""
Lab 1: Environment and first system measurements
Course: Machine Learning [CII5SE039CI], Group 03
Institution: Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Task 1: Print Python and every package version, saving output to results/versions.txt.
"""

import sys
import platform
import os
import importlib.metadata


def get_package_version(pkg_name: str) -> str:
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not installed in current environment"


def main():
    lines = []
    lines.append("=" * 60)
    lines.append("MACHINE LEARNING ENVIRONMENT SPECIFICATION")
    lines.append("Course: Machine Learning [CII5SE039CI] - Vistula University")
    lines.append(f"Student: Emir Shadybekov (Album: 73429)")
    lines.append("=" * 60)
    lines.append(f"Python Version: {platform.python_version()} ({platform.architecture()[0]})")
    lines.append(f"Python Executable: {sys.executable}")
    lines.append(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    lines.append(f"Machine / Processor: {platform.machine()} - {platform.processor()}")
    lines.append("-" * 60)
    lines.append("PINNED REQUIREMENTS STATUS:")
    lines.append("-" * 60)

    pinned_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "torchinfo",
        "thop",
        "onnx",
        "onnxruntime",
        "mlflow",
        "memory-profiler",
        "psutil",
        "codecarbon",
        "fastapi",
        "uvicorn",
        "pytest",
        "httpx",
        "locust",
        "requests",
        "pyarrow",
        "joblib",
        "tqdm"
    ]

    for pkg in pinned_packages:
        ver = get_package_version(pkg)
        lines.append(f"{pkg:<20}: {ver}")

    lines.append("=" * 60)

    output_text = "\n".join(lines)
    print(output_text)

    # Save to results/versions.txt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(current_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "versions.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(f"\n[Saved] Environment versions saved to {out_path}")


if __name__ == "__main__":
    main()
