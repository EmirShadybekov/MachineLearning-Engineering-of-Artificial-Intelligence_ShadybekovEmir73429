"""
Lab 5: Environment & Library Versions
Course: Machine Learning [CII5SE039CI], Group 03
Institution: Faculty of Computer Science, Graphics and Architecture, Vistula University
Author: Emir Shadybekov (Album: 73429)
"""

import sys
import platform
import os
import importlib.metadata


def log_versions(results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    lines = [
        "=" * 60,
        "MACHINE LEARNING LAB 5: ENVIRONMENT & LIBRARY VERSIONS",
        "Course: Machine Learning [CII5SE039CI], Group 03",
        "Student: Emir Shadybekov (Album: 73429)",
        "=" * 60,
        f"Python Version: {platform.python_version()} ({platform.architecture()[0]})",
        f"Python Executable: {sys.executable}",
        f"Operating System: {platform.system()} {platform.release()} ({platform.version()})",
        f"Machine Architecture: {platform.machine()}",
        "-" * 60,
        "CORE PACKAGES FOR LAB 5:",
        "-" * 60,
    ]

    pkgs = ["torch", "torchvision", "torchinfo", "thop", "scikit-learn", "numpy", "pandas", "matplotlib"]
    for pkg in pkgs:
        try:
            ver = importlib.metadata.version(pkg)
        except Exception:
            ver = "Not installed"
        lines.append(f"{pkg:<20}: {ver}")

    lines.append("=" * 60)
    output_text = "\n".join(lines)
    print(output_text)

    out_file = os.path.join(results_dir, "versions.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")
    print(f"\n[Saved] Versions logged to: {out_file}")


if __name__ == "__main__":
    curr = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(curr), "results")
    log_versions(res)
