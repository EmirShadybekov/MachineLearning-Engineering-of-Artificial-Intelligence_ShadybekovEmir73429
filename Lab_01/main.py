"""
Lab 1: Main execution runner
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Runs all required Lab 1 tasks in sequence:
1. Environment version audit (src/print_versions.py -> results/versions.txt)
2. Baseline models training, benchmarking, and budget evaluation (src/train_and_benchmark.py -> results/)
"""

import sys
import subprocess
import os


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 80)
    print(" EXECUTING LAB 1: ENVIRONMENT AND FIRST SYSTEM MEASUREMENTS")
    print("=" * 80)

    lab_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(lab_dir, "src")

    # 1. Run print_versions.py
    print("\n--- [Task 1] Running src/print_versions.py ---")
    ret1 = subprocess.run([sys.executable, os.path.join(src_dir, "print_versions.py")], cwd=lab_dir)
    if ret1.returncode != 0:
        print("[ERROR] print_versions.py failed.")
        sys.exit(ret1.returncode)

    # 2. Run train_and_benchmark.py
    print("\n--- [Tasks 2, 3, 4] Running src/train_and_benchmark.py ---")
    ret2 = subprocess.run([sys.executable, os.path.join(src_dir, "train_and_benchmark.py")], cwd=lab_dir)
    if ret2.returncode != 0:
        print("[ERROR] train_and_benchmark.py failed.")
        sys.exit(ret2.returncode)

    print("\n" + "=" * 80)
    print(" LAB 1 EXECUTION COMPLETE. ALL EVIDENCE GENERATED IN results/")
    print(" Report available in Lab_01/report.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
