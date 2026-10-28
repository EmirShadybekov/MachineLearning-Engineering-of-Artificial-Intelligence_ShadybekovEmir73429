"""
Lab 3: Main execution runner
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Executes all Lab 3 tasks:
1. Dataset preparation (Fashion-MNIST, 55k train, 5k val, 10k test, batch 128)
2. Architecture definitions (MLP, CNN, Patch-Attention)
3. Cost computation (torchinfo parameters, hand FLOPs vs thop, latency b1 and b128)
4. Model training (5 epochs, Adam lr=1e-3, test accuracy)
5. Architectural trade-off plots (results/tradeoffs.png)
"""

import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from train import run_training_and_tradeoffs


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 80)
    print(" EXECUTING LAB 3: ARCHITECTURES AND THEIR COSTS")
    print(" Course: Machine Learning [CII5SE039CI], Group 03 - Vistula University")
    print(" Author: Emir Shadybekov (Album: 73429)")
    print("=" * 80)

    run_training_and_tradeoffs()

    print("\n" + "=" * 80)
    print(" LAB 3 EXECUTION COMPLETE. ALL EVIDENCE GENERATED IN results/")
    print(" Report available in Lab_03/report.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
