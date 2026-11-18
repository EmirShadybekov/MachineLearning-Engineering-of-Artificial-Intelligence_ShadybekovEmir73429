"""
Lab 5: Verification against PyTorch autograd and numerical finite differences.
Course: Machine Learning [CII5SE039CI], Group 03
Author: Emir Shadybekov (Album: 73429)

Task 4: Verify against PyTorch autograd and numerical gradients on:
expression = ((a * b + c) ** 2).relu().sum()
Max absolute difference must be < 1e-6.
"""

import torch
import numpy as np
import pandas as pd
from engine import Value


def compute_expression_custom(a_vals, b_vals, c_vals):
    terms = []
    for a, b, c in zip(a_vals, b_vals, c_vals):
        term = ((a * b + c) ** 2).relu()
        terms.append(term)
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def evaluate_numerical_gradient(a_data, b_data, c_data, eps=1e-6):
    """Computes central finite differences for each variable."""
    grads_a = []
    grads_b = []
    grads_c = []

    def f(a_d, b_d, c_d):
        val = 0.0
        for ai, bi, ci in zip(a_d, b_d, c_d):
            term = max(0.0, (ai * bi + ci) ** 2)
            val += term
        return val

    # Gradients for a
    for i in range(len(a_data)):
        a_plus = list(a_data); a_plus[i] += eps
        a_minus = list(a_data); a_minus[i] -= eps
        g = (f(a_plus, b_data, c_data) - f(a_minus, b_data, c_data)) / (2.0 * eps)
        grads_a.append(g)

    # Gradients for b
    for i in range(len(b_data)):
        b_plus = list(b_data); b_plus[i] += eps
        b_minus = list(b_data); b_minus[i] -= eps
        g = (f(a_data, b_plus, c_data) - f(a_data, b_minus, c_data)) / (2.0 * eps)
        grads_b.append(g)

    # Gradients for c
    for i in range(len(c_data)):
        c_plus = list(c_data); c_plus[i] += eps
        c_minus = list(c_data); c_minus[i] -= eps
        g = (f(a_data, b_data, c_plus) - f(a_data, b_data, c_minus)) / (2.0 * eps)
        grads_c.append(g)

    return grads_a, grads_b, grads_c


def verify_autograd():
    print("=" * 80)
    print(" TASK 4: VERIFICATION ON ((a * b + c) ** 2).relu().sum()")
    print("=" * 80)

    # Diverse test data covering positive, negative, and zero values
    a_raw = [-2.0, 1.5, 0.5, -1.0, 2.5]
    b_raw = [3.0, -2.0, 4.0, 1.0, -0.5]
    c_raw = [1.0, 0.5, -3.0, 2.0, 1.5]

    # 1. Custom Autograd Engine
    a_custom = [Value(x) for x in a_raw]
    b_custom = [Value(x) for x in b_raw]
    c_custom = [Value(x) for x in c_raw]

    out_custom = compute_expression_custom(a_custom, b_custom, c_custom)
    out_custom.backward()

    # 2. PyTorch Autograd (float64 for high-precision verification)
    a_torch = torch.tensor(a_raw, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b_raw, dtype=torch.float64, requires_grad=True)
    c_torch = torch.tensor(c_raw, dtype=torch.float64, requires_grad=True)

    out_torch = ((a_torch * b_torch + c_torch) ** 2).relu().sum()
    out_torch.backward()

    # 3. Numerical Gradients (Finite Differences)
    num_grad_a, num_grad_b, num_grad_c = evaluate_numerical_gradient(a_raw, b_raw, c_raw, eps=1e-6)

    # Compile Verification Table
    records = []
    max_diff_pytorch = 0.0
    max_diff_numerical = 0.0

    variables = [('a', a_custom, a_torch.grad, num_grad_a),
                 ('b', b_custom, b_torch.grad, num_grad_b),
                 ('c', c_custom, c_torch.grad, num_grad_c)]

    for var_name, custom_list, torch_grads, num_grads in variables:
        for idx in range(len(custom_list)):
            c_g = custom_list[idx].grad
            t_g = torch_grads[idx].item()
            n_g = num_grads[idx]

            diff_pt = abs(c_g - t_g)
            diff_num = abs(c_g - n_g)

            if diff_pt > max_diff_pytorch:
                max_diff_pytorch = diff_pt
            if diff_num > max_diff_numerical:
                max_diff_numerical = diff_num

            records.append({
                "Variable": f"{var_name}[{idx}]",
                "Custom Grad": round(c_g, 8),
                "PyTorch Grad": round(t_g, 8),
                "Numerical Grad": round(n_g, 8),
                "|Custom - PyTorch|": f"{diff_pt:.2e}",
                "|Custom - Numerical|": f"{diff_num:.2e}",
                "Status": "PASS" if diff_pt < 1e-6 and diff_num < 1e-6 else "FAIL"
            })

    results_df = pd.DataFrame(records)

    print(results_df.to_string(index=False))
    print("-" * 80)
    print(f"Forward Output: Custom = {out_custom.data:.8f} | PyTorch = {out_torch.item():.8f}")
    print(f"Max Absolute Difference vs PyTorch:   {max_diff_pytorch:.2e}")
    print(f"Max Absolute Difference vs Numerical: {max_diff_numerical:.2e}")

    assert max_diff_pytorch < 1e-6, f"PyTorch difference {max_diff_pytorch} exceeds 1e-6!"
    assert max_diff_numerical < 1e-6, f"Numerical difference {max_diff_numerical} exceeds 1e-6!"
    print("\n[PASS] Task 4 Complete: Gradients verified with max difference < 1e-6.")

    return results_df, max_diff_pytorch, max_diff_numerical


if __name__ == "__main__":
    df, _, _ = verify_autograd()
