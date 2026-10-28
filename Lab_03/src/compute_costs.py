"""
Lab 3: Cost analysis (FLOPs, Parameters, Profiling, Latency Benchmarking)
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)
"""

import time
import torch
import numpy as np
import pandas as pd
from torchinfo import summary
import thop
from models import FashionMLP, FashionCNN, PatchAttentionModel


def calculate_hand_flops():
    """
    Hand-calculate FLOPs for representative layers and full networks using Ch. 4 formulas:
    - Linear: 2 * in * out
    - Conv2d: 2 * K * K * C_in * C_out * H_out * W_out
    - Attention:
      * QKV projection: 2 * seq * d_model * (3 * d_model)
      * Attention scores (Q @ K.T): 2 * seq^2 * d_model
      * Attention context (A @ V): 2 * seq^2 * d_model
      * Out projection: 2 * seq * d_model * d_model
      * FFN (Linear 1 + Linear 2): 2 * seq * d_model * d_ff + 2 * seq * d_ff * d_model
    """
    # 1. MLP Hand FLOPs
    # Linear 1: 784 -> 256
    mlp_l1 = 2 * 784 * 256
    # Linear 2: 256 -> 128
    mlp_l2 = 2 * 256 * 128
    # Linear 3: 128 -> 10
    mlp_l3 = 2 * 128 * 10
    mlp_total = mlp_l1 + mlp_l2 + mlp_l3

    # 2. CNN Hand FLOPs
    # Conv 1: 1 -> 16, K=3, H_out=28, W_out=28
    cnn_c1 = 2 * 3 * 3 * 1 * 16 * 28 * 28
    # Conv 2: 16 -> 32, K=3, H_out=14, W_out=14
    cnn_c2 = 2 * 3 * 3 * 16 * 32 * 14 * 14
    # Classifier: Linear(32 * 7 * 7, 10) = 1568 -> 10
    cnn_fc = 2 * 1568 * 10
    cnn_total = cnn_c1 + cnn_c2 + cnn_fc

    # 3. Attention Hand FLOPs (seq=16, d_model=64, d_ff=128, 2 layers)
    seq = 16
    d_model = 64
    d_ff = 128
    # Patch projection: 16 patches * (49 -> 64)
    patch_proj = 2 * seq * 49 * d_model
    # Per Transformer layer:
    qkv = 2 * seq * d_model * (3 * d_model)
    attn_scores = 2 * (seq ** 2) * d_model
    attn_context = 2 * (seq ** 2) * d_model
    out_proj = 2 * seq * d_model * d_model
    ffn = (2 * seq * d_model * d_ff) + (2 * seq * d_ff * d_model)
    per_layer = qkv + attn_scores + attn_context + out_proj + ffn
    transformer_layers = 2 * per_layer
    head = 2 * d_model * 10
    attn_total = patch_proj + transformer_layers + head

    return {
        "MLP": {
            "representative_layer": ("Linear(784, 256)", mlp_l1),
            "hand_calculated_flops": mlp_total
        },
        "CNN": {
            "representative_layer": ("Conv2d(1, 16, 3x3, H=28, W=28)", cnn_c1),
            "hand_calculated_flops": cnn_total
        },
        "Attention": {
            "representative_layer": ("QKV Projections (seq=16, d=64)", qkv),
            "hand_calculated_flops": attn_total
        }
    }


def measure_inference_latency(model: torch.nn.Module, input_shape: tuple, num_runs: int, warmup_runs: int = 5):
    """Measure inference latency using repeatable protocol."""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    return float(np.median(latencies))


def profile_models():
    models = {
        "MLP": FashionMLP(),
        "CNN": FashionCNN(),
        "Attention": PatchAttentionModel()
    }

    hand_flops = calculate_hand_flops()
    results = []

    print("\n" + "=" * 80)
    print(" MODEL PARAMETER & FLOP COST ANALYSIS")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n--- Model: {name} ---")
        
        # 1. Parameter counting with torchinfo
        info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        total_params = info.total_params
        trainable_params = info.trainable_params

        # 2. Profiling with thop
        dummy = torch.randn(1, 1, 28, 28)
        thop_macs, thop_params = thop.profile(model, inputs=(dummy,), verbose=False)
        thop_flops = thop_macs * 2.0  # 1 MAC = 2 FLOPs

        # 3. Hand-calculated FLOPs
        h_flops = hand_flops[name]["hand_calculated_flops"]
        rep_layer_name, rep_layer_flops = hand_flops[name]["representative_layer"]

        # 4. Latency measurement (batch 1: 100 runs, batch 128: 20 runs)
        lat_b1 = measure_inference_latency(model, (1, 1, 28, 28), num_runs=100, warmup_runs=10)
        lat_b128 = measure_inference_latency(model, (128, 1, 28, 28), num_runs=20, warmup_runs=5)

        print(f"Total Parameters (torchinfo): {total_params:,}")
        print(f"Representative Layer:         {rep_layer_name} -> {rep_layer_flops:,} FLOPs")
        print(f"Hand-Calculated Total FLOPs:  {h_flops:,} ({h_flops/1e6:.3f} MFLOPs)")
        print(f"thop Profiled MACs:           {int(thop_macs):,} ({thop_macs/1e6:.3f} MMACs)")
        print(f"thop Implied FLOPs (2x MACs): {int(thop_flops):,} ({thop_flops/1e6:.3f} MFLOPs)")
        print(f"Latency Batch 1 (100 runs):   {lat_b1:.4f} ms")
        print(f"Latency Batch 128 (20 runs):  {lat_b128:.4f} ms")

        gap_pct = abs(h_flops - thop_flops) / h_flops * 100.0 if h_flops > 0 else 0.0

        results.append({
            "Model": name,
            "Parameters": total_params,
            "Representative Layer": rep_layer_name,
            "Rep Layer FLOPs": rep_layer_flops,
            "Hand FLOPs": h_flops,
            "thop MACs": int(thop_macs),
            "thop FLOPs": int(thop_flops),
            "FLOPs Gap (%)": round(gap_pct, 2),
            "Latency B1 (ms)": round(lat_b1, 4),
            "Latency B128 (ms)": round(lat_b128, 4)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = profile_models()
    print("\n" + "=" * 80)
    print(" SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
