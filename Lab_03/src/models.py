"""
Lab 3: DNN Architectures in PyTorch
Course: Machine Learning [CII5SE039CI], Group 03
Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Definitions for:
1. Multi-Layer Perceptron (MLP)
2. Convolutional Neural Network (CNN)
3. Patch-based Vision Transformer / Attention Network
"""

import torch
import torch.nn as nn


class FashionMLP(nn.Module):
    """
    MLP Architecture:
    Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FashionCNN(nn.Module):
    """
    CNN Architecture:
    Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool2 ->
    Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2 ->
    Flatten -> Linear(32 * 7 * 7, 10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class PatchAttentionModel(nn.Module):
    """
    Attention Architecture:
    - Split 28x28 image into 16 patches of 7x7 (49 features each)
    - Linear projection to d_model=64
    - 2 Transformer encoder layers: d_model=64, nhead=4, dim_feedforward=128
    - Mean pool over sequence length (16 tokens)
    - Linear projection 64 -> 10
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128, num_layers: int = 2):
        super().__init__()
        self.patch_size = 7
        self.num_patches = (28 // 7) * (28 // 7)  # 16 patches
        self.patch_dim = 1 * 7 * 7               # 49 values
        self.d_model = d_model

        # Linear projection from patch space (49) to embedding space (64)
        self.patch_proj = nn.Linear(self.patch_dim, d_model)

        # Learnable 1D position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="relu",
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.classifier = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 28, 28)
        B = x.shape[0]
        # Extract 7x7 patches: (B, 1, 4, 4, 7, 7)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape to (B, 16, 49)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, self.num_patches, self.patch_dim)

        # Project to d_model
        tokens = self.patch_proj(patches) + self.pos_embed

        # Pass through Transformer encoder
        features = self.transformer(tokens)

        # Mean pooling across tokens
        pooled = features.mean(dim=1)

        # Final logits
        logits = self.classifier(pooled)
        return logits
