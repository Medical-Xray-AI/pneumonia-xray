"""
baseline.py

Small, from-scratch CNN baseline for binary chest X-ray classification
(NORMAL vs PNEUMONIA). The point of a baseline is to be a fair, honest
floor that the pretrained DenseNet121 (Uzv 3) must beat — not to be
competitive on its own.

Model contract (src/models/README.md):
    - Input shape:  [batch, 3, 224, 224]
    - Output shape: [batch, 1] raw pneumonia logits (sigmoid applied only
      when probabilities are actually needed, e.g. at eval/inference time)
    - Target mapping: normal=0, pneumonia=1
    - Training loss: weighted BCEWithLogitsLoss (pos_weight from configs)

Architecture: 4 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool), global
average pooling, small classifier head. ~1-2M parameters — trains
comfortably within the project's compute budget.

USAGE
-----
    from src.models.baseline import BaselineCNN, build_baseline_from_config

    model = build_baseline_from_config(model_cfg)   # model_cfg = cfg["model"]
    logits = model(images)                          # images: (B, 3, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class BaselineCNN(nn.Module):
    """
    Input:  (B, 3, H, W) — RGB (grayscale replicated to 3 channels)
    Output: (B, output_units) raw logits. output_units=1 for binary
            classification with BCEWithLogitsLoss, matching
            configs/baseline.yaml `model.output_units`.
    """

    def __init__(self, output_units: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),    # H/2
            conv_block(32, 64),   # H/4
            conv_block(64, 128),  # H/8
            conv_block(128, 256),  # H/16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, output_units),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline_from_config(model_cfg: dict) -> BaselineCNN:
    """
    Config-driven constructor so src/models/factory.py (Uzv 3's planned
    module) can build this model purely from configs/baseline.yaml
    `model:` block, the same way it will build DenseNet121 from
    configs/densenet121.yaml — no baseline-specific branching needed
    in the factory itself.
    """
    return BaselineCNN(
        output_units=model_cfg.get("output_units", 1),
        dropout=model_cfg.get("dropout", 0.3),
    )


if __name__ == "__main__":
    # Quick shape / parameter-count smoke test (also covered by
    # tests/test_baseline.py — keep this for a fast manual check).
    model = BaselineCNN()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Trainable parameters: {model.num_parameters():,}")
