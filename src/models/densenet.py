"""Pretrained DenseNet121 adapted to one-logit pneumonia classification."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Binary(nn.Module):
    """Thin wrapper that enforces one scalar logit per input image."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_strategy: str = "last_block",
        weights: Optional[DenseNet121_Weights] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        freeze_strategy = freeze_strategy.lower()
        if freeze_strategy not in {"head_only", "last_block", "full"}:
            raise ValueError("freeze_strategy must be head_only, last_block, or full")

        if pretrained and weights is None:
            weights = DenseNet121_Weights.DEFAULT
        if not pretrained:
            weights = None

        try:
            backbone = densenet121(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Could not load ImageNet DenseNet121 weights. On the shared GPU "
                    "server, cache the torchvision weights before the booked run or ask "
                    "the server steward to enable/download them. For offline smoke tests "
                    "set model.pretrained=false only; do not use that for the final main run."
                ) from exc
            raise

        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone
        self.freeze_strategy = freeze_strategy
        self._apply_freeze_strategy(freeze_strategy)

    def _apply_freeze_strategy(self, strategy: str) -> None:
        # Classifier always remains trainable.
        for p in self.backbone.parameters():
            p.requires_grad = True

        if strategy == "head_only":
            for p in self.backbone.features.parameters():
                p.requires_grad = False
        elif strategy == "last_block":
            for p in self.backbone.features.parameters():
                p.requires_grad = False
            # Partial fine-tuning: last dense block + final norm.
            for p in self.backbone.features.denseblock4.parameters():
                p.requires_grad = True
            for p in self.backbone.features.norm5.parameters():
                p.requires_grad = True
        elif strategy == "full":
            pass

    def train(self, mode: bool = True):
        """Keep frozen BatchNorm statistics frozen during fine-tuning."""
        super().train(mode)
        if mode and self.freeze_strategy == "head_only":
            self.backbone.features.eval()
        elif mode and self.freeze_strategy == "last_block":
            # Earlier frozen feature blocks stay in eval mode so their BatchNorm
            # running statistics do not drift. The trainable final block/norm
            # remain in train mode.
            self.backbone.features.eval()
            self.backbone.features.denseblock4.train(True)
            self.backbone.features.norm5.train(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise RuntimeError(f"Unexpected DenseNet binary head output: {tuple(logits.shape)}")
        return logits[:, 0]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_densenet121(
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_strategy: str = "last_block",
) -> DenseNet121Binary:
    return DenseNet121Binary(
        pretrained=pretrained,
        dropout=dropout,
        freeze_strategy=freeze_strategy,
    )
