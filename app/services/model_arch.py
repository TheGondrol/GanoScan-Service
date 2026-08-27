"""GanodermaCNN architecture.

Copied structurally from the training notebook so ``best_model.pth`` state_dicts
load cleanly. Only needed when serving raw ``.pth`` weights; ``model_jit.pt``
(TorchScript) does not use this class.
"""

import torch.nn as nn
from torchvision import models


class GanodermaCNN(nn.Module):
    """CNN classifier with a pretrained backbone + custom head (4 classes)."""

    def __init__(self, num_classes=4, backbone="resnet50", pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            backbone_model = models.resnet50(weights=None)
            feature_dim = 2048
        elif backbone == "resnet101":
            backbone_model = models.resnet101(weights=None)
            feature_dim = 2048
        elif backbone == "resnet18":
            backbone_model = models.resnet18(weights=None)
            feature_dim = 512
        elif backbone == "densenet121":
            backbone_model = models.densenet121(weights=None)
            feature_dim = 1024
        elif backbone == "efficientnet_b0":
            from torchvision.models import efficientnet_b0

            backbone_model = efficientnet_b0(weights=None)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if "resnet" in backbone or "densenet" in backbone:
            self.features = nn.Sequential(*list(backbone_model.children())[:-1])
        elif "efficientnet" in backbone:
            self.features = backbone_model.features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
