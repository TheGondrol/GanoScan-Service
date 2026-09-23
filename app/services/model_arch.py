"""GanodermaCNN classifier + EnhancementAutoencoderV2 architectures.

Copied structurally from the training notebook so the ``*_best.pth``
state_dicts load cleanly. Only needed when serving raw ``.pth`` weights;
``classifier_jit.pt`` / ``autoencoder_jit.pt`` (TorchScript) do not use these
classes.
"""

import torch
import torch.nn as nn
from torchvision import models


class GanodermaCNN(nn.Module):
    """CNN classifier with a pretrained backbone + custom head (2 classes:
    Healthy / Infected)."""

    def __init__(self, num_classes=2, backbone="resnet50", pretrained=False, dropout=0.5):
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


class EnhancementAutoencoderV2(nn.Module):
    """U-Net-style enhancement autoencoder (3 skip connections). Trained
    self-supervised: input = gamma-degraded image, target = clean image,
    L1 loss. Sigmoid output, so it produces an image in [0, 1], not a score.
    Copied structurally from the training notebook so ``autoencoder_best.pth``
    loads cleanly; ``autoencoder_jit.pt`` (TorchScript) does not use this class.
    """

    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(nn.Conv2d(64, latent_dim, 3, 1, 1), nn.BatchNorm2d(latent_dim), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2, 2)  # bottleneck

        self.up3 = nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1)
        self.dec3 = nn.Sequential(nn.Conv2d(64 + latent_dim, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec2 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(32, in_channels, 4, 2, 1)
        self.dec1 = nn.Sequential(nn.Conv2d(in_channels + 32, in_channels, 3, 1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        d3 = self.dec3(torch.cat([self.up3(p3), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_act(d1)
