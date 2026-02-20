# src/models/efficientnet.py
import torch
import torch.nn as nn
import timm

class KidneyStoneClassifier(nn.Module):
    def __init__(self,
                 backbone: str   = 'efficientnet_b4',
                 num_classes: int = 2,
                 drop_rate: float = 0.4,
                 pretrained: bool = True):
        super().__init__()

        # Load pre-trained backbone — downloads ~75MB first time
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,         # remove original head
            drop_rate=drop_rate,
        )
        in_features = self.backbone.num_features  # 1792 for B4

        # New classification head for our binary task
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)  # raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)  # (B, C, H, W)
        logits   = self.head(features)                 # (B, num_classes)
        return logits

    def freeze_backbone(self):
        """Freeze backbone — only train head. Use for first 3 epochs."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print('Backbone frozen — training head only')

    def unfreeze_backbone(self):
        """Unfreeze all — fine-tune the whole network."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print('Backbone unfrozen — fine-tuning entire network')


