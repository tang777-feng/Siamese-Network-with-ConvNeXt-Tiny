import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden_dim=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )
    def forward(self, x):
        return self.layers(x)

class ContrastiveLossWithHardMining(nn.Module):
    """Contrastive Loss integrated with intelligent Hard Negative Mining"""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, emb1, emb2, labels):
        sim = F.cosine_similarity(emb1, emb2)
        loss_pos = labels * torch.pow(1 - sim, 2)
        loss_neg = (1 - labels) * torch.pow(F.relu(sim - self.margin), 2)
        
        # Hard Mining Logic
        scores = torch.mm(emb1, emb2.t())
        mask = torch.eye(emb1.size(0), device=emb1.device).bool()
        scores.masked_fill_(mask, -999)
        hard_sim, _ = scores.max(dim=1)
        loss_hard = torch.pow(F.relu(hard_sim - self.margin), 2)
        
        return torch.mean(loss_pos + loss_neg + 0.5 * loss_hard)

class DualTowerModel(nn.Module):
    def __init__(self, backbone_name="convnext_tiny", pretrained=True, feature_dim=512):
        super().__init__()
        self.backbone, self.feat_dim = self._build_backbone(backbone_name, pretrained)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = ProjectionHead(self.feat_dim, feature_dim)
        self.criterion = ContrastiveLossWithHardMining(margin=0.5)

    def _build_backbone(self, name, pretrained):
        if name == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            full = models.convnext_tiny(weights=weights)
            return full.features, 768
        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward_one(self, x):
        x = self.backbone(x)     # [B, 768, 7, 7]
        x = self.pool(x)         # [B, 768, 1, 1]
        x = torch.flatten(x, 1)  # [B, 768]
        return self.projector(x)

    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return F.normalize(emb1, p=2, dim=1), F.normalize(emb2, p=2, dim=1)

    def compute_loss(self, emb1, emb2, labels):
        return self.criterion(emb1, emb2, labels)

    def compute_similarity(self, emb1, emb2):
        return F.cosine_similarity(emb1, emb2)