import os
import torch
from albumentations import (
    Compose, Normalize, RandomRotate90, HorizontalFlip, VerticalFlip
)
from albumentations.pytorch import ToTensorV2

class Config:
    def __init__(self, fold_idx=0, total_folds=5):
        # Model identifier
        self.model_key = 'hard_mining'
        
        # Base parameters
        self.backbone_name = 'convnext_tiny'
        self.batch_size = 16
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.base_feature_dim = 768
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = 40 
        
        # Validation Strategy
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        self.holdout_ratio = 0.20
        self.seed = 42

        # Path configurations
        self.data_root = "./data"
        self.save_dir = f"./results/{self.model_key}/fold_{fold_idx}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Image and Model Hyperparameters
        self.img_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.projected_dim = 512
        self.margin = 0.5

    def get_train_transform(self):
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], additional_targets={'image2': 'image'})

    def get_val_transform(self):
        return Compose([
            Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ], additional_targets={'image2': 'image'})
    
    def get_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)