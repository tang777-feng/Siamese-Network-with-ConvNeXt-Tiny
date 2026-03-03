import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from config import Config
from dataset import PathologyDataset, DataSplitter
from model import DualTowerModel

def evaluate(model, loader, device):
    """Evaluate model on validation set"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for qp, lk, labels in loader:
            qp, lk = qp.to(device), lk.to(device)
            emb1, emb2 = model(qp, lk)
            sim = model.compute_similarity(emb1, emb2)
            all_probs.extend(sim.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    try:
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    except Exception:
        auc, acc = 0.5, 0.0
    return auc, acc

def train_one_fold(fold_idx, total_folds=5):
    """Train execution for a single fold"""
    cfg = Config(fold_idx=fold_idx, total_folds=total_folds)
    device = torch.device(cfg.device)
    
    # 1. Obtain Cross-Validation Case IDs (Test IDs are strictly ignored during this phase)
    train_ids, val_ids, _ = DataSplitter.get_split_ids(
        cfg.data_root, fold_idx, total_folds, cfg.holdout_ratio, cfg.seed
    )
    
    print(f"\n" + "="*50)
    print(f"🚀 Training Fold {fold_idx+1}/{total_folds}")
    print(f"📦 Train Cases: {len(train_ids)} | Val Cases: {len(val_ids)}")
    print("="*50)
    
    # 2. Build Dataset
    train_set = PathologyDataset(cfg.data_root, train_ids, "train", cfg.get_train_transform(), similar_neg_ratio=0.8)
    val_set = PathologyDataset(cfg.data_root, val_ids, "val", cfg.get_val_transform(), similar_neg_ratio=0.8)
    
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 3. Initialize Model, Optimizer, and LR Scheduler
    model = DualTowerModel(cfg.backbone_name).to(device)
    optimizer = cfg.get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader), eta_min=1e-6
    )
    
    # 4. Logger and Model Checkpointing
    writer = SummaryWriter(os.path.join(cfg.save_dir, "logs"))
    best_auc = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for qp, lk, labels in pbar:
            qp, lk, labels = qp.to(device), lk.to(device), labels.to(device)
            
            optimizer.zero_grad()
            emb1, emb2 = model(qp, lk)
            loss = model.compute_loss(emb1, emb2, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        val_auc, val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f} | Val ACC: {val_acc:.4f}")
        
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("ACC/val", val_acc, epoch)
        
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(cfg.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': best_auc
            }, save_path)
            print(f"  🌟 [New Best] Model saved! AUC: {best_auc:.4f}")

    print(f"\n✅ Fold {fold_idx+1} completed. Best Val AUC: {best_auc:.4f}")
    writer.close()

if __name__ == "__main__":
    total_folds = 5
    for i in range(total_folds):
        train_one_fold(fold_idx=i, total_folds=total_folds)
        
    print("\n🎉 5-Fold Cross-Validation complete! You can now run test.py to generate final publication metrics.")
