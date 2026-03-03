import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from config import Config
from dataset import PathologyDataset, DataSplitter
from model import DualTowerModel

# ================= Global Plot Settings (Publication Ready) =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 14
# ============================================================================

def evaluate_fold(model, loader, device):
    """Inference process for a single fold model"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for qp, lk, labels in loader:
            qp, lk = qp.to(device), lk.to(device)
            emb1, emb2 = model(qp, lk)
            sim = model.compute_similarity(emb1, emb2)
            all_probs.extend(sim.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

def denormalize_tensor_to_cv2(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize Tensor back to OpenCV format"""
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(0, 1) 
    img_np = tensor.permute(1, 2, 0).numpy() * 255.0
    img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr

def visualize_augmented_predictions(dataset, ensemble_probs, ensemble_preds, save_dir, max_per_class=20):
    """Generate high-quality paired visualization output with annotations"""
    vis_dir = os.path.join(save_dir, "Visualization_Pairs_Augmented")
    categories = ['TP', 'TN', 'FP', 'FN']
    for cat in categories:
        os.makedirs(os.path.join(vis_dir, cat), exist_ok=True)
        
    counts = {cat: 0 for cat in categories}
    print(f"\n📸 Generating qualitative sample visualizations at: {vis_dir}")

    for i in tqdm(range(len(dataset))):
        prob = ensemble_probs[i]
        pred = ensemble_preds[i]
        
        qp_tensor, lk_tensor, true_label_tensor = dataset[i]
        true_label = int(true_label_tensor.item())
        
        if true_label == 1 and pred == 1:
            cat, color = 'TP', (0, 180, 0)
        elif true_label == 0 and pred == 0:
            cat, color = 'TN', (0, 180, 0)
        elif true_label == 0 and pred == 1:
            cat, color = 'FP', (0, 0, 220)
        elif true_label == 1 and pred == 0:
            cat, color = 'FN', (0, 0, 220)
            
        if counts[cat] >= max_per_class: continue
            
        qp_img = denormalize_tensor_to_cv2(qp_tensor)
        lk_img = denormalize_tensor_to_cv2(lk_tensor)
        
        qp_img = cv2.resize(qp_img, (400, 400))
        lk_img = cv2.resize(lk_img, (400, 400))
        
        white_gap = np.ones((15, 400, 3), dtype=np.uint8) * 255
        pair_img = np.vstack((qp_img, white_gap, lk_img))
        
        info_bar = np.ones((90, 400, 3), dtype=np.uint8) * 255
        
        label_str = "Match" if true_label == 1 else "Mismatch"
        pred_str = "Match" if pred == 1 else "Mismatch"
        
        cv2.putText(info_bar, f"True: {label_str}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(info_bar, f"Pred: {pred_str}", (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.rectangle(info_bar, (280, 20), (380, 70), color, -1)
        cv2.putText(info_bar, f"{cat}", (305, 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                    
        final_img = np.vstack((pair_img, info_bar))
        cv2.imwrite(os.path.join(vis_dir, cat, f"{cat}_id{i}.jpg"), final_img)
        counts[cat] += 1

def plot_roc_curves(all_fprs, all_tprs, tprs_interp, mean_fpr, mean_tpr, mean_auc, std_auc, save_path):
    """Plot ROC curves with confidence interval shadow"""
    plt.figure(figsize=(7, 6))
    
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightcoral', alpha=0.3)
    
    for i in range(len(all_fprs)):
        plt.plot(all_fprs[i], all_tprs[i], color='gray', lw=1.2, alpha=0.3, linestyle='--')
        
    plt.plot(mean_fpr, mean_tpr, color='#d62728', lw=2.5,
             label=rf'AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f}')

    plt.plot([0, 1], [0, 1], linestyle=':', lw=2, color='black', alpha=0.5)
    plt.xlim([-0.02, 1.02]), plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.legend(loc="lower right", frameon=False, prop={'weight':'bold', 'size': 13})

    plt.grid(True, linestyle='-', alpha=0.2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot Confusion Matrix annotated with counts and percentages"""
    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f"{cm[i, j]}\n({cm[i, j]/total:.1%})"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot_labels, fmt="", cmap='Blues', cbar=False,
                annot_kws={'size': 16, 'weight': 'bold'},
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.xlabel('Predicted Label', fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontweight='bold', labelpad=10)
    plt.title('Ensemble Confusion Matrix (Opt. Threshold)', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_bar_with_scatter(ensemble_metrics, fold_metrics_dict, save_path):
    """Plot advanced bar charts containing single fold jitter-scatters"""
    names = [k for k in ensemble_metrics.keys() if k != 'AUC']
    values = [ensemble_metrics[k] for k in names]
    
    plt.figure(figsize=(7, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2'] 

    x_pos = np.arange(len(names))
    bars = plt.bar(x_pos, values, color=colors, width=0.5, alpha=0.8, edgecolor='black', linewidth=1.2)

    np.random.seed(42)
    for i, metric_name in enumerate(names):
        fold_vals = fold_metrics_dict[metric_name]
        x_jitter = x_pos[i] + np.random.uniform(-0.1, 0.1, size=len(fold_vals))
        plt.scatter(x_jitter, fold_vals, color=colors[i], s=45, alpha=1.0, zorder=3, edgecolors='black', linewidth=1.2)
        
        max_height = max(values[i], max(fold_vals))
        plt.text(x_pos[i], max_height + 0.02, f'{values[i]:.3f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.ylim(0, 1.15)
    plt.ylabel('Score', fontweight='bold')
    plt.title('Overall Classification Performance', fontweight='bold', pad=15)
    plt.xticks(x_pos, names, fontweight='bold')
                 
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    total_folds = 5
    cfg = Config(fold_idx=0, total_folds=total_folds)
    device = torch.device(cfg.device)
    
    out_dir = f"./results/{cfg.model_key}/test_results"
    os.makedirs(out_dir, exist_ok=True)

    _, _, test_ids = DataSplitter.get_split_ids(
        cfg.data_root, fold_idx=0, total_folds=total_folds, holdout_ratio=cfg.holdout_ratio, seed=cfg.seed
    )

    test_set = PathologyDataset(cfg.data_root, test_ids, "test", cfg.get_val_transform(), similar_neg_ratio=0.8)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_fold_probs, all_fprs, all_tprs = [], [], []
    fold_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'AUC': []}
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = []
    true_labels = None

    for fold_idx in range(total_folds):
        weight_path = f"./results/{cfg.model_key}/fold_{fold_idx}/best_model.pth"
        model = DualTowerModel(cfg.backbone_name).to(device)
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        probs, labels = evaluate_fold(model, test_loader, device)
        all_fold_probs.append(probs)
        if true_labels is None: true_labels = labels
            
        fpr, tpr, thresholds = roc_curve(labels, probs)
        
        opt_idx = np.argmax(tpr - fpr)
        fold_opt_th = thresholds[opt_idx]
        preds = (probs > fold_opt_th).astype(int)
        
        fold_metrics['AUC'].append(auc(fpr, tpr))
        fold_metrics['Accuracy'].append(accuracy_score(labels, preds))
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        fold_metrics['Precision'].append(p)
        fold_metrics['Recall'].append(r)
        fold_metrics['F1-Score'].append(f)
        
        all_fprs.append(fpr); all_tprs.append(tpr)
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0

    # 1. Ensemble probabilities
    ensemble_probs = np.mean(all_fold_probs, axis=0) 
    
    # 2. Compute the optimal threshold using Youden's J statistic for the ensemble model
    ens_fpr, ens_tpr, ens_thresholds = roc_curve(true_labels, ensemble_probs)
    ens_opt_idx = np.argmax(ens_tpr - ens_fpr)
    optimal_threshold = ens_thresholds[ens_opt_idx]
    
    ensemble_preds = (ensemble_probs > optimal_threshold).astype(int)

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(fold_metrics['AUC']) 

    ep, er, ef, _ = precision_recall_fscore_support(true_labels, ensemble_preds, average='binary', zero_division=0)
    ensemble_metrics = {
        'Accuracy': accuracy_score(true_labels, ensemble_preds),
        'Precision': ep,
        'Recall': er,
        'F1-Score': ef,
        'AUC': mean_auc
    }

    print("\n" + "="*45)
    print(f"💡 [Auto Threshold] Optimal Youden's J threshold: {optimal_threshold:.4f}")
    print("Test Results (Ensemble + Optimal Threshold)")
    for k, v in ensemble_metrics.items(): print(f"{k:>12}: {v:.4f}")
    print("="*45 + "\n")

    # Generate advanced visual plots
    visualize_augmented_predictions(test_set, ensemble_probs, ensemble_preds, out_dir, max_per_class=400)
    
    print(f"📊 Generating Publication-ready SCI figures at: {out_dir}")
    
    plot_roc_curves(all_fprs, all_tprs, tprs_interp, mean_fpr, mean_tpr, mean_auc, std_auc, os.path.join(out_dir, "ROC_5folds.png"))
    plot_confusion_matrix(true_labels, ensemble_preds, os.path.join(out_dir, "Confusion_Matrix.png"))
    plot_metrics_bar_with_scatter(ensemble_metrics, fold_metrics, os.path.join(out_dir, "Metrics_Bar.png"))

    print("🎉 Evaluation and Plot Generation Completed!")

if __name__ == "__main__":
    main()