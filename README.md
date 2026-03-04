# An Artificial Intelligence Framework for Automated Quality Control of Paraffin Block and Slide Consistency 🔬

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official Implementation** of the paper: *"An Artificial Intelligence Framework for Automated Quality Control of Paraffin Block and Slide Consistency: A Clinical Evaluation and Human-Machine Comparison Study"*.

## 📖 Overview

Ensuring the consistency between paraffin blocks and their corresponding histological slides is a critical quality control (QC) step in surgical pathology to prevent specimen mismatch. Currently, this relies on manual visual verification, which is time-consuming and prone to human error.

This repository provides the **Siamese Network Matching Phase** of our proposed AI framework. It automates the morphological matching between paraffin block tissues and H&E-stained slide tissues. 

---

## 🧠 Framework Architecture

Our complete clinical workflow consists of two stages. **This repository contains the code for Stage 2.**

1. **Stage 1: Tissue Localization (YOLOv11)** - *Prior to running this code*, YOLOv11 is used to automatically detect, crop, and standardize tissue regions from raw block and slide images, eliminating background noise.
2. **Stage 2: Consistency Verification (Provided Here)** - A weight-sharing Dual-Tower Siamese Network (`ConvNeXt-Tiny`) extracts unified features from the cropped pairs. It utilizes **Online Hard Negative Mining** to calculate feature-space distances and predict consistency.

### ✨ Key Code Features
* **Clinically-Driven Negative Sampling:** Automatically generates negative pairs with an 80% size-matched hard-negative ratio to simulate real-world, morphologically similar clinical mismatches.
* **Hard-Mining Contrastive Loss:** Custom loss function (`Margin=0.5`) designed to force the model to learn fine-grained discriminative boundaries.
* **Strict Zero-Leakage Evaluation:** Implements a rigorous 3-way split (60% Train, 20% Val for 5-Fold CV, 20% strict Hold-out Test).
* **Automated SCI-Grade Visualizations:** Dynamically calculates the optimal Youden's J Index threshold and generates publication-ready ROC curves, Jitter-bar charts, and annotated image pair visualizations.

---

## 🛠️ Installation & Setup

We recommend using a Conda environment with GPU support for optimal performance.

```bash
# Clone the repository
git clone https://github.com/YourUsername/AutoQC-BlockSlide.git
cd AutoQC-BlockSlide

# Create and activate environment
conda create -n block_slide_qc python=3.9 -y
conda activate block_slide_qc

# Install PyTorch (Update the CUDA version according to your hardware)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies (Albumentations, Scikit-learn, OpenCV, Seaborn, etc.)
pip install -r requirements.txt

📂 Dataset Preparation
Since this code focuses on the matching phase, your images should already be cropped to the tissue regions of interest (via YOLOv11 or manual cropping).
Organize your dataset in the ./data directory exactly as follows. The system automatically parses the CaseID by splitting the filenames at _QP (Query Patch / Paraffin Block) and _LK (Local Knowledge / Slide).
code
Text
AutoQC-BlockSlide/
├── data/
│   ├── QP/                 # Paraffin Block crops (e.g., Case001_QP.jpg)
│   └── LK/                 # Slide crops (e.g., Case001_LK_1.jpg)
├── config.py               # Hyperparameters & transforms
├── dataset.py              # Dataloaders & 80% hard-negative generation
├── model.py                # ConvNeXt-Tiny Siamese Network & Hard-Mining Loss
├── train.py                # 5-Fold CV training script
└── test.py                 # Hold-out evaluation & visualization generator
🚀 Usage
1. Model Training (5-Fold Cross-Validation)
The training script automatically splits the data (leaving 20% untouched for testing) and performs a 5-fold cross-validation on the remaining 80%.
code
Bash
python train.py
Outputs: TensorBoard logs and best model weights (best_model.pth) are saved in ./results/hard_mining/fold_[0-4]/.
2. Independent Testing & Visualization
After all 5 folds are trained, run the testing script to evaluate the ensemble model on the 20% isolated hold-out set.
code
Bash
python test.py
What it does:
Computes the optimal decision threshold dynamically using Youden's J statistic.
Evaluates Accuracy, Precision, Recall, F1-Score, and AUC.
Generates Publication-Ready Figures in ./results/hard_mining/test_results/:
 confidence interval shadows.
Confusion_Matrix.png: Matrix based on the optimal threshold, annotated with percentages.
Metrics_Bar.png: Advanced bar charts with single-fold scatter (jitter) plots.
Visualization_Pairs_Augmented/: High-resolution, vertically aligned visual comparisons of TP, TN, FP, and FN pairs with clear text overlays.
📊 Results Showcase
Our model achieves highly robust discriminative performance across various specimen types (surgical resections and small biopsies).
(Note: Add your generated ROC_5folds.png and Confusion_Matrix.png here once you upload to GitHub)
code
Markdown
<!-- Uncomment and add image paths when uploading to GitHub -->
<!-- 
<p align="center">
  <img src="results/hard_mining/test_results/ROC_5folds.png" width="45%" />
  <img src="results/hard_mining/test_results/Confusion_Matrix.png" width="45%" />
</p> 
-->

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
