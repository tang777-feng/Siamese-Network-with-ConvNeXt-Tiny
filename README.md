# An Artificial Intelligence Framework for Automated Quality Control of Paraffin Block and Slide Consistency 🔬

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official Implementation** of the paper: *"An Artificial Intelligence Framework for Automated Quality Control of Paraffin Block and Slide Consistency: A Clinical Evaluation and Human-Machine Comparison Study"*.

## 📖 Overview

Ensuring the consistency between paraffin blocks and their corresponding histological slides is a critical quality control (QC) step in surgical pathology to prevent specimen mismatch. Currently, this relies on manual visual verification, which is time-consuming and prone to human error.

This repository provides the **Siamese Network Matching Phase** of our proposed AI framework. It automates the morphological matching between paraffin block tissues and H&E-stained slide tissues. 

**Clinical Highlights:**
- **Exceptional Accuracy:** Achieves an overall accuracy of **95.05%** and a mean AUC of **0.9809**, outperforming human expert verification (91.52%).
- **High Efficiency:** Processes at less than **0.05 seconds per slide**, ~80x faster than manual workflows.
- **AI Rescue Capability:** Successfully identifies challenging mismatches often overlooked by human technicians, enhancing patient safety.

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
