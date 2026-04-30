# Liveness Detection using Vision Transformer (ViT)

A face liveness / anti-spoofing detection system built with **Vision Transformer (ViT)** and fine-tuned on the **LCC_FASD dataset** to classify facial inputs as **real** or **spoof**.

This project explores transformer-based computer vision for biometric security, leveraging Hugging Face Transformers and transfer learning for high-performance spoof detection.

---

## Overview

Traditional face recognition systems are vulnerable to spoofing attacks such as:

- Printed photos
- Replay attacks
- Screen displays
- Static facial reproductions

This project enhances biometric security by performing **binary liveness classification**:

- **Real Face**
- **Spoof Attack**

---

## Features

- Fine-tuned **Vision Transformer (ViT)**
- Transfer learning from `google/vit-base-patch16-224-in21k`
- Balanced training via random oversampling
- Hugging Face Trainer pipeline
- Exported model to Hugging Face Hub
- Inference pipeline for real-world testing

---

## Model Architecture

Base Model:

- **Vision Transformer (ViT Base Patch16 224)**

Frameworks:

- **PyTorch**
- **Hugging Face Transformers**
- **Datasets**
- **Accelerate**

---

## Dataset

Trained on **LCC_FASD Dataset**

### Original Dataset Size

| Class | Samples |
|--------|--------|
| Real | Minority |
| Spoof | Majority |

### After Oversampling

| Total Samples | 33,770 |
|--------------|-------|

---

## Performance

### Test Metrics

| Metric | Score |
|--------|------|
| Accuracy | **92.08%** |
| F1 Score | **92.05%** |
| Precision (Real) | **87.88%** |
| Recall (Real) | **97.62%** |
| Precision (Spoof) | **97.32%** |
| Recall (Spoof) | **86.54%** |

---

## Training Configuration

```bash
Model: google/vit-base-patch16-224-in21k
Epochs: 2
Learning Rate: 1e-6
Train Batch Size: 32
Eval Batch Size: 8
Weight Decay: 0.02
Image Size: 224x224
