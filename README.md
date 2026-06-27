# Lung & Colon Cancer Histopathological Image Classification

A deep learning project that builds a Convolutional Neural Network (CNN) from scratch to classify histopathological images across 5 lung and colon cancer tissue types.

---

## Project Overview

Histopathological imaging is a key diagnostic tool in oncology. This project explores whether a custom-built CNN can learn to distinguish between cancerous and healthy tissue samples across lung and colon cancer types using image data alone.

The model is trained on 25,000 microscopy images across 5 classes and evaluated using accuracy metrics, loss curves, and a confusion matrix.

---

## Dataset

**Source:** [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) - Kaggle (Andrew MVD)

| Class | Description |
|---|---|
| `colon_aca` | Colon adenocarcinoma |
| `colon_n` | Colon benign tissue (healthy) |
| `lung_aca` | Lung adenocarcinoma |
| `lung_n` | Lung benign tissue (healthy) |
| `lung_scc` | Lung squamous cell carcinoma |

- **Total images:** 25,000 (5,000 per class)
- **Image size:** 200 × 200 pixels (RGB)
- **Train/Validation split:** 80% / 20%

---

## Model Architecture

A custom CNN built using TensorFlow/Keras with the following structure:

```
Input (200x200x3)
→ Conv2D(32) + BatchNorm + MaxPooling
→ Conv2D(64) + BatchNorm + MaxPooling
→ Conv2D(128) + BatchNorm + MaxPooling
→ GlobalAveragePooling2D
→ Dense(128, ReLU) + Dropout(0.5)
→ Dense(5, Softmax)
```

**Compiled with:**
- Optimiser: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

---

## Tech Stack

- Python 3
- TensorFlow / Keras
- scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Google Colab

---

## Results & Observations

The model achieved high **training accuracy (~99%)** but showed unstable **validation accuracy**, indicating overfitting — a common challenge when training small custom CNNs from scratch on large, fine-grained medical imaging datasets.

| Metric | Value |
|---|---|
| Training Accuracy | ~99% |
| Validation Accuracy | Unstable (ranged 50–96% across epochs) |

**Key observations:**
- Training loss steadily decreased while validation loss spiked unpredictably — a classic sign of overfitting
- The confusion matrix showed near-uniform prediction distribution, confirming the model did not generalise well across all 5 classes
- Histopathological images are highly fine-grained, making them difficult for small custom CNNs without pretrained feature extractors

---

## What I Would Improve

Given more time and compute, the following changes would likely improve performance significantly:

1. **Transfer Learning** — Use a pretrained model (e.g. ResNet50, EfficientNet) via Keras Applications instead of training from scratch. Pretrained models already understand low-level visual features and generalise far better on medical imaging tasks
2. **Stronger Data Augmentation** — Add rotation, flipping, zoom, and brightness shifts to reduce overfitting
3. **Learning Rate Scheduling** — Use a learning rate scheduler or ReduceLROnPlateau callback to stabilise training
4. **Early Stopping** — Stop training when validation loss stops improving to avoid overfitting past the optimal point

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Set up your Kaggle API credentials (`kaggle.json`) and place them in `~/.kaggle/`
3. Run all cells in order — the notebook will download the dataset automatically via the Kaggle API
4. Training takes approximately 30–60 minutes depending on GPU availability

> **Note:** A free Colab GPU (T4) is recommended. CPU training will be significantly slower on a dataset of this size.

---

## Acknowledgements

Dataset sourced from Kaggle, originally prepared by [Andrew MVD](https://www.kaggle.com/andrewmvd).  
Academic project : Taylor's University, Bachelor of Computer Science (AI).
