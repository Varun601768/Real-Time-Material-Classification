# Real‑Time Material Classification — Scrap Simulation Challenge

![Project](https://img.shields.io/badge/Project-Real--Time%20Material%20Classification-blue)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

---

## Project Overview

This repository implements an end‑to‑end pipeline for **real‑time material classification** in a conveyor simulation scenario. The system classifies five scrap material types — **metal, plastic, e‑waste, paper, and fabric** — using images and video streams. It is designed for quick iteration with transfer learning, lightweight model export (TorchScript), manual correction, and an active‑learning loop to improve model performance over time.

---

## Features

* Real‑time inference on images and video frames (CPU/GPU)
* Transfer‑learning using a ResNet18 backbone pretrained on ImageNet
* TorchScript export for lightweight deployment
* Manual override UI (CLI/optional web UI) for correcting misclassifications
* Active learning: misclassified frames saved to `retrain_queue/` for future retraining
* Logging of predictions, timestamps and override decisions into `results/conveyor_results.csv`

---

## Dataset

**Folder layout (expected):**

```
Dataset/
├── e-waste/
├── fabric/
├── metal/
├── paper/
└── plastic/
```

* **Source:** Combined publicly available scrap datasets (Kaggle and other open datasets).
* **Rationale:** 5 distinct classes, each with ~700+ images (recommended) and variations in lighting, angle and background to better simulate conveyor conditions.
* **Image format:** JPEG/PNG; recommended input size 224×224 (RGB).

---

## Model & Approach

* **Base architecture:** ResNet18 (ImageNet pretrained)
* **Method:** Transfer learning — replace final fully connected layer to output 5 classes
* **Input:** 224×224 RGB images
* **Output:** Softmax probability vector over 5 classes
* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam (default lr=1e-4)

---

## Architecture & Training

### Model
- **Base Architecture:** ResNet18 (pretrained on ImageNet)  
- **Approach:** Transfer learning for faster convergence  
- **Input:** 224x224 RGB images  
- **Output:** Probability vector over 5 classes  

### Training Process
1. Data preprocessing: resizing, normalization, and augmentation (rotation, flips).  
2. Split: 80% training, 20% validation.  
3. Loss function: Cross-Entropy Loss  
4. Optimizer: Adam with learning rate 1e-4  
5. Epochs: 20 (early stopping applied)  
6. Metrics: Accuracy, Precision, Recall, Confusion Matrix  

---

## Deployment

- Model exported as **TorchScript** for lightweight inference.  
- Supports both **images and video files**.  
- **Manual override**: allows correction of misclassified frames.  
- **Active learning**: misclassified frames stored in `retrain_queue/` for future retraining.  
- Optimized for CPU and GPU (CUDA).  

---


**Notes:** `src/Real-Time Material Classification.ipynb` should save the best model (`best_resnet18_materials.pth`) and an exportable TorchScript or traced model (`resnet18_materials.pt`) to `models/`.

---
## Folder Structure

```
Project/
│
├── src/
│   ├── conveyor_simulation.py     
│   ├── inference.py              
│   └── Real-Time Material Classification.ipynb  # exploratory notebook
│
├── data/
│   ├── Dataset/                   
│   ├── conveyor_frames/           
│   └── retrain_queue/             
│
├── results/
│   ├── conveyor_results.csv       
│   └── screenshots/              
│
├── models/
│   ├── resnet18_materials.pt      # TorchScript export
│   └── best_resnet18_materials.pth # best checkpoint
│
├── README.md
└── performance_report.md/.pptx    # slides/report with evaluation results
```
# How to Run
- Setup Environment : git clone
  cd Project
## Create virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.\.venv\Scripts\activate # Windows
## Install dependencies
pip install -r requirements.txt
## Run Inference
- Single Image:  python src/inference.py --image ../Data/conveyor_frames/image1.jpg
## Captured at intervals from a video or image folder
- python src/conveyor_simulation.py 
  
