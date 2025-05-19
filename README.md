# Extended YOLOv8-SCSA for Skin Lesion Detection

This repository extends the original YOLOv8-SCSA architecture by integrating a custom preprocessing pipeline directly into the model's training process. The pipeline includes artifact removal, contrast enhancement, color normalization, and edge enhancement, significantly improving skin lesion detection accuracy on the HAM10000 dataset.

## 🚀 Key Features

- **Integrated Preprocessing**: Real-time image preprocessing using morphological filtering, CLAHE, normalization, and edge enhancement
- **YOLOv8-SCSA Backbone**: Utilizes spatially coordinated shuffling attention (SCSA) and center-weighted masking (CWM)
- **Flexible Model Training**: Multiple variants like basic, CA, SA, ResCBAM, and SCSA
- **Evaluation + Inference Support**: Easy scripts for reproducibility

---

## 🔧 Installation

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```
├── ultralytics/
│   ├── cfg/models/v8/             # Model YAML files
│   ├── data/                      # Data loading and preprocessing scripts
│       └── preprocessing.py       # Custom preprocessing pipeline
│   └── utils/                     # Logging, visualization, etc.
├── datasets/                      # Contains images and data.yaml
│   └── data_local.yaml
├── model.py                       # Main script for training/eval/inference
├── requirements.txt
└── README.md
```

---

## 📦 Usage

### 🔬 Training
```bash
python model.py train --model_type scsa --data_path "./datasets/data_local.yaml" --img_size 512 --size s
```

### ✅ Evaluation
```bash
python model.py evaluate --model_path "./logs/SCSA(s)/weights/best.pt" --data_path "./datasets/data_local.yaml" --img_size 512
```

### 🔍 Inference
```bash
python model.py inference --model_paths "./logs/SCSA(s)/weights/best.pt" --images_path "./datasets/valid/images" --labels_path "./datasets/valid/labels" --img_size 512
```

---

## 📸 Preprocessing Pipeline

- **Artifact Removal**: Morphological blackhat + inpainting
- **CLAHE**: Adaptive histogram equalization
- **Color Normalization**: Mean scaling RGB channels
- **Edge Enhancement**: Sobel filter + weighted merging

---

## 📊 Results

- Enhanced SCSA showed up to **+4.1% mAP@50** improvement over YOLOv8 baseline
- Maintained lightweight performance: **same params, GFLOPs** as baseline
- Lower inference time than ResCBAM

---
