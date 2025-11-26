# 🧠 CNN Natural Images Classifier  
A complete deep-learning pipeline for **7-class image classification** using a custom **Convolutional Neural Network (CNN)** trained on the *Natural Images Dataset*.  
Built for academic submission, portfolio use, and real-world deployment.

---

## 🔰 Badges
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![Build Status](https://img.shields.io/github/actions/workflow/status/orkotasin10/cnn-natural-images-classifier/ci.yml?branch=main)](https://github.com/orkotasin10/cnn-natural-images-classifier/actions)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()  

---

# 📌 Table of Contents
- [📘 Project Overview](#-project-overview)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Installation](#️-installation)
- [📦 Dataset Setup](#-dataset-setup)
- [🏋️ Model Training](#️-model-training)
- [🔮 Making Predictions](#-making-predictions)
- [📊 Results \& Visuals](#-results--visuals)
- [📁 Screenshots \& Demo GIF](#-screenshots--demo-gif)
- [♻️ Reproducibility](#️-reproducibility)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📧 Contact](#-contact)

---

# 📘 Project Overview
This project builds a **CNN-based multiclass classifier** capable of identifying images from **7 categories**:


The repository includes:
- A full **Jupyter Notebook** for experimentation  
- Clean & modular **Python scripts** (`src/`) for training and inference  
- A formatted **project report**  
- A professional-level **README**, badges, and demo assets  
- Instructions for dataset setup, training, and prediction  

✔ Suitable for **academic submission**  
✔ Perfect for **portfolio / resume / GitHub showcase**  
✔ Clean modular code following best practices  

---

# 📂 Repository Structure

# ⚙️ Installation
 1️⃣ Clone the repository

git clone https://github.com/orkotasin10/cnn-natural-images-classifier.git
cd cnn-natural-images-classifier
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
kaggle datasets download -d prasunroy/natural-images
unzip natural-images.zip -d data/
data/
├── train/
│   ├── airplane/
│   ├── car/
│   └── ...
└── val/
    ├── airplane/
    ├── car/
    └── ...
python src/train.py \
    --data_dir data/ \
    --epochs 25 \
    --batch_size 32 \
    --img_size 128 \
    --num_classes 7 \
    --output_dir artifacts
artifacts/
├── model.h5
├── history.npy
└── class_indices.json
python src/predict.py \
    --model artifacts/model.h5 \
    --class_indices artifacts/class_indices.json \
    --img demos/sample_input.jpg \
    --img_size 128
![Accuracy Curve](demos/loss_accuracy.png)
![Confusion Matrix](demos/confusion_matrix.png)


 2. model.py

# src/model.py
from tensorflow.keras import layers, models, optimizers

def create_model(input_shape=(128, 128, 3), num_classes=7, lr=1e-3):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
