# src/predict.py
import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Predict image class")
    parser.add_argument("--model", type=str, required=True, help="Path to model.h5")
    parser.add_argument("--class_indices", type=str, required=True, help="Path to class_indices.json")
    parser.add_argument("--img", type=str, required=True, help="Path to image file")
    parser.add_argument("--img_size", type=int, default=128)
    return parser.parse_args()

def load_and_prep(img_path, target_size):
    img = image.load_img(img_path, target_size=(target_size, target_size))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    args = parse_args()
    model = load_model(args.model)
    with open(args.class_indices, "r") as f:
        class_indices = json.load(f)
    # invert mapping
    idx_to_class = {v:k for k,v in class_indices.items()}

    x = load_and_prep(args.img, args.img_size)
    preds = model.predict(x)[0]
    top_idx = int(np.argmax(preds))
    top_class = idx_to_class[top_idx]
    top_prob = float(preds[top_idx])

    print(f"Predicted: {top_class} ({top_prob*100:.2f}%)")

if __name__ == "__main__":
    main()
