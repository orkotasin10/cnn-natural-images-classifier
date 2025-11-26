# src/train.py
import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from src.model import create_model
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for image classification")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root. Should contain 'train' and 'val' subfolders.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    return parser.parse_args()

def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    os.makedirs(args.output_dir, exist_ok=True)

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    val_flow = val_gen.flow_from_directory(
        val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    model = create_model(input_shape=(args.img_size, args.img_size, 3), num_classes=args.num_classes)

    ckpt_path = os.path.join(args.output_dir, "model.h5")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_flow,
        epochs=args.epochs,
        validation_data=val_flow,
        callbacks=callbacks
    )

    # Save history
    hist_path = os.path.join(args.output_dir, "history.npy")
    np.save(hist_path, history.history)

    # Save class indices for prediction mapping
    classes_path = os.path.join(args.output_dir, "class_indices.json")
    with open(classes_path, "w") as f:
        json.dump(train_flow.class_indices, f)

    print("Training complete. Model and history saved to:", args.output_dir)

if __name__ == "__main__":
    main()
