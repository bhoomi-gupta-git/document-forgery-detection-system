"""
train_model.py — DocForge CNN Training Script
=============================================
Trains a MobileNetV2-based forgery detection model on CASIA2 dataset.

Usage:
    python train_model.py --dataset ./dataset/CASIA2 --epochs 20

Output:
    models/forgery_model.h5  ← drop this into your DocForge models/ folder

Requirements:
    pip install tensorflow opencv-python scikit-learn matplotlib numpy

Dataset structure expected:
    dataset/CASIA2/
        Au/    ← authentic images (label = 0)
        Tp/    ← tampered images  (label = 1)
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ── Config ────────────────────────────────────────────────────────────────────

IMG_SIZE      = (224, 224)       # must match config.MODEL_INPUT_SIZE
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-4
OUTPUT_PATH   = os.path.join('models', 'forgery_model.h5')

AUTHENTIC_LABEL = 0
FORGED_LABEL    = 1

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train DocForge forgery detection model')
    parser.add_argument(
        '--dataset', type=str, default='./dataset/CASIA2',
        help='Path to CASIA2 root folder (contains Au/ and Tp/ subfolders)'
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--batch', type=int, default=BATCH_SIZE,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--output', type=str, default=OUTPUT_PATH,
        help='Output path for saved model (default: models/forgery_model.h5)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit images per class (useful for quick testing, e.g. --limit 500)'
    )
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_images_from_folder(folder: str, label: int, limit=None) -> tuple:
    """Load and preprocess all images from a folder."""
    images, labels = [], []
    files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    if limit:
        files = files[:limit]

    total = len(files)
    print(f"  Loading {total} images from {folder} (label={'authentic' if label==0 else 'forged'})...")

    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")

        # Progress indicator
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total}", end='\r')

    print(f"\n  Loaded: {len(images)} images")
    return images, labels


def load_dataset(dataset_path: str, limit=None):
    """Load authentic and forged images from CASIA2 folder structure."""
    au_path = os.path.join(dataset_path, 'Au')
    tp_path = os.path.join(dataset_path, 'Tp')

    # Validate paths
    if not os.path.exists(au_path):
        print(f"ERROR: Authentic folder not found: {au_path}")
        print("Expected structure: dataset/CASIA2/Au/ and dataset/CASIA2/Tp/")
        sys.exit(1)
    if not os.path.exists(tp_path):
        print(f"ERROR: Tampered folder not found: {tp_path}")
        sys.exit(1)

    print("\n── Loading Dataset ──────────────────────────────────")
    au_images, au_labels = load_images_from_folder(au_path, AUTHENTIC_LABEL, limit)
    tp_images, tp_labels = load_images_from_folder(tp_path, FORGED_LABEL, limit)

    # Combine
    all_images = np.array(au_images + tp_images, dtype=np.float32) / 255.0
    all_labels = np.array(au_labels + tp_labels, dtype=np.int32)

    print(f"\nDataset summary:")
    print(f"  Authentic: {len(au_images)}")
    print(f"  Forged:    {len(tp_images)}")
    print(f"  Total:     {len(all_images)}")

    return all_images, all_labels


# ── ELA feature generation ────────────────────────────────────────────────────

def generate_ela_batch(images_uint8: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    Generate ELA images for a batch.
    Input: uint8 images (0-255)
    Output: ELA images normalised 0-1
    """
    ela_images = []
    for img in images_uint8:
        # Save at reduced quality and reload
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded   = cv2.imencode('.jpg', img, encode_param)
        compressed   = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Compute amplified difference
        diff = cv2.absdiff(img, compressed)
        ela  = np.clip(diff.astype(np.float32) * 20, 0, 255)
        ela_images.append(ela / 255.0)

    return np.array(ela_images, dtype=np.float32)


# ── Model architecture ────────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """
    MobileNetV2-based binary classifier for forgery detection.

    Architecture:
        MobileNetV2 (pretrained ImageNet, frozen initially)
        → GlobalAveragePooling2D
        → Dense(256, relu) + Dropout(0.5)
        → Dense(128, relu) + Dropout(0.3)
        → Dense(2, softmax)  ← [authentic, forged]
    """
    print("\n── Building Model ───────────────────────────────────")

    # Base model — pretrained on ImageNet
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base initially — only train the head
    base_model.trainable = False
    print(f"  Base: MobileNetV2 ({len(base_model.layers)} layers, frozen)")

    # Build full model
    inputs  = layers.Input(shape=(*IMG_SIZE, 3), name='input_image')
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D(name='gap')(x)
    x       = layers.Dense(256, activation='relu', name='dense_256')(x)
    x       = layers.Dropout(0.5, name='dropout_1')(x)
    x       = layers.Dense(128, activation='relu', name='dense_128')(x)
    x       = layers.Dropout(0.3, name='dropout_2')(x)
    outputs = layers.Dense(2, activation='softmax', name='output')(x)

    model = models.Model(inputs, outputs, name='DocForge_v1')
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")
    return model


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30):
    """Unfreeze the top N layers of the base model for fine-tuning."""
    base_model = model.layers[1]   # MobileNetV2 is layer index 1
    base_model.trainable = True

    # Freeze all except the top num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n  Fine-tuning: {trainable} layers unfrozen in base model")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),   # lower LR for fine-tune
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_dataset(args.dataset, limit=args.limit)

    # Train / validation / test split: 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\nSplit:")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)

    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=args.output,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger('training_logs/training_log.csv'),
    ]

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    print("\n── Phase 1: Training head (base frozen) ────────────")
    phase1_epochs = min(10, args.epochs // 2)

    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=phase1_epochs,
        batch_size=args.batch,
        callbacks=callback_list,
        verbose=1
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────────
    print("\n── Phase 2: Fine-tuning top layers ─────────────────")
    model = unfreeze_top_layers(model, num_layers=30)
    remaining_epochs = args.epochs - phase1_epochs

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=remaining_epochs,
        batch_size=args.batch,
        callbacks=callback_list,
        verbose=1
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n── Evaluation on Test Set ───────────────────────────")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Loss:     {test_loss:.4f}")

    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names=['Authentic', 'Forged']
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negatives  (Auth→Auth):   {cm[0][0]}")
    print(f"  False Positives (Auth→Forged): {cm[0][1]}")
    print(f"  False Negatives (Forg→Auth):   {cm[1][0]}")
    print(f"  True Positives  (Forg→Forg):   {cm[1][1]}")

    fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f"\n  False Positive Rate: {fp_rate:.2%}")
    print(f"  PRD Target ≥80% accuracy: {'✓ PASS' if test_acc >= 0.80 else '✗ FAIL'}")
    print(f"  PRD Target ≤15% FP rate:  {'✓ PASS' if fp_rate <= 0.15 else '✗ FAIL'}")

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(args.output)
    print(f"\n── Model Saved ──────────────────────────────────────")
    print(f"  Path: {args.output}")
    print(f"  Copy this file to your DocForge models/ folder")

    # ── Plot training curves ──────────────────────────────────────────────────
    _plot_history(history1, history2)

    return model, test_acc


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_history(history1, history2):
    """Save training accuracy and loss curves."""
    # Combine both phases
    acc  = history1.history['accuracy']     + history2.history['accuracy']
    val  = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss']         + history2.history['loss']
    vloss= history1.history['val_loss']     + history2.history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(acc,  label='Train Accuracy')
    ax1.plot(val,  label='Val Accuracy')
    ax1.axvline(len(history1.history['accuracy']) - 1,
                color='gray', linestyle='--', label='Fine-tune start')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss,  label='Train Loss')
    ax2.plot(vloss, label='Val Loss')
    ax2.axvline(len(history1.history['loss']) - 1,
                color='gray', linestyle='--', label='Fine-tune start')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_logs/training_curves.png', dpi=150)
    print(f"  Training curves saved to training_logs/training_curves.png")
    plt.close()


# ── Quick test ────────────────────────────────────────────────────────────────

def quick_test(model_path: str, image_path: str):
    """
    Test a saved model on a single image.
    Usage: Call this function manually after training.
    """
    model = tf.keras.models.load_model(model_path)
    img   = cv2.imread(image_path)
    img   = cv2.resize(img, IMG_SIZE)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img   = img.astype(np.float32) / 255.0
    img   = np.expand_dims(img, axis=0)

    preds      = model.predict(img, verbose=0)[0]
    verdict    = 'Forged' if np.argmax(preds) == 1 else 'Authentic'
    confidence = float(np.max(preds))

    print(f"\nTest result for: {image_path}")
    print(f"  Verdict:    {verdict}")
    print(f"  Confidence: {confidence:.2%}")
    return verdict, confidence


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 54)
    print("  DocForge — Model Training Script v1.0")
    print("=" * 54)

    # Check TensorFlow
    print(f"\nTensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected — training on CPU (will be slower)")

    args = parse_args()

    print(f"\nConfig:")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Output:     {args.output}")
    if args.limit:
        print(f"  Limit:      {args.limit} images per class")

    model, accuracy = train(args)

    print("\n" + "=" * 54)
    print(f"  Training complete!")
    print(f"  Final accuracy: {accuracy:.2%}")
    print(f"  Model saved to: {args.output}")
    print("=" * 54)