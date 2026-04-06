"""
utils.py - Shared utilities for the project.

Usage:  from utils import *
"""

import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seeds(seed: int = 42):
    """Set Python, NumPy, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# Data Pipeline
# ============================================================

def load_datasets(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    img_size: tuple,
    batch_size: int,
    seed: int = 42,
):
    """
    Load train/val/test as tf.data.Dataset with categorical labels.
    Returns (train_ds, val_ds, test_ds, class_names).
    Does NOT apply cache/shuffle/prefetch - do that in the notebook.
    """
    common = dict(image_size=img_size, batch_size=batch_size,
                  label_mode="categorical", seed=seed)

    train_ds = keras.utils.image_dataset_from_directory(train_dir, **common)
    val_ds = keras.utils.image_dataset_from_directory(val_dir, **common)
    test_ds = keras.utils.image_dataset_from_directory(test_dir, **common)

    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    return train_ds, val_ds, test_ds, class_names


# ============================================================
# Training Curves
# ============================================================

def plot_learning_curves(history, title: str = ""):
    """
    Plot loss, accuracy, and macro F1 (3 subplots). Prints best epoch.
    Accepts a History, a list of Histories, or a dict from load_history().
    """
    if isinstance(history, (list, tuple)):
        combined = {}
        for h in history:
            for k, v in h.history.items():
                combined.setdefault(k, []).extend(v)
    elif isinstance(history, dict):
        combined = history
    else:
        combined = history.history

    epochs = range(1, len(combined["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(epochs, combined["loss"], label="Train")
    axes[0].plot(epochs, combined["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, combined["accuracy"], label="Train")
    axes[1].plot(epochs, combined["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    f1_key, val_f1_key = "f1_macro", "val_f1_macro"
    if f1_key in combined:
        axes[2].plot(epochs, combined[f1_key], label="Train")
        axes[2].plot(epochs, combined[val_f1_key], label="Val")
        axes[2].set_title("F1 Score (macro)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "F1 not tracked", ha="center", va="center",
                     transform=axes[2].transAxes)

    if title:
        plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    best_epoch = int(np.argmin(combined["val_loss"]))
    print(f"Best epoch (lowest val_loss): {best_epoch + 1}")
    print(f"  Val Loss: {combined['val_loss'][best_epoch]:.4f}")
    print(f"  Val Acc:  {combined['val_accuracy'][best_epoch]:.4f}")
    if val_f1_key in combined:
        print(f"  Val F1:   {combined[val_f1_key][best_epoch]:.4f}")


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, test_ds, class_names: list[str], model_name: str = "Model"):
    """
    Full test evaluation: prints metrics, classification report,
    confusion matrix, and per-class F1 bar chart.
    Returns a metrics dict for compare_models().
    """
    # Collect predictions
    y_true, y_pred_probs = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute metrics from predictions (avoids Keras metric name issues)
    test_loss = model.evaluate(test_ds, verbose=0)[0]
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"  {model_name} - Test Evaluation")
    print(f"{'='*50}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 Macro: {test_f1:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names,
                          title=f"Confusion Matrix - {model_name}")
    plot_per_class_f1(y_true, y_pred, class_names,
                      title=f"Per-Class F1 - {model_name}")

    return {
        "model_name": model_name,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
    }


def plot_confusion_matrix(
    y_true, y_pred, class_names: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: tuple = (14, 12),
):
    """Confusion matrix heatmap. normalize=True for row-wise percentages."""
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_per_class_f1(y_true, y_pred, class_names: list[str], title: str = ""):
    """Horizontal bar chart of per-class F1, sorted ascending, color-coded."""
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True)
    f1_scores = {cls: report[cls]["f1-score"] for cls in class_names}
    sorted_items = sorted(f1_scores.items(), key=lambda x: x[1])
    names = [item[0].replace("_", " ") for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(6, len(class_names) * 0.35)))
    colors = ["#e74c3c" if s < 0.5 else "#f39c12" if s < 0.7 else "#2ecc71"
              for s in scores]
    ax.barh(names, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title(title or "Per-Class F1 Score")
    ax.axvline(np.mean(scores), color="black", ls="--", lw=1,
               label=f"Macro avg: {np.mean(scores):.3f}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Model Comparison
# ============================================================

def compare_models(metrics_list: list[dict]):
    """
    Comparison table + bar chart from a list of evaluate_model() dicts.
    """
    print(f"\n{'='*65}")
    print(f"  Model Comparison")
    print(f"{'='*65}")
    header = f"{'Model':<30s} {'Accuracy':>10s} {'F1 (macro)':>12s} {'Loss':>10s}"
    print(header)
    print("-" * 65)
    for m in metrics_list:
        print(f"{m['model_name']:<30s} "
              f"{m['test_accuracy']:>10.4f} "
              f"{m['test_f1_macro']:>12.4f} "
              f"{m['test_loss']:>10.4f}")
    print("-" * 65)

    names = [m["model_name"] for m in metrics_list]
    accs = [m["test_accuracy"] for m in metrics_list]
    f1s = [m["test_f1_macro"] for m in metrics_list]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(names)), 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1 (macro)", color="#55A868")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================
# History Serialization
# ============================================================

def save_history(history, filepath: str):
    """
    Save training history to JSON. Accepts a History, list of Histories,
    or dict. Useful to re-plot curves without re-training.
    """
    if isinstance(history, (list, tuple)):
        combined = {}
        for h in history:
            for k, v in h.history.items():
                combined.setdefault(k, []).extend(v)
    elif isinstance(history, dict):
        combined = history
    else:
        combined = history.history

    serializable = {k: [float(x) for x in v] for k, v in combined.items()}
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"History saved to {filepath}")


def load_history(filepath: str) -> dict:
    """Load saved history dict from JSON. Pass to plot_learning_curves()."""
    with open(filepath, "r") as f:
        return json.load(f)