"""Plots: example images, training curves, confusion matrix, Grad-CAM heatmaps, and saved reports."""

import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import unnormalize


def save_examples(dataset, class_names, path):
    """One example image per class (dataset items must be (C, H, W) tensors in [0, 1])."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    shown = {}
    for img, label in dataset:
        if label not in shown:
            shown[label] = img
        if len(shown) == len(class_names):
            break
    for label, img in sorted(shown.items()):
        axes[label].imshow(img.permute(1, 2, 0).numpy())
        axes[label].set_title(f"Class: {class_names[label]}", fontsize=14, fontweight="bold")
        axes[label].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_batch(loader, class_names, title, path, n=4):
    """A few (augmented, normalized) images from a loader."""
    images, labels = next(iter(loader))
    plt.figure(figsize=(14, 4))
    for i in range(min(n, len(images))):
        plt.subplot(1, n, i + 1)
        plt.imshow(unnormalize(images[i]))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.suptitle(title)
    plt.savefig(path, dpi=100)
    plt.close()


def save_history(history, title, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 5))
    plt.suptitle(title, fontsize=16)
    for i, metric in enumerate(("loss", "acc")):
        plt.subplot(1, 2, i + 1)
        plt.plot(epochs, history[f"train_{metric}"], label=f"Train {metric}", marker="o")
        plt.plot(epochs, history[f"val_{metric}"], label=f"Validation {metric}", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss" if metric == "loss" else "Accuracy")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_confusion_matrix(labels, preds, class_names, title, path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_report(labels, preds, class_names, out_dir, prefix, title, loss=None):
    """Write <prefix>_report.txt, <prefix>_metrics.json and <prefix>_confusion_matrix.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = classification_report(labels, preds, labels=list(range(len(class_names))),
                                 target_names=class_names, digits=4, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "loss": loss,
        "report": classification_report(labels, preds, labels=list(range(len(class_names))),
                                        target_names=class_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
    }
    (out / f"{prefix}_report.txt").write_text(text)
    (out / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(labels, preds, class_names, title, out / f"{prefix}_confusion_matrix.png")
    return metrics, text


def save_gradcam_grid(model, target_layer, images, labels, class_names, device, title, path,
                      num_images=10, seed=None):
    """Original image next to its Grad-CAM heatmap for the true class, two pairs per row."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    num_images = min(num_images, len(images))
    indices = random.Random(seed).sample(range(len(images)), num_images)
    pairs_per_row = 2
    rows = math.ceil(num_images / pairs_per_row)
    fig, axes = plt.subplots(rows, 2 * pairs_per_row, figsize=(16, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for i, idx in enumerate(indices):
        row, col = i // pairs_per_row, (i % pairs_per_row) * 2
        true_label = int(labels[idx])
        heatmap = cam(input_tensor=images[idx:idx + 1].to(device),
                      targets=[ClassifierOutputTarget(true_label)])[0]
        rgb = unnormalize(images[idx]).astype("float32")
        axes[row, col].imshow(rgb)
        axes[row, col].set_title(f"Original ({class_names[true_label]})")
        axes[row, col + 1].imshow(show_cam_on_image(rgb, heatmap, use_rgb=True))
        axes[row, col + 1].set_title(f"Grad-CAM ({title})")

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
