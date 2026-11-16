"""Evaluate saved weights on the test set (classification report and confusion matrix).

    python evaluate.py --weights checkpoints/densenet121.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from brain_tumor.data import get_test_loader, resolve_data_dir
from brain_tumor.engine import predict
from brain_tumor.models import MODEL_NAMES, load_checkpoint
from brain_tumor.visualize import save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, help="only needed for weights saved without metadata")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name, classes = load_checkpoint(args.weights, device, args.model)
    test_loader, _ = get_test_loader(resolve_data_dir(args.data_dir), args.batch_size, args.workers)
    preds, labels, loss = predict(model, test_loader, device, nn.CrossEntropyLoss())
    metrics, text = save_report(labels, preds, classes, Path(args.out) / name, "test", f"{name}: test set", loss)
    print(f"Test images: {len(labels)} | loss {loss:.4f} | accuracy {metrics['accuracy']:.4f}\n{text}")
    print(f"Results saved to {Path(args.out) / name}")


if __name__ == "__main__":
    main()
