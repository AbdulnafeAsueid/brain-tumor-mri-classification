"""Classify one or more MRI images with saved weights.

    python predict.py --weights checkpoints/densenet121.pth path/to/scan.jpg [more images ...]
"""

import argparse

import torch
from PIL import Image

from brain_tumor.data import test_transform
from brain_tumor.models import MODEL_NAMES, load_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, help="only needed for weights saved without metadata")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name, classes = load_checkpoint(args.weights, device, args.model)
    for path in args.images:
        x = test_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x), dim=1)[0].cpu()
        best = int(probs.argmax())
        details = ", ".join(f"{c} {p:.3f}" for c, p in zip(classes, probs.tolist()))
        print(f"{path}: {classes[best]} ({details})")


if __name__ == "__main__":
    main()
