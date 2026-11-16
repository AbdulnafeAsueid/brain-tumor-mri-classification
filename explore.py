"""Look at the data: images per class and split, one example per class, and augmented training samples.

    python explore.py [--data-dir PATH] [--out outputs/data]
"""

import argparse
import os
from pathlib import Path

from torchvision import datasets

from brain_tumor.data import SEED, base_transform, get_train_val_loaders, resolve_data_dir
from brain_tumor.visualize import save_batch, save_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="outputs/data")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args.data_dir)
    for split in ("Training", "Testing"):
        counts = {c: len(os.listdir(os.path.join(data_dir, split, c)))
                  for c in sorted(os.listdir(os.path.join(data_dir, split)))}
        print(f"{split}: {sum(counts.values())} images {counts}")

    dataset = datasets.ImageFolder(os.path.join(data_dir, "Training"), transform=base_transform)
    save_examples(dataset, dataset.classes, out / "examples_per_class.png")
    train_loader, _, classes = get_train_val_loaders(data_dir, seed=SEED, num_workers=0)
    save_batch(train_loader, classes, "Augmented training samples", out / "augmented_samples.png")
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
