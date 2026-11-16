"""Grad-CAM heatmaps: which image regions drive the prediction for the true class.

    python gradcam.py --weights checkpoints/densenet121.pth
    python gradcam.py --weights checkpoints/custom_cnn.pth --split test --num-images 12
"""

import argparse
from pathlib import Path

import torch

from brain_tumor.data import SEED, get_test_loader, get_train_val_loaders, resolve_data_dir
from brain_tumor.models import MODEL_NAMES, gradcam_target_layer, load_checkpoint
from brain_tumor.visualize import save_gradcam_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, help="only needed for weights saved without metadata")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, name, classes = load_checkpoint(args.weights, device, args.model)
    data_dir = resolve_data_dir(args.data_dir)
    if args.split == "val":
        _, loader, _ = get_train_val_loaders(data_dir, batch_size=32, seed=args.seed, num_workers=0)
    else:
        loader, _ = get_test_loader(data_dir, batch_size=64, num_workers=0)
        # the test folder is ordered by class: draw a shuffled batch instead of the first one
        loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=True,
                                             generator=torch.Generator().manual_seed(args.seed))
    images, labels = next(iter(loader))

    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gradcam_{args.split}.png"
    save_gradcam_grid(model, gradcam_target_layer(model, name), images, labels, classes, device,
                      name, path, num_images=args.num_images, seed=args.seed)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
