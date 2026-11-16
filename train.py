"""Train one or more models, evaluate them on the validation and test sets, and save the weights.

Examples:
    python train.py --model densenet121          # final model
    python train.py --model custom_cnn
    python train.py --model all                  # all four; the best by validation loss is reported

The dataset is downloaded with kagglehub unless --data-dir points to a folder
with `Training/` and `Testing/`. Weights go to checkpoints/<model>.pth, results
(curves, reports, confusion matrices) to outputs/<model>/.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from brain_tumor.data import SEED, get_test_loader, get_train_val_loaders, resolve_data_dir
from brain_tumor.engine import predict, train_model
from brain_tumor.models import MODEL_NAMES, TRAIN_SETTINGS, build_model, save_checkpoint
from brain_tumor.visualize import save_history, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=["densenet121"], choices=MODEL_NAMES + ["all"])
    parser.add_argument("--data-dir", default=None, help="folder with Training/ and Testing/ (default: download)")
    parser.add_argument("--epochs", type=int, default=None, help="maximum epochs (default: per model)")
    parser.add_argument("--patience", type=int, default=None, help="early stopping patience (default: per model)")
    parser.add_argument("--lr", type=float, default=None, help="learning rate (default: per model)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=None, help="data loading processes")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--checkpoints", default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = resolve_data_dir(args.data_dir)
    train_loader, val_loader, classes = get_train_val_loaders(
        data_dir, args.batch_size, seed=args.seed, num_workers=args.workers)
    test_loader, _ = get_test_loader(data_dir, args.batch_size, num_workers=args.workers)
    print(f"Training: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)} images, classes: {classes}")

    names = MODEL_NAMES if "all" in args.model else args.model
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in names:
        print(f"\n========== {name} ==========")
        settings = dict(TRAIN_SETTINGS[name])
        for key in ("epochs", "patience", "lr"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)

        model = build_model(name, num_classes=len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=settings["scheduler_patience"])
        model, history, best_val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            scheduler=scheduler, num_epochs=settings["epochs"], patience=settings["patience"])

        ckpt_path = Path(args.checkpoints) / f"{name}.pth"
        save_checkpoint(model, name, classes, ckpt_path)
        print(f"Saved weights to {ckpt_path}")

        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        save_history(history, f"Training history: {name}", out / "history.png")

        preds, labels, _ = predict(model, val_loader, device)
        val_metrics, _ = save_report(labels, preds, classes, out, "val", f"{name}: validation set")
        preds, labels, test_loss = predict(model, test_loader, device, criterion)
        test_metrics, text = save_report(labels, preds, classes, out, "test", f"{name}: test set", test_loss)
        print(f"\nTest set ({name}), loss {test_loss:.4f}:\n{text}")

        summary[name] = {"best_val_loss": best_val_loss, "val_accuracy": val_metrics["accuracy"],
                         "test_accuracy": test_metrics["accuracy"], "epochs_run": len(history["train_loss"])}

    print("\n========== Summary ==========")
    for name, s in summary.items():
        print(f"{name:16s} best val loss {s['best_val_loss']:.4f}  val acc {s['val_accuracy']:.4f}  "
              f"test acc {s['test_accuracy']:.4f}  ({s['epochs_run']} epochs)")
    if len(summary) > 1:
        best = min(summary, key=lambda n: summary[n]["best_val_loss"])
        print(f"Best model by validation loss: {best}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
