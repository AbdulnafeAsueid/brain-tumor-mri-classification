"""Export saved weights to ONNX, e.g. for a web app with onnxruntime-web.

    python export_onnx.py --weights checkpoints/densenet121.pth      # -> checkpoints/densenet121.onnx

The model takes a (batch, 3, 224, 224) float tensor normalized with the ImageNet
mean and standard deviation, and returns 4 logits in the order
glioma, meningioma, notumor, pituitary. Needs `pip install onnx onnxscript`.
"""

import argparse
from pathlib import Path

import torch

from brain_tumor.data import IMAGE_SIZE
from brain_tumor.models import MODEL_NAMES, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, help="only needed for weights saved without metadata")
    parser.add_argument("--out", default=None, help="output file (default: next to the weights)")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    model, name, classes = load_checkpoint(args.weights, "cpu", args.model)
    out = Path(args.out) if args.out else Path(args.weights).with_suffix(".onnx")
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(model, dummy, str(out), export_params=True, opset_version=args.opset,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    print(f"Exported {name} to {out} (classes: {classes})")


if __name__ == "__main__":
    main()
