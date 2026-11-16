"""Custom CNN, transfer learning models, their training settings and checkpoint loading."""

import torch
import torch.nn as nn
from torchvision import models

MODEL_NAMES = ["custom_cnn", "resnet50", "densenet121", "efficientnet_b0"]
TRANSFER_MODELS = MODEL_NAMES[1:]

# Training settings per model (as in the original experiments).
TRAIN_SETTINGS = {
    "custom_cnn": dict(epochs=10, patience=3, lr=1e-3, scheduler_patience=2),
}
for _name in TRANSFER_MODELS:
    # all layers are fine-tuned, with a small learning rate
    TRAIN_SETTINGS[_name] = dict(epochs=20, patience=5, lr=1e-4, scheduler_patience=3)


class BrainTumorCNN(nn.Module):
    """Three Conv(3x3) -> ReLU -> MaxPool blocks (32, 64, 128 filters) and a 128-unit dense layer."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # 224 -> 112 -> 56 -> 28 after three poolings
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)  # raw logits, CrossEntropyLoss applies softmax

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        return self.fc2(x)


def build_model(name, num_classes=4, pretrained=True):
    """Custom CNN, or an ImageNet model with its last layer replaced by a `num_classes` output."""
    if name == "custom_cnn":
        return BrainTumorCNN(num_classes)
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")
    return model


def gradcam_target_layer(model, name):
    """Last convolutional layer, used by Grad-CAM."""
    if name == "custom_cnn":
        return model.conv3
    if name == "resnet50":
        return model.layer4[-1]
    return model.features[-1]  # densenet121, efficientnet_b0


def save_checkpoint(model, name, classes, path):
    torch.save({"model": name, "classes": list(classes), "state_dict": model.state_dict()}, path)


def load_checkpoint(path, device="cpu", name=None):
    """Load a checkpoint saved by `train.py`. Returns (model, name, classes)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in ckpt:  # plain state_dict, e.g. saved from the notebook
        if name is None:
            raise ValueError("this file holds only weights: pass the model name too")
        ckpt = {"model": name, "classes": ["glioma", "meningioma", "notumor", "pituitary"], "state_dict": ckpt}
    model = build_model(ckpt["model"], num_classes=len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["model"], ckpt["classes"]
