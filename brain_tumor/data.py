"""Dataset download, transforms, train/validation split and data loaders."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

IMAGE_SIZE = 224
# ImageNet statistics, also used by the pre-trained transfer learning models.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SEED = 42


def resolve_data_dir(data_dir=None):
    """Folder that holds `Training/` and `Testing/`. Downloads the dataset with kagglehub if none is given."""
    if data_dir is None:
        import kagglehub

        data_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    for split in ("Training", "Testing"):
        if not os.path.isdir(os.path.join(data_dir, split)):
            raise FileNotFoundError(f"{data_dir} has no '{split}' folder")
    return data_dir


# Images are first loaded as 224x224 tensors in [0, 1] ...
base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ... then the training images get mild augmentation, MRI scans are taken in a
# controlled position, so strong augmentation would create unrealistic images.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Validation images are only normalized.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Test images (and single images for prediction) are loaded straight from disk.
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


class ApplyTransform(Dataset):
    """Applies its own transform to a subset, so training and validation can differ after `random_split`."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def default_workers():
    return min(4, (os.cpu_count() or 2) // 2)


def get_train_val_loaders(data_dir, batch_size=32, val_fraction=0.2, seed=SEED, num_workers=None):
    """80/20 split of the `Training` folder (fixed seed). Returns train loader, val loader and class names."""
    num_workers = default_workers() if num_workers is None else num_workers
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Training"), transform=base_transform)
    train_size = int((1 - val_fraction) * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(ApplyTransform(train_subset, train_transform), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ApplyTransform(val_subset, val_transform), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset.classes


def get_test_loader(data_dir, batch_size=32, num_workers=None):
    """The separate `Testing` folder, never used for training or model selection."""
    num_workers = default_workers() if num_workers is None else num_workers
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Testing"), transform=test_transform)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader, test_dataset.classes


def unnormalize(image):
    """Normalized (C, H, W) tensor -> (H, W, C) numpy image in [0, 1] for plotting."""
    img = image.detach().cpu().numpy().transpose((1, 2, 0))
    return np.clip(np.array(STD) * img + np.array(MEAN), 0, 1)
