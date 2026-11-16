# Brain Tumor MRI Classification

Classifying brain MRI scans into **glioma, meningioma, pituitary tumor or no
tumor** with a custom convolutional neural network and three fine-tuned
ImageNet models (ResNet50, DenseNet121, EfficientNet-B0), with **Grad-CAM**
heatmaps to check which image regions drive each prediction.

The best model, **DenseNet121, reaches 95.5% accuracy on the 1,600-image test
set**, against 90% for the custom CNN. Grad-CAM showed the custom CNN often
looked at the skull and background rather than the tumor, while DenseNet121
mostly focused on the tumor.

Manual reading of MRI scans is slow and needs specialists, and some tumor types
(gliomas and meningiomas) look alike. This project explores an automated
second opinion. It is a course project, **not a medical device**.

## Repository structure

```
brain-tumor-mri-classification/
├── brain_tumor/              # Python package
│   ├── data.py               # dataset download, transforms, train/validation split, loaders
│   ├── models.py             # custom CNN, transfer learning models, checkpoints
│   ├── engine.py             # training loop (LR scheduling, early stopping), evaluation
│   └── visualize.py          # training curves, confusion matrices, Grad-CAM, reports
├── train.py                  # train models, evaluate on validation and test sets
├── evaluate.py               # evaluate saved weights on the test set
├── gradcam.py                # Grad-CAM heatmaps for saved weights
├── predict.py                # classify your own images
├── export_onnx.py            # export weights to ONNX (e.g. for a web app)
├── explore.py                # class counts, example images, augmented samples
├── notebooks/
│   └── brain_tumor_mri_classification.ipynb   # original notebook code, with outputs
└── requirements.txt
```

## Installation

Python 3.9 or newer. A GPU is strongly recommended for training (the original
experiments ran on a Google Colab GPU).

```bash
git clone https://github.com/AbdulnafeAsueid/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For a GPU build of PyTorch, install `torch` and `torchvision` first with the
command from https://pytorch.org/get-started/locally/.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
by Masoud Nickparvar (Kaggle), built from several public MRI collections.
The scripts download it automatically with `kagglehub` (about 160 MB); to use
a copy you already have, pass `--data-dir PATH` (the folder that contains
`Training/` and `Testing/`).

| Split | Glioma | Meningioma | No tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| `Training` | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| `Testing` | 400 | 400 | 400 | 400 | 1,600 |

`Training` is split 80/20 (fixed seed 42) into 4,480 training and 1,120
validation images. `Testing` is used only for the final evaluation.

## Usage

```bash
python explore.py                                   # class counts, example and augmented images
python train.py --model densenet121                 # final model
python train.py --model custom_cnn
python train.py --model all                         # all four; reports the best by validation loss
python evaluate.py --weights checkpoints/densenet121.pth
python gradcam.py  --weights checkpoints/densenet121.pth            # validation images
python gradcam.py  --weights checkpoints/custom_cnn.pth --split test
python predict.py  --weights checkpoints/densenet121.pth my_scan.jpg
python export_onnx.py --weights checkpoints/densenet121.pth         # -> checkpoints/densenet121.onnx
```

`train.py` saves the weights to `checkpoints/<model>.pth` and, in
`outputs/<model>/`, the training curves, the training history, and for both the
validation and the test set a classification report, metrics (JSON) and a
confusion matrix. `--epochs`, `--patience`, `--lr` and `--batch-size` override
the defaults below.

The notebook in `notebooks/` holds the original experiment code with its
outputs.

## Methods

**Preprocessing.** Images are resized to 224×224 and normalized with the
ImageNet mean and standard deviation. Training images get only mild
augmentation (random horizontal flip, rotation up to 10°): MRI scans are taken
in a controlled position, and strong augmentation would create anatomically
unrealistic images.

**Custom CNN.** Three blocks of 3×3 convolution, ReLU and 2×2 max pooling (32,
64 and 128 filters), then a 128-unit dense layer and a 4-class output.

**Transfer learning.** ResNet50, DenseNet121 and EfficientNet-B0 with ImageNet
weights; the last layer is replaced by a 4-class output and **all layers are
fine-tuned**. ImageNet was preferred over medical pre-training such as chest
X-rays, which is a different imaging modality.

**Training.** Cross-entropy loss and Adam. The learning rate is divided by 10
when the validation loss stops improving (`ReduceLROnPlateau`), and training
stops early when it has not improved for `patience` epochs, keeping the
weights with the lowest validation loss. The model with the lowest validation
loss is selected as the final model.

| Model | Learning rate | Max epochs | Early stopping patience | LR scheduler patience |
|---|---:|---:|---:|---:|
| Custom CNN | 1e-3 | 10 | 3 | 2 |
| ResNet50, DenseNet121, EfficientNet-B0 | 1e-4 | 20 | 5 | 3 |

**Explainability.** Grad-CAM on the last convolutional layer (`conv3` of the
custom CNN, the last feature block of the ImageNet models), for the true class.

## Results

From the original notebook run (Google Colab GPU).

**Validation set** (1,120 images; lowest validation loss during training):

| Model | Best validation loss | Validation accuracy | Epoch |
|---|---:|---:|---:|
| Custom CNN | 0.1681 | 94.3% | 6 |
| ResNet50 | 0.0207 | 99.2% | 10 |
| **DenseNet121** | **0.0198** | **99.6%** | 13 |
| EfficientNet-B0 | 0.0311 | 98.9% | 9 |

**Test set** (1,600 unseen images, 400 per class):

| Model | Accuracy | Glioma (P / R) | Meningioma (P / R) | No tumor (P / R) | Pituitary (P / R) |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 90.3% | 0.91 / 0.78 | 0.91 / 0.84 | 0.87 / 1.00 | 0.93 / 0.99 |
| **DenseNet121** | **95.5%** | 1.00 / 0.83 | 0.89 / 0.99 | 0.95 / 1.00 | 1.00 / 1.00 |

P = precision, R = recall. ResNet50 and EfficientNet-B0 were not evaluated on
the test set in the original run; `train.py` now reports it for every model.

- **DenseNet121** classifies all 400 no-tumor scans, 399 pituitary and 398
  meningioma scans correctly and never predicts glioma wrongly, but misses 69
  of the 400 gliomas: 46 called meningioma and 23 called no tumor.
- **The custom CNN** misses 22% of gliomas, 47 of them called no tumor, and
  confuses gliomas and meningiomas in both directions.

**Grad-CAM.** On the custom CNN, most heatmaps highlighted the skull edge,
background or ventricles instead of the tumor, so its 90% accuracy partly
rests on cues unrelated to the pathology (a "Clever Hans" effect). On
DenseNet121 the heatmaps mostly covered the tumor regions, and for no-tumor
scans spread over the brain. This is a visual check on a few images, not a
quantitative localization measure.

## Known limitations

- **Missed gliomas.** Even the best model calls 23 of 400 gliomas "no tumor",
  the most harmful kind of error for screening.
- **Validation accuracy is optimistic.** The `Training` folder contains 171
  repeated images, and with the 80/20 split 67 of the 1,120 validation images
  (mostly no-tumor) have an identical copy among the training images. This also
  affects model selection. No test image is an exact copy of a training image
  (checked by file hash; near-duplicates were not checked).
- Single 2D slices from a public dataset, of mixed orientations and MRI
  sequences; no patient-level information or external validation.
- Results vary somewhat between runs, since GPU training is not fully
  deterministic.

## Future work

- Reduce glioma false negatives (e.g. class weighting or a lower decision
  threshold for "no tumor").
- Skull stripping, so models cannot rely on the skull or background.
- Attention mechanisms, and quantitative Grad-CAM evaluation against tumor masks.
- Remove duplicates before splitting, and test on an external dataset.

## Authors

- **Muhammad Asad Ejaz**
- **Abdulnafe Asueid**
- **Mohammed Rumaan**

## License

The code is released under the [MIT License](LICENSE). The dataset is not part
of this repository and keeps its own license on Kaggle.

## Acknowledgements

Dataset: Masoud Nickparvar, *Brain Tumor MRI Dataset*, Kaggle. Grad-CAM via
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
