# WikiArt Artist Classification
Project for the Deep Learning course, Masters in Data Science and Advanced Analytics, NOVA IMS.

---

## Project Authors
Carolina Luz – 20250409 – [20250409@novaims.unl.pt](mailto:20250409@novaims.unl.pt) \
João Paulo de Ávila – 20250436 – [20250436@novaims.unl.pt](mailto:20250436@novaims.unl.pt) \
Lucas Ferreira – 20250448 – [20250448@novaims.unl.pt](mailto:20250448@novaims.unl.pt) \
Pedro Santos – 20250399 – [20250399@novaims.unl.pt](mailto:20250399@novaims.unl.pt) \
Pedro Fernandes – 20250418 – [20250418@novaims.unl.pt](mailto:20250418@novaims.unl.pt)

---

## Project Overview
The WikiArt dataset contains 13 340 JPEG images spanning 23 artist classes.
The objective of this project is to build deep learning models capable of identifying the author of a painting from image data alone, handling high intra-class variability, stylistic overlap between artists, and class imbalance.

---

## Project Goals
1. **Data preparation** – Near-duplicate removal using difference hashing, stratified 70–15–15 split, and augmentation pipeline tuned for high-resolution inputs.
2. **CNN from scratch** – Baseline model, Optuna hyperparameter search, and a residual block architecture with progressive capacity increases, label smoothing, and regularization.
3. **Transfer learning** – Fine-tuned VGG16, EfficientNetB3, EfficientNetB5, ResNet50, and EfficientNetV2S backbones with dual pooling heads and frozen batch normalization.
4. **Ensemble and TTA** – Weighted ensemble combining ResNet50 and EfficientNetV2S with test-time augmentation at inference.
5. **Interpretability** – Grad-CAM visualizations to inspect model attention and characterize misclassification patterns.

---

## Outcome
A final ensemble combining ResNet50 and EfficientNetV2S with TTA, evaluated on a held-out test set:

| Model | Test Accuracy | Test Macro F1 |
|---|---|---|
| Baseline CNN | 0.3042 | 0.2341 |
| Optuna CNN (Trial 21) | 0.5258 | 0.5142 |
| Residual Block CNN | 0.7266 | 0.7019 |
| VGG16 | 0.7102 | 0.6941 |
| EfficientNetB3 | 0.7247 | 0.7089 |
| EfficientNetB5 | 0.7490 | 0.7381 |
| ResNet50 | 0.8067 | 0.7918 |
| EfficientNetV2S | 0.8484 | 0.8294 |
| **Ensemble + TTA** | **0.8743** | **0.8670** |

---

## Notebooks
- **01_eda.ipynb** – data audit, near-duplicate detection, class distribution analysis, split strategy, and augmentation exploration
- **02_baseline.ipynb** – minimal CNN baseline establishing a lower bound for accuracy and macro F1
- **03_residual_block.ipynb** – Optuna hyperparameter search and residual block model progression with ablation experiments
- **04_ResNet50.ipynb** – two-phase transfer learning with ResNet50, dual pooling head, MixUp, label smoothing, and AdamW
- **05_EfficientNetV2S.ipynb** – two-phase transfer learning with EfficientNetV2S, same pipeline as ResNet50
- **06_VGGNet.ipynb** – transfer learning with VGG16 backbone
- **07_efficientNet_B3.ipynb** – transfer learning with EfficientNetB3 backbone
- **08_efficientNet_B5.ipynb** – transfer learning with EfficientNetB5 backbone
- **09_ensemble_and_analysis.ipynb** – weighted ensemble, TTA evaluation, Grad-CAM visualizations, and misclassification analysis

---

## Setup
### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
Open notebooks in Jupyter or VSCode and run cells sequentially.
