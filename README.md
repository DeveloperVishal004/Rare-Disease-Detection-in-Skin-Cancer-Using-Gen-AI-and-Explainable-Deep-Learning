# 🔬 Skin Cancer Classification
### ResNet18 · DCGAN Augmentation · Explainable AI

> *Solving extreme class imbalance in dermoscopic image classification through a rigorous ablation study — from weighted loss penalties to generative synthetic augmentation, validated with Grad-CAM interpretability.*

---

## 📋 Table of Contents

- [Overview](#overview)
- [The Problem: Class Imbalance](#the-problem-class-imbalance)
- [Dataset](#dataset)
- [Methodology: Ablation Study](#methodology-ablation-study)
- [Results](#results)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Medical datasets suffer from severe class imbalance — a reality that silently causes deep learning models to ignore rare diseases while appearing highly accurate. This project addresses this failure head-on using the **HAM10000 dermoscopic dataset**, focusing on the extreme minority class **Dermatofibroma (df)**, which has a **70:1 imbalance ratio** against the majority class.

Rather than jumping to a single solution, this project is structured as a **four-stage ablation study**, isolating the true contribution of each intervention:

1. **Baseline ResNet18** — observing how a standard model fails
2. **Weighted Loss** — mathematical intervention to force attention
3. **Classical Augmentation** — testing the limits of geometric transforms
4. **DCGAN Hybrid Pipeline** — generative synthetic data as the solution

Finally, **Grad-CAM** is applied to the best model to verify that predictions are grounded in actual lesion morphology, not spurious background correlations.

---

## The Problem: Class Imbalance

In imbalanced datasets, neural networks take the **path of least mathematical resistance**. A model trained on raw HAM10000 data can achieve **77.63% overall accuracy** by simply predicting the majority class — while being completely blind to rare, potentially dangerous conditions.

Standard accuracy metrics **hide this failure entirely**.

```
Majority class (Melanocytic Nevi):   4,683 images   ████████████████████████████████ 
Extreme minority (Dermatofibroma):      71 images   ▌
                                                     ──────────────── 70:1 ratio
```

This project demonstrates why **Macro F1-Score** — which treats every class equally regardless of sample size — is the correct metric for imbalanced medical classification.

---

## Dataset

**HAM10000** — *Human Against Machine with 10,000 Training Images*

A large public collection of dermoscopic skin lesion images across **7 disease classes**.

| Class | Medical Name | Train Images | Status |
|:---:|:---|:---:|:---|
| `nv` | Melanocytic Nevi | 4,683 | Extreme Majority |
| `mel` | Melanoma | 773 | Minority |
| `bkl` | Benign Keratosis | 772 | Minority |
| `bcc` | Basal Cell Carcinoma | 361 | Minority |
| `akiec` | Actinic Keratoses | 222 | Severe Minority |
| `vasc` | Vascular Lesions | 99 | Severe Minority |
| `df` | **Dermatofibroma** | **71** | **Extreme Minority** |

**Final Split (lesion-level stratified):**

| Split | Images | Purpose |
|:---|:---:|:---|
| Train | 6,981 | Model training |
| Validation | 1,532 | Hyperparameter tuning |
| Test | 1,502 | Final locked evaluation |

### ⚠️ Data Leakage Prevention

The HAM10000 dataset contains multiple photographs of the **same physical lesion** (tracked by `lesion_id`). A naïve random split would allow the model to memorize patients rather than learn diseases — inflating test metrics and producing a system that fails on new patients.

**Solution: Lesion-Level Stratified Split**

```
Step 1: Group dataset by unique lesion_id (physical bumps, not photos)
Step 2: Stratified split at the lesion level (70% / 15% / 15%)
Step 3: Map lesion assignments back to all corresponding images
```

This guarantees a strict firewall — no patient appears in both training and test sets.

---

## Methodology: Ablation Study

### Architecture: Modified ResNet18 with Transfer Learning

```
INPUT IMAGE (128×128×3)
        │
        ▼
┌─────────────────────────────────────────────┐
│           FROZEN BACKBONE                   │  ← Pre-trained on ImageNet
│  Conv1 → BN → ReLU → MaxPool               │
│  Layer 1 (2 ResBlocks)                      │
│  Layer 2 (2 ResBlocks)                      │
│  Layer 3 (2 ResBlocks)                      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│           UNFROZEN LAYER 4                  │  ← Fine-tuned for dermatology
│  Layer 4 (2 ResBlocks)                      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        CUSTOM CLASSIFICATION HEAD           │  ← Newly initialized
│  Global Average Pooling                     │
│  FC(512→128) → Dropout(0.5) → ReLU         │
│  FC(128→7) → Softmax                        │
└─────────────────────────────────────────────┘
        │
        ▼
  7-CLASS OUTPUT
```

- **Optimizer:** AdamW with ReduceLROnPlateau scheduler  
- **Batch Size:** 32  
- **Image Resolution:** 128 × 128 pixels  

---

### Experiment 1 — Baseline ResNet18

Standard ResNet18 with frozen backbone, fine-tuned Layer 4, and **Standard Cross-Entropy Loss** (equal penalty for all mistakes).

**Hypothesis:** The model will achieve high overall accuracy but fail on the minority class by taking the path of least resistance.

---

### Experiment 2 — Weighted Loss

Replaced Standard Cross-Entropy with **inverse-frequency class weights** to make rare-class mistakes mathematically expensive.

**Applied Weights:**

| Class | Weight |
|:---:|:---:|
| `df` (Dermatofibroma) | **14.0463** |
| `vasc` | 10.0736 |
| `akiec` | 4.4923 |
| `bcc` | 2.7626 |
| `bkl` | 1.2918 |
| `mel` | 1.2901 |
| `nv` (Nevus) | **0.2130** |

Missing a Dermatofibroma is now **66× more costly** than missing an ordinary mole.

---

### Experiment 2.5 — Classical Augmentation

Combined weighted loss with a **heavy geometric augmentation pipeline**:

- Horizontal & Vertical Flips
- Random Rotation (±20°)
- Random Translation
- Color Jitter (brightness, contrast, saturation)
- Affine Transforms (scaling & shearing)

**Key Insight Being Tested:** Does generating *volume* from existing images equal generating *diversity*?

---

### Experiment 3 — DCGAN Hybrid Pipeline

Trained a **Deep Convolutional GAN** on the 71 Dermatofibroma images to generate 400 new synthetic images with morphological variation.

#### Why DCGAN (Not Diffusion / StyleGAN)?

Modern high-capacity generative models are data-hungry — they require tens of thousands of images to learn a meaningful distribution. On a 71-image micro-dataset, they simply memorize and reproduce. DCGAN's lighter architecture can be engineered to extract genuine morphological variation from minimal data.

#### GAN Training Challenges & Stabilization

Training any GAN on 71 images leads to immediate collapse:

| Failure Mode | Description |
|:---|:---|
| **Discriminator Memorization** | Discriminator memorizes all 71 real images, giving Generator zero gradient signal |
| **Discriminator Domination** | Generator loss explodes; feedback becomes meaningless |
| **Mode Collapse** | Generator finds one winning image and repeats it endlessly |

**Stabilization interventions applied:**

```python
# 1. Discriminator Dropout ("Amnesia")
#    Randomly disables 30% of Discriminator neurons — prevents pixel-perfect memorization

# 2. One-Sided Label Smoothing
#    Real image labels: 1.0 → 0.9
#    Prevents Discriminator from becoming overconfident; smooths gradients

# 3. Two Time-Scale Update Rule (TTUR)
#    Discriminator LR: 0.00005  (slow)
#    Generator LR:     0.00030  (fast)

# 4. Asymmetric Update Ratio (3:1)
#    Generator updates 3× per 1 Discriminator update
#    Gives Generator a fighting chance against the faster-learning Discriminator
```

#### Hybrid Dataset Construction

```
Original training set:    71 real Dermatofibroma images
DCGAN generated:        + 400 synthetic images
                        ─────────────────────────────
Hybrid Dermatofibroma:   471 total images

Imbalance ratio:  70:1  →  ~10:1  (extreme minority → standard minority)
```

Synthetic images were **mixed with** real images (not used alone) to anchor the classifier in real morphology while providing synthetic structural variation.

---

## Results

| Metric | Exp 1: Baseline | Exp 2: Weighted Loss | Exp 2.5: Classical Aug | Exp 3: DCGAN Hybrid |
|:---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | 0.7763 | 0.7417 | 0.6844 | **0.7723** |
| **DF Precision** | 0.0000 | 0.5000 | 0.2692 | **0.4211** |
| **DF Recall** | 0.0000 | 0.5000 | 0.7000 | 0.4000 |
| **DF F1-Score** | 0.0000 | 0.5000 | 0.3889 | 0.4103 |
| **Macro F1-Score** | 0.5307 | 0.6019 | 0.5705 | **0.6166** ✅ |

### Reading the Results

**Why Experiment 3 wins despite lower recall than Experiment 2.5:**

Experiment 2.5's 0.70 recall was not a clean improvement — it was the result of the model *over-predicting* the rare class due to lack of morphological diversity. The network panicked and flagged anything remotely ambiguous as Dermatofibroma, collapsing precision to 0.2692 and overall accuracy to 68.44%.

Experiment 3 provided genuine synthetic variation, allowing the model to form **calibrated decision boundaries**. The result:

- ✅ Precision recovered: `0.2692 → 0.4211`
- ✅ Overall accuracy recovered: `0.6844 → 0.7723`  
- ✅ **Macro F1 reached project-high: 0.6166**
- ⬇️ Recall decreased: `0.7000 → 0.4000` — but this is *selectivity*, not failure

> The goal was never to maximize recall at the expense of the system. The goal was **balanced classification** — and Experiment 3 achieved it.

---

## Explainable AI (Grad-CAM)

High metrics on a spreadsheet cannot generate clinical trust. **Gradient-weighted Class Activation Mapping (Grad-CAM)** was applied to the final model to verify that predictions are grounded in lesion morphology.

### Why Grad-CAM?

| XAI Method | Issue |
|:---|:---|
| LIME | Super-pixel boundaries don't align with organic lesion shapes |
| SHAP | Computationally prohibitive; produces noisy static on dense color images |
| Saliency Maps | Highlights random edges and background noise |
| **Grad-CAM** | ✅ Targets final convolutional layer; smooth spatial heatmaps; single forward/backward pass |

**Target Layer:** `layer4[-1]` — the deepest spatial reasoning block, before the classification head.

### Key Visual Findings

**✅ Lesion-Anchored Attention**  
Bright red "hot zones" consistently fell over the physical skin lesion, not the image corners or background.

**✅ Zero Clever Hans Behavior**  
Body hair, lighting glare, and background skin tones remained in the cold/blue zones. The model ignored spurious correlations.

**✅ Focus on Diagnostically Relevant Features**  
Attention traced outer lesion boundaries and pooled on darkly pigmented cores — the exact features targeted by our synthetic augmentation.

**✅ Clinically Plausible Mistakes**  
When the model misclassified a Benign Keratosis (bkl) as Melanoma (mel), the Grad-CAM heatmap revealed the hot zone was concentrated on a **dark, highly asymmetrical, jagged border** — a legitimate clinical warning sign of melanoma. The model was wrong, but for the right reason.

---

## Project Structure

```
skin-cancer-classification/
│
├── skincancerclassification.ipynb     # Main notebook — all 4 experiments + Grad-CAM
│
├── data/
│   └── HAM10000/                      # Dataset (download separately — see below)
│       ├── HAM10000_metadata.csv
│       ├── HAM10000_images_part_1/
│       └── HAM10000_images_part_2/
│
├── models/
│   ├── baseline_resnet18.pth          # Experiment 1 weights
│   ├── weighted_loss_resnet18.pth     # Experiment 2 weights
│   ├── classical_aug_resnet18.pth     # Experiment 2.5 weights
│   ├── dcgan_generator.pth            # Trained DCGAN Generator
│   └── hybrid_resnet18.pth            # Experiment 3 weights (final model)
│
├── outputs/
│   ├── synthetic_images/              # 400 DCGAN-generated Dermatofibroma images
│   ├── gradcam_heatmaps/              # Grad-CAM visualizations
│   └── confusion_matrices/            # Per-experiment confusion matrices
│
└── README.md
```

---

## Installation & Setup

### Prerequisites

```bash
Python >= 3.8
CUDA-compatible GPU (recommended) or Kaggle/Colab environment
```

### Install Dependencies

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn Pillow tqdm
pip install opencv-python
```

Or install all at once:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow tqdm opencv-python
```

### Download the Dataset

The HAM10000 dataset is available publicly via Kaggle:

```
https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
```

After downloading, place the images and metadata CSV in the `data/HAM10000/` directory as shown in the project structure above.

---

## Usage

The entire pipeline is contained within the notebook `skincancerclassification.ipynb`. Run cells sequentially:

### 1. Data Preparation & Leakage Prevention
```python
# Lesion-level stratified split
# Ensures no patient appears in both train and test sets
split_dataset_by_lesion(metadata_df, seed=42)
```

### 2. Train Baseline Model
```python
model = build_resnet18(num_classes=7, freeze_backbone=True)
train(model, criterion=nn.CrossEntropyLoss(), ...)
```

### 3. Train with Weighted Loss
```python
class_weights = compute_inverse_frequency_weights(train_df)
criterion = nn.CrossEntropyLoss(weight=class_weights)
train(model, criterion=criterion, ...)
```

### 4. Train DCGAN & Generate Synthetic Images
```python
generator, discriminator = build_dcgan()
train_dcgan(
    real_images=df_images,          # 71 Dermatofibroma images
    epochs=1000,
    discriminator_dropout=0.3,      # Amnesia
    label_smoothing=0.9,            # One-sided smoothing
    lr_G=0.00030,                   # TTUR
    lr_D=0.00005,
    update_ratio=3                  # Generator updates 3× per Discriminator update
)
synthetic_images = generator.generate(n=400)
```

### 5. Train Hybrid Model
```python
hybrid_train_df = inject_synthetic_images(train_df, synthetic_images)
train(model, data=hybrid_train_df, ...)
```

### 6. Generate Grad-CAM Heatmaps
```python
gradcam = GradCAM(model, target_layer=model.layer4[-1])
heatmap = gradcam.generate(image, target_class=predicted_class)
visualize_overlay(image, heatmap)
```

---

## Key Findings

| Finding | Lesson |
|:---|:---|
| Baseline achieved 77.63% accuracy with **zero** minority recall | Overall accuracy is a dangerous illusion in imbalanced medical data |
| Weighted loss raised DF recall to 0.50 but couldn't teach what the class *looks like* | Mathematical incentives ≠ visual information |
| Classical augmentation pushed recall to 0.70 but collapsed precision to 0.27 | Volume ≠ morphological diversity. A rotated tumor is still the same tumor |
| DCGAN hybrid achieved **Macro F1: 0.6166** — the project-best balanced score | Synthetic visual variation restores system equilibrium |
| Grad-CAM confirmed lesion-anchored attention with zero Clever Hans artifacts | Explainability is essential — metrics alone are insufficient for clinical trust |

---

## Limitations

- **Single dataset:** HAM10000 is biased toward lighter skin tones; generalizability to all demographics is unverified
- **Single random seed (seed=42):** No multi-seed statistical analysis; margin of error is not established
- **No K-Fold cross-validation:** Computationally prohibitive given the generative pipeline
- **No external validation dataset:** Model was not tested on images from different clinical systems
- **No FID/IS evaluation:** Mathematically unstable on a 71-image reference set; synthetic quality assessed functionally
- **No dermatologist review:** Synthetic images are not clinically validated — they are morphological buffers, not medical ground truth
- **Grad-CAM shows spatial attention, not clinical reasoning:** Hot zones over a border ≠ proof the model understands asymmetry as a concept
- **Not ready for clinical deployment:** Requires multi-center trials, bias auditing, and regulatory approval before touching real patient care

---

## Future Work

- **Larger, multi-dataset training** across global populations for demographic fairness
- **Advanced generative models** (StyleGAN, Diffusion Models) when sufficient base data exists
- **Multi-seed statistical analysis** to establish rigorous confidence intervals
- **Combined XAI methods** — Grad-CAM + SHAP + Integrated Gradients for pixel-level precision
- **Vision Transformer architectures** or Hybrid CNN-Transformer systems
- **Multi-modal input** — combining image analysis with patient age, sex, lesion location, and medical history
- **Formal FID/IS evaluation** on a supplemented dataset of sufficient size

---

## References

- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR.
- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised representation learning with deep convolutional generative adversarial networks.* arXiv:1511.06434.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV.
- Heusel, M., et al. (2017). *GANs trained by a two time-scale update rule converge to a local Nash equilibrium.* NeurIPS. *(TTUR)*

---

<div align="center">

</div>
