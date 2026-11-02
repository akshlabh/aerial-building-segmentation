# Aerial Building Footprint Segmentation
## U-Net (ResNet34) vs SegFormer-B0 | SpaceNet 2 Paris

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision benchmark comparing Convolutional Neural Networks (**U-Net with ResNet34**) against Vision Transformers (**SegFormer-B0**) for binary building footprint segmentation from high-resolution aerial GeoTIFF imagery.

---

## Benchmark Results (Held-Out Test Set)

> **Note:** Fill these placeholders directly from the printed output of **Cell 9** after running the notebook on Kaggle.

| Metric | U-Net (ResNet34) | SegFormer-B0 | Relative Winner |
|---|---|---|---|
| **Test IoU (Jaccard)** | `[FILL: unet_iou]` | `[FILL: sf_iou]` | `[FILL: winner_iou]` |
| **Test Dice (F1 Score)** | `[FILL: unet_dice]` | `[FILL: sf_dice]` | `[FILL: winner_dice]` |
| **Precision** | `[FILL: unet_prec]` | `[FILL: sf_prec]` | `[FILL: winner_prec]` |
| **Recall** | `[FILL: unet_rec]` | `[FILL: sf_rec]` | `[FILL: winner_rec]` |
| **Trainable Parameters** | `[FILL: unet_params]` (~24.4M) | `[FILL: sf_params]` (~3.7M) | SegFormer (6.6× lighter) |
| **Inference Latency** (512² patch) | `[FILL: unet_ms]` ms | `[FILL: sf_ms]` ms | `[FILL: faster_model]` |
| **Training Time** (Kaggle GPU) | `[FILL: unet_train_min]` min | `[FILL: sf_train_min]` min | `[FILL: faster_train]` |

### Distribution Drift & Robustness (Atmospheric Haze = 0.30)
- **Population Stability Index (PSI):** `[FILL: psi_score]` (`[FILL: drift_status]`)
- **Kolmogorov-Smirnov Test:** Statistic = `[FILL: ks_stat]`, $p$-value = `[FILL: ks_pval]`
- **IoU Degradation:** Under severe noise/haze, U-Net drops by `[FILL: unet_drop_pct]`% vs SegFormer `[FILL: sf_drop_pct]`%.

---

## Key Engineering & Architecture Choices

### 1. Leakage-Free Tile-Level Splitting
- Aerial scenes cannot be naively tiled into patches prior to splitting: adjacent crops share identical roof textures, building structures, and illumination angles.
- We partition the dataset **strictly by image tile** (70% train / 15% validation / 15% test). The test set consists entirely of unseen geographical scenes.

### 2. Loss Function: Balanced BCE + Soft Dice
$$\mathcal{L} = 0.5 \times \mathcal{L}_{\text{BCEWithLogits}} + 0.5 \times \mathcal{L}_{\text{SoftDice}}$$
- **BCE:** Provides well-calibrated, smooth pixel-level probability gradients.
- **Soft Dice:** Directly optimizes spatial overlap, addressing the heavy background-to-building class imbalance (often > 80% background).

### 3. SegFormer-B0 Decoder Resolution Alignment
- Hugging Face's `SegformerForSemanticSegmentation` decoder naturally outputs logits at an **output stride of 4** ($H/4 \times W/4$).
- We implement bilinear interpolation to upsample decoder logits to full input dimensions ($512 \times 512$) before loss calculation, preventing label blurring.

### 4. Overlapping Sliding-Window Full-Tile Inference
- Full satellite tiles are processed using overlapping $512 \times 512$ windows with a $50\%$ stride ($256$ pixels).
- Overlapping predictions are averaged in logit space prior to sigmoid activation, eliminating grid boundary seams.

### 5. Automated Distribution Drift Detection
- Synthetic covariate shifts (brightness changes, atmospheric haze, sensor Gaussian noise) are applied to test sets.
- Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov tests on feature distributions quantitatively flag out-of-distribution batches before inference errors cascade in production.

---

## Kaggle Execution Instructions

1. **Create a Kaggle Notebook:**
   - Go to [kaggle.com/code](https://kaggle.com/code) -> **New Notebook**.
2. **Attach SpaceNet 2 Paris Dataset:**
   - Click **+ Add Data** (top right) -> search for dataset: `ugorjiir/spacenet-2-paris-buildings`.
3. **Configure Environment:**
   - **Accelerator:** GPU (T4 or P100)
   - **Internet:** On (required for pip packages & pretrained backbone weights)
4. **Upload and Run:**
   - Upload `aerial_segmentation.ipynb` and click **Run All**.
   - Total runtime is capped under **2.5 hours** via mixed precision (`torch.cuda.amp`), iteration caps (`100` iters/epoch), and early stopping (`patience=8`).
5. **Download Outputs:**
   - Run the final cell (**Cell 10**) to generate and download `aerial_segmentation_artifacts.zip` containing all evaluation plots, metrics, and logs.

---

## Repository Structure

```
aerial/
├── aerial_segmentation.ipynb   # Complete 10-stage Kaggle benchmark notebook
├── README.md                   # Benchmark results and methodology documentation
└── .gitignore                  # Git ignore rules for checkpoints, cache, and outputs
```

---

## Output Artifacts (Generated in `/kaggle/working/`)

| Artifact | Purpose |
|---|---|
| `aerial_segmentation_artifacts.zip` | One-click bundle containing all outputs below |
| `test_results.json` | Final quantitative metrics (IoU, Dice, Precision, Recall, Speed) |
| `drift_results.json` | PSI score, KS statistic, and robustness degradation sweep |
| `sanity_check.png` | Visual verification of GeoJSON polygon rasterization alignment |
| `unet_curves.png` | Training and validation loss/IoU convergence curves for U-Net |
| `segformer_curves.png` | Training and validation convergence curves for SegFormer |
| `error_analysis.png` | 6 worst-case test scenes with Ground Truth vs False Positives/Negatives |
| `robustness_curves.png` | IoU vs shift strength across brightness, haze, and noise perturbations |
| `drift_distributions.png` | Histogram comparing clean vs shifted image intensity distributions |
| `unet_best.pth` | PyTorch checkpoint of best performing U-Net model |
| `segformer_best.pth` | PyTorch checkpoint of best performing SegFormer model |
| `split.json` | Exact list of train/val/test tile IDs for reproducible replication |
