# 🫘 Kidney Stone Detection — CNN Project

> **Author:** devaguru  
> **Last Updated:** February 2026  
> **Overall Status:** Phases 1–3 Complete ✅ | Phase 4–6 Upcoming 🔄

---

## 📊 Project Progress

| Phase | Description | Status | Duration |
|-------|-------------|--------|----------|
| 1 | Data Acquisition & Label Verification | ✅ Complete | ~2 Days |
| 2 | Model Training & First Experiments | ✅ Complete | ~3 Days |
| 3 | Evaluation & Explainability | ✅ Complete | ~2 Days |
| 4 | API Development (FastAPI) | 🔄 Upcoming | — |
| 5 | Deployment (Docker + Kubernetes) | 🔄 Upcoming | — |
| 6 | Monitoring & MLOps | 🔄 Upcoming | — |

---

## 🏆 Key Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| AUC-ROC | **1.0000** | ≥ 0.95 | 🔥 Exceeded |
| Sensitivity | **1.0000** | ≥ 0.92 | 🔥 Exceeded |
| Specificity | **0.9917** | ≥ 0.88 | 🔥 Exceeded |
| F2-Score | **0.9877** | ≥ 0.90 | 🔥 Exceeded |
| False Negatives | **0** | Minimise | 🔥 Zero missed stones |
| False Positives | **14** | < 5% of negatives | ✅ 0.83% |

> **Model:** EfficientNet-B4 + custom classification head · **Test set:** 1,904 images · **Zero missed stones**

---

## 📁 Full Project Structure

```
kidney-stone-cnn/
├── data/
│   ├── external/                        # Raw downloaded datasets (never modified)
│   │   ├── kidney_kaggle/
│   │   │   ├── Stone/                   # 1,377 CT images (positive class)
│   │   │   ├── Cyst/                    # CT images (mapped → no_stone)
│   │   │   ├── Normal/                  # CT images (mapped → no_stone)
│   │   │   ├── Tumor/                   # CT images (mapped → no_stone)
│   │   │   └── kidneyData.csv
│   │   └── kidney_ultrasound/
│   │       ├── stone/                   # Ultrasound positives
│   │       └── Normal/                  # Ultrasound negatives
│   │
│   ├── processed/                       # Clean 224×224 preprocessed images
│   │   ├── train/
│   │   │   ├── stone/                   # 952 images
│   │   │   └── no_stone/               # 7,728 images
│   │   ├── val/
│   │   │   ├── stone/                   # 201 images
│   │   │   └── no_stone/               # 1,661 images
│   │   └── test/
│   │       ├── stone/                   # 224 images
│   │       └── no_stone/               # 1,680 images
│   │
│   └── labels/
│       ├── splits.csv                   # Train/val/test assignment per image
│       ├── annotations.json             # COCO-format metadata for all images
│       └── label_verification/
│           ├── qa_report.txt
│           ├── duplicates.json
│           ├── class_distribution.png
│           ├── sample_images.png
│           ├── intensity_dist.png
│           └── test_results.png         # ROC curve + confusion matrix
│
├── src/
│   ├── data/
│   │   ├── dataset.py                   # PyTorch Dataset class
│   │   ├── datamodule.py               # DataLoaders + WeightedRandomSampler
│   │   └── augmentations.py            # Albumentations train/val transforms
│   ├── models/
│   │   └── efficientnet.py             # EfficientNet-B4 + classification head
│   ├── training/
│   │   ├── losses.py                   # Focal Loss (γ=2.0, α=0.75)
│   │   ├── metrics.py                  # Sensitivity, AUC, F2, confusion matrix
│   │   └── trainer.py
│   └── evaluation/
│       ├── gradcam.py                  # Grad-CAM++ heatmap generation
│       ├── error_analysis.py           # False positive/negative visualisation
│       └── calibration.py             # Threshold optimisation + calibration curve
│
├── scripts/
│   ├── organize_data.py
│   ├── preprocess_data.py
│   ├── split_data.py
│   ├── generate_annotations.py
│   ├── verify_labels.py
│   ├── train.py                        # Full training loop with MLflow
│   └── generate_report.py             # Auto-generates clinical HTML report
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Phase 1 — Exploratory data analysis
│   ├── 02_training.ipynb               # Phase 2 — Training monitoring
│   └── 03_gradcam.ipynb               # Phase 3 — Grad-CAM visualisations
│
├── checkpoints/
│   └── best_model.pth                  # Best model (val AUC = 1.0, epoch 7)
│
├── reports/
│   ├── clinical_report.html            # Full clinical evaluation report
│   ├── model_card.md                   # Regulatory model documentation
│   ├── gradcam_stone.png
│   ├── gradcam_no_stone.png
│   ├── false_positives.png
│   ├── threshold_curve.png
│   └── calibration_curve.png
│
├── mlruns/                             # MLflow experiment tracking
├── requirements.txt
└── README.md
```

---

## 🗃️ Datasets Used

### Dataset 1 — CT Kidney Dataset (Primary)
| Field | Detail |
|-------|--------|
| Source | Kaggle — CT KIDNEY DATASET: Normal-Cyst-Tumor-Stone |
| URL | kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone |
| Total images | 12,446 |
| Format | JPEG, color |
| Original classes | Stone, Cyst, Normal, Tumor |
| License | CC BY 4.0 |

**Label mapping applied:**
| Original Class | Mapped To | Reason |
|---------------|-----------|--------|
| Stone | `stone` | Direct positive class |
| Cyst | `no_stone` | Different condition, not a stone |
| Normal | `no_stone` | Healthy kidney |
| Tumor | `no_stone` | Different pathology |

### Dataset 2 — Kidney Ultrasound Dataset
| Field | Detail |
|-------|--------|
| Source | Kaggle — Kidney Stone Ultrasound Image Dataset |
| URL | kaggle.com/datasets/safurahajiheidari/kidney-stone-ultrasound-image-dataset |
| Format | PNG/JPG |
| Classes | stone, Normal (already binary) |
| License | CC BY 4.0 |

---

## 📊 Dataset Statistics

| Split | Stone | No-Stone | Total | Stone % |
|-------|-------|----------|-------|---------|
| Train | 952 | 7,728 | 8,680 | 11.0% |
| Val | 201 | 1,661 | 1,862 | 10.8% |
| Test | 224 | 1,680 | 1,904 | 11.8% |
| **Total** | **1,377** | **11,069** | **12,446** | **11.1%** |

**Class imbalance:** 8.0:1 — handled with Focal Loss (γ=2.0, α=0.75) + WeightedRandomSampler

---

## ✅ Phase 1 — Data Acquisition & Label Verification

### Preprocessing Applied
| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | Resize | 224 × 224 pixels, Lanczos interpolation |
| 2 | CLAHE | clipLimit=4.0, tileGridSize=(8,8) |
| 3 | Format | Saved as JPEG, BGR→RGB corrected for display |

### Label Verification Results
| Check | Result | Detail |
|-------|--------|--------|
| ✅ Class balance | WARNING (expected) | 8.1:1 imbalance — handled in Phase 2 |
| ✅ Duplicate detection | WARNING (expected) | 2,579 sequential CT slice groups — not true duplicates |
| ✅ Corrupt / blank images | PASSED | 0 corrupt, 0 blank found |
| ✅ Train/test leakage | PASSED | No filename appears in both train and test |
| ✅ Image size consistency | PASSED | All sampled images are exactly (224, 224) |

### Split Strategy
Deterministic filename hashing (MD5) — same split every run, no random seed dependency, approximately 70/15/15 distribution.

```python
def stable_hash(filename: str) -> float:
    h = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0
```

### Known Limitation
Without patient IDs, slices from the same patient may appear in both train and test. The leakage check confirmed no *identical* images appear across splits.

---

## ✅ Phase 2 — Model Training & First Experiments

### Model Architecture
| Component | Detail |
|-----------|--------|
| Backbone | EfficientNet-B4 (pretrained on ImageNet) |
| Head | AdaptiveAvgPool → BN → Dropout(0.4) → Linear(1792→512) → GELU → Dropout(0.3) → Linear(512→2) |
| Parameters | 18,471,242 |
| Loss | Focal Loss (γ=2.0, α=0.75) |
| Optimiser | AdamW — backbone lr=1e-4, head lr=1e-3, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR |
| Device | Apple MPS (MacBook Air M-series) |

### Training Strategy
| Setting | Value | Reason |
|---------|-------|--------|
| Freeze backbone | Epochs 1–3 | Let head adapt to new task first |
| Unfreeze backbone | Epoch 4+ | Fine-tune entire network |
| Batch size | 8 | MPS memory constraint |
| Early stopping patience | 7 epochs | Stop if val AUC plateaus |
| Imbalance handling | WeightedRandomSampler | ~50/50 stone/no_stone per batch |

### Training Progress
| Epoch | AUC-ROC | Sensitivity | Note |
|-------|---------|-------------|------|
| 1 | 0.9086 | 0.9502 | Backbone frozen |
| 2 | 0.9296 | 0.9403 | Backbone frozen |
| 3 | 0.9578 | 0.9751 | Backbone frozen |
| 4 | 0.9965 | 0.9950 | Backbone unfrozen |
| 5 | 0.9996 | 0.9950 | Fine-tuning |
| 6 | 0.9998 | 0.9900 | Fine-tuning |
| **7** | **1.0000** | **1.0000** | **Converged — training stopped** |

### Final Test Set Results
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sensitivity | 1.0000 | ≥ 0.92 | 🔥 Exceeded |
| Specificity | 0.9917 | ≥ 0.88 | 🔥 Exceeded |
| AUC-ROC | 1.0000 | ≥ 0.95 | 🔥 Exceeded |
| Precision | 0.9412 | ≥ 0.85 | ✅ Passed |
| F2-Score | 0.9877 | ≥ 0.90 | 🔥 Exceeded |
| True Positives | 224 | — | All stones detected |
| False Negatives | 0 | Minimise | 🔥 Zero |
| False Positives | 14 | — | 0.83% of negatives |
| True Negatives | 1,666 | — | — |

---

## ✅ Phase 3 — Evaluation & Explainability

### Grad-CAM Visual Explanations
Grad-CAM++ heatmaps generated for stone and no_stone test images using the last EfficientNet backbone block as the target layer. Heatmaps confirm the model focuses on kidney and urinary tract anatomy rather than image artifacts or borders.

Charts: `reports/gradcam_stone.png`, `reports/gradcam_no_stone.png`

### False Positive Analysis
14 false positives identified and visualised with Grad-CAM overlays. Common patterns:
- Cysts with high radiodensity mimicking stones
- Vascular calcifications outside the kidney
- Image compression artifacts triggering dense-region detector

**Clinical impact:** All 14 FPs would trigger follow-up imaging — no patient harm. Zero false negatives means zero missed stones.

Chart: `reports/false_positives.png`

### Threshold Calibration
Optimal decision threshold found on validation set using F2-score (β=2, weights recall 2× over precision). Calibration curve confirms model probability estimates are well-calibrated.

Charts: `reports/threshold_curve.png`, `reports/calibration_curve.png`

### Clinical Report
Auto-generated HTML report at `reports/clinical_report.html`

```bash
open reports/clinical_report.html
```

---

## 🚀 How to Reproduce From Scratch

```bash
# 1. Clone and enter project
git clone <repo-url>
cd kidney-stone-cnn

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets into data/external/
#    kidney_kaggle/  and  kidney_ultrasound/

# 5. Phase 1 — Data pipeline
python scripts/organize_data.py
python scripts/preprocess_data.py
python scripts/split_data.py
python scripts/generate_annotations.py
python scripts/verify_labels.py

# 6. Phase 2 — Train model
python scripts/train.py
# Best model saved to checkpoints/best_model.pth

# 7. Phase 3 — Evaluate and explain
# Run notebooks/03_gradcam.ipynb in VS Code
python scripts/generate_report.py
open reports/clinical_report.html
```

---

## 📦 Dependencies

```
torch==2.2.0
torchvision==0.17.0
timm==0.9.16
pytorch-lightning==2.2.0
albumentations==1.3.1
mlflow==2.11.0
torchmetrics==1.3.1
grad-cam==1.5.0
opencv-python==4.9.0.80
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.4
matplotlib==3.8.2
seaborn==0.13.2
Pillow==10.2.0
tqdm==4.66.2
imagehash==4.3.1
pydicom==2.4.3
SimpleITK==2.3.1
jupyter==1.0.0
ipykernel
pyarrow
```

---

## ⚠️ Known Limitations

1. **No patient-level split** — Kaggle dataset has no patient IDs. Sequential CT slices from the same patient may appear in both train and test, which may inflate metrics. External validation recommended before clinical use.

2. **AUC = 1.0 caveat** — Perfect test score likely reflects CT slice similarity between splits rather than true generalisation. Must be validated on an independent external dataset before any clinical deployment.

3. **Low stone image count** — Only 952 stone training images. Rare stone variants (< 3mm, faint calcifications) may be underdetected. Adding TCIA data is recommended.

4. **No bounding box annotations** — Classification only. Stone localisation requires manual annotation via Label Studio or CVAT — deferred to a later phase.

5. **CT-heavy dataset** — Ultrasound images are underrepresented. Performance on ultrasound should be evaluated separately on a dedicated ultrasound test set.

---

## ➡️ Next — Phase 4: API Development

Phase 4 wraps the trained model in a **FastAPI REST endpoint**:
- `POST /predict` — accepts an image, returns JSON with prediction, confidence score, and Grad-CAM heatmap
- `POST /predict/batch` — batch inference endpoint
- `GET /health` — health check
- Containerised with Docker for consistent cross-platform deployment

---

*Kidney Stone Detection CNN — Internal Research Project*
