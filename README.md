# 🫘 Kidney Stone Detection — Phase 1: Data Acquisition & Label Verification

> **Status:** ✅ Complete  
> **Duration:** ~2 Days  
> **Last Updated:** February 2026  
> **Author:** devaguru

---

## 📋 Phase Overview

Phase 1 covers everything needed to go from a blank project to a clean, verified, split dataset ready for model training. No model code is written in this phase — the sole focus is **data quality**.

> A model trained on bad data will always produce bad predictions, no matter how good the architecture is. Phase 1 exists to prevent that.

---

## 📁 Final Folder Structure (After Phase 1)

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
│           ├── qa_report.txt            # Automated QA check results
│           ├── duplicates.json          # Duplicate groups (CT slices — expected)
│           ├── class_distribution.png   # EDA chart
│           ├── sample_images.png        # Visual inspection grid
│           └── intensity_dist.png       # Pixel intensity histogram
│
├── scripts/
│   ├── organize_data.py                 # Maps 4-class → binary labels
│   ├── preprocess_data.py              # Resize to 224×224 + CLAHE
│   ├── split_data.py                   # Train/val/test split
│   ├── generate_annotations.py         # Generates annotations.json
│   └── verify_labels.py               # 5-check automated QA
│
├── notebooks/
│   └── 01_eda.ipynb                    # Exploratory data analysis
│
├── requirements.txt
└── README.md                           # This file
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

## 📊 Final Dataset Statistics

### Image Counts After Splitting

| Split | Stone | No-Stone | Total | Stone % |
|-------|-------|----------|-------|---------|
| Train | 952 | 7,728 | 8,680 | 11.0% |
| Val | 201 | 1,661 | 1,862 | 10.8% |
| Test | 224 | 1,680 | 1,904 | 11.8% |
| **Total** | **1,377** | **11,069** | **12,446** | **11.1%** |

### Class Imbalance
- **Ratio:** 8.0:1 (no_stone : stone)
- **Assessment:** Manageable — under the 10:1 danger threshold
- **Mitigation plan (Phase 2):** Focal Loss (γ=2.0, α=0.75) + WeightedRandomSampler

---

## ⚙️ Preprocessing Applied

Every image in `data/processed/` has had the following applied in order:

| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | Resize | 224 × 224 pixels, Lanczos interpolation |
| 2 | CLAHE | clipLimit=4.0, tileGridSize=(8,8) |
| 3 | Format | Saved as JPEG, BGR→RGB corrected for display |

All preprocessing is handled by `scripts/preprocess_data.py`.

---

## 🔍 Label Verification Results

Automated QA run via `scripts/verify_labels.py` — 5 checks performed on all 12,446 images.

| Check | Result | Detail |
|-------|--------|--------|
| ✅ Class balance | WARNING (expected) | 8.1:1 imbalance — handled in Phase 2 |
| ✅ Duplicate detection | WARNING (expected) | 2,579 sequential CT slice groups — not true duplicates |
| ✅ Corrupt / blank images | PASSED | 0 corrupt, 0 blank found |
| ✅ Train/test leakage | PASSED | No filename appears in both train and test |
| ✅ Image size consistency | PASSED | All 500 sampled images are exactly (224, 224) |

### Note on "Duplicates"
The 2,579 duplicate groups detected by pHash are **consecutive CT scan slices** from the same patient (e.g., `Stone-(816).jpg` and `Stone-(817).jpg`). These are genuinely different images — adjacent cross-sectional slices of the same kidney. pHash flags them as similar because they are visually near-identical, which is expected for sequential scan slices. **No images were deleted.**

A proper fix would require patient-level metadata to group all slices per patient before splitting — this Kaggle dataset does not provide patient IDs. This is noted as a known limitation.

---

## ✂️ Split Strategy

Splits were assigned using **deterministic filename hashing** (MD5) rather than random shuffling. This ensures:

- The same split is reproduced every time the script runs
- No randomness dependency on a seed
- Approximately 70 / 15 / 15 distribution

```python
# From scripts/split_data.py
def stable_hash(filename: str) -> float:
    h = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0
```

**Known limitation:** Without patient IDs, slices from the same patient may appear in both train and test splits. This is a limitation of the Kaggle dataset, not the split methodology. The leakage check confirmed no *identical* images appear across splits.

---

## 🚀 How to Reproduce Phase 1 From Scratch

```bash
# 1. Clone the repo and enter the project
git clone <repo-url>
cd kidney-stone-cnn

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets from Kaggle into data/external/
#    kidney_kaggle/  and  kidney_ultrasound/

# 5. Run the pipeline in order
python scripts/organize_data.py
python scripts/preprocess_data.py
python scripts/split_data.py
python scripts/generate_annotations.py
python scripts/verify_labels.py

# 6. Launch EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📦 Dependencies

```
pydicom==2.4.3
SimpleITK==2.3.1
opencv-python==4.9.0.80
Pillow==10.2.0
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
imagehash==4.3.1
matplotlib==3.8.2
seaborn==0.13.2
tqdm==4.66.2
albumentations==1.3.1
jupyter==1.0.0
ipykernel
pyarrow
```

Install all: `pip install -r requirements.txt`

---

## ⚠️ Known Limitations

1. **No patient-level split** — Kaggle dataset provides no patient IDs, so consecutive CT slices from the same patient may appear in both train and test sets. This may cause slight overfitting to appear better than it is at test time.

2. **Low stone image count** — Only 1,377 stone images across the full dataset. The model may struggle with rare stone variants (very small stones <3mm, faint calcifications). Adding TCIA data in a later phase is recommended.

3. **No bounding box annotations** — The current annotations.json contains only image-level labels. Stone localization (bounding boxes) requires manual annotation using Label Studio or CVAT, which is deferred to a later phase.

4. **CT-heavy dataset** — The majority of images are CT scans. Ultrasound images are underrepresented. Model performance on ultrasound may be lower and should be evaluated separately.

---

## ➡️ Next Phase

**Phase 2 — Model Training** starts with:
- Building the PyTorch `Dataset` and `DataModule` classes that load from `data/processed/`
- Setting up the EfficientNet-B4 backbone with transfer learning
- Configuring Focal Loss and WeightedRandomSampler to handle the 8:1 imbalance
- Running the first training experiment with MLflow tracking

See `notebooks/03_baseline_model.ipynb` to begin.

---

*Kidney Stone Detection CNN — Internal Research Project*