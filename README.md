# 🫘 NephroScan AI — Kidney Stone Detection CNN

> **Author:** Devaguru  
> **Last Updated:** February 2026  
> **Status:** ✅ All 6 Phases Complete  
> **Live API:** `http://localhost:8000/docs`

---

## 🏆 Results at a Glance

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| AUC-ROC | **1.0000** | ≥ 0.95 | 🔥 Exceeded |
| Sensitivity | **1.0000** | ≥ 0.92 | 🔥 Exceeded |
| Specificity | **0.9917** | ≥ 0.88 | 🔥 Exceeded |
| F2-Score | **0.9877** | ≥ 0.90 | 🔥 Exceeded |
| False Negatives | **0** | Minimise | 🔥 Zero missed stones |
| False Positives | **14** | < 5% of negatives | ✅ 0.83% |

> **Model:** EfficientNet-B4 · **Test set:** 1,904 images · **Zero missed stones across entire test set**

---

## 📊 Project Progress

| Phase | Description | Status | Duration |
|-------|-------------|--------|----------|
| 1 | Data Acquisition & Label Verification | ✅ Complete | ~2 Days |
| 2 | Model Training | ✅ Complete | ~3 Days |
| 3 | Evaluation & Explainability | ✅ Complete | ~2 Days |
| 4 | API Development (FastAPI) | ✅ Complete | ~1 Day |
| 5 | UI Development (HTML/CSS/JS) | ✅ Complete | ~1 Day |
| 6 | Monitoring (Prometheus) | ✅ Complete | ~1 Day |

---

## 📁 Project Structure

```
kidney-stone-cnn/
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app — 5 endpoints + Prometheus metrics
│   ├── inference.py          # KidneyStonePredictor — loads model once at startup
│   ├── metrics.py            # Custom Prometheus metric definitions
│   └── schemas.py            # Pydantic request/response schemas
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset class
│   │   ├── datamodule.py     # DataLoaders + WeightedRandomSampler
│   │   └── augmentations.py  # Albumentations train/val transforms
│   ├── models/
│   │   └── efficientnet.py   # EfficientNet-B4 + custom classification head
│   ├── training/
│   │   ├── losses.py         # Focal Loss (γ=2.0, α=0.75)
│   │   ├── metrics.py        # Sensitivity, AUC, F2, confusion matrix
│   │   └── trainer.py
│   └── evaluation/
│       ├── gradcam.py        # Grad-CAM++ heatmap generation
│       ├── error_analysis.py # False positive/negative visualisation
│       └── calibration.py    # Threshold optimisation + calibration curve
│
├── scripts/
│   ├── organize_data.py      # Maps 4-class → binary labels
│   ├── preprocess_data.py    # Resize to 224×224 + CLAHE
│   ├── split_data.py         # Deterministic train/val/test split
│   ├── generate_annotations.py
│   ├── verify_labels.py      # 5-check automated QA
│   ├── train.py              # Full training loop with MLflow
│   ├── export_onnx.py        # Export model to ONNX (20× CPU speedup)
│   ├── test_api.py           # Automated API test suite
│   └── generate_report.py    # Auto-generates clinical HTML report
│
├── notebooks/
│   ├── 01_eda.ipynb          # Phase 1 — Exploratory data analysis
│   ├── 02_training.ipynb     # Phase 2 — Training monitoring
│   └── 03_gradcam.ipynb      # Phase 3 — Grad-CAM visualisations
│
├── monitoring/
│   ├── docker-compose.yml    # Prometheus + Grafana containers
│   └── prometheus.yml        # Scrape config pointing to /metrics
│
├── checkpoints/
│   ├── best_model.pth        # PyTorch checkpoint (val AUC = 1.0, epoch 7)
│   └── best_model.onnx       # ONNX export (20× faster on CPU)
│
├── reports/
│   ├── clinical_report.html  # Full clinical evaluation report
│   ├── model_card.md         # Regulatory model documentation
│   ├── gradcam_stone.png
│   ├── gradcam_no_stone.png
│   ├── false_positives.png
│   ├── threshold_curve.png
│   └── calibration_curve.png
│
├── data/
│   ├── external/             # Raw downloaded datasets (never modified)
│   ├── processed/            # Clean 224×224 preprocessed images
│   └── labels/               # splits.csv, annotations.json, QA reports
│
├── mlruns/                   # MLflow experiment tracking
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── nephroscan.html           # Single-file web dashboard UI
├── requirements.txt          # Full training + serving dependencies
├── requirements_api.txt      # API-only dependencies (for Docker)
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

**Label mapping:**
| Original Class | Mapped To | Reason |
|---------------|-----------|--------|
| Stone | `stone` | Direct positive class |
| Cyst | `no_stone` | Different condition |
| Normal | `no_stone` | Healthy kidney |
| Tumor | `no_stone` | Different pathology |

### Dataset 2 — Kidney Ultrasound Dataset
| Field | Detail |
|-------|--------|
| Source | Kaggle — Kidney Stone Ultrasound Image Dataset |
| URL | kaggle.com/datasets/safurahajiheidari/kidney-stone-ultrasound-image-dataset |
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

### Preprocessing
| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | Resize | 224 × 224 pixels, Lanczos interpolation |
| 2 | CLAHE | clipLimit=4.0, tileGridSize=(8,8) |
| 3 | Format | Saved as JPEG, BGR→RGB corrected |

### Label Verification (5 automated checks)
| Check | Result | Detail |
|-------|--------|--------|
| Class balance | WARNING (expected) | 8.1:1 imbalance — handled in Phase 2 |
| Duplicate detection | WARNING (expected) | 2,579 sequential CT slice groups — not true duplicates |
| Corrupt / blank images | PASSED | 0 corrupt, 0 blank found |
| Train/test leakage | PASSED | No filename appears in both splits |
| Image size consistency | PASSED | All images exactly (224, 224) |

### Split Strategy
Deterministic MD5 filename hashing — same split every run, no random seed dependency, 70/15/15 distribution.

```python
def stable_hash(filename: str) -> float:
    h = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0
```

---

## ✅ Phase 2 — Model Training

### Architecture
| Component | Detail |
|-----------|--------|
| Backbone | EfficientNet-B4 (pretrained ImageNet) |
| Head | AdaptiveAvgPool → BN → Dropout(0.4) → Linear(1792→512) → GELU → Dropout(0.3) → Linear(512→2) |
| Parameters | 18,471,242 |
| Loss | Focal Loss (γ=2.0, α=0.75) |
| Optimiser | AdamW — backbone lr=1e-4, head lr=1e-3 |
| Scheduler | CosineAnnealingLR |
| Device | Apple MPS (MacBook Air M-series) |

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

---

## ✅ Phase 3 — Evaluation & Explainability

- **Grad-CAM++** heatmaps confirm model focuses on kidney anatomy, not image artifacts
- **14 false positives** analysed — cysts, vascular calcifications, compression artifacts
- **Threshold calibration** using F2-score on validation set
- **Clinical report** auto-generated at `reports/clinical_report.html`

```bash
open reports/clinical_report.html
```

---

## ✅ Phase 4 — FastAPI Inference Server

### Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single image → prediction + optional Grad-CAM |
| `POST` | `/predict/batch` | Up to 10 images in one request |
| `GET` | `/health` | Server + model status |
| `GET` | `/model-info` | Architecture, parameters, metrics |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/metrics` | Prometheus scrape endpoint |

### Sample Response
```json
{
  "prediction": "stone",
  "confidence": 0.9988,
  "probability_stone": 0.9988,
  "probability_no_stone": 0.0012,
  "gradcam_heatmap": "<base64 PNG or null>",
  "model_version": "efficientnet_b4_v1",
  "threshold_used": 0.5
}
```

### ONNX Export
| Runtime | Latency per image |
|---------|-------------------|
| PyTorch (MPS) | 482.6ms |
| ONNX (CPU) | 23.6ms |
| **Speedup** | **20.4×** |

### Start the Server
```bash
cd '/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn'
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Quick Test
```bash
cp "/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn/data/processed/test/stone/Stone- (1004).jpg" /tmp/test_stone.jpg

curl -X POST "http://localhost:8000/predict?include_gradcam=false" \
  -F "file=@/tmp/test_stone.jpg"
```

---

## ✅ Phase 5 - NephroScan Dashboard

Open `nephroscan.html` in your browser — no install needed. Requires FastAPI on `http://localhost:8000`.

Features: drag & drop upload, stone/no-stone verdict, confidence bars, Grad-CAM heatmap, prediction history, model status badge.

---

## ✅ Phase 6 — Prometheus Monitoring

### Metrics at `/metrics`
| Metric | Type | Description |
|--------|------|-------------|
| `kidney_predictions_total` | Counter | Total predictions labelled by class |
| `kidney_confidence_score` | Histogram | Distribution of confidence scores |
| `kidney_inference_latency_seconds` | Histogram | Per-request inference time |
| `kidney_model_loaded` | Gauge | 1 = loaded, 0 = unloaded |
| `kidney_active_requests` | Gauge | Requests currently being processed |
| `http_requests_total` | Counter | Total HTTP requests (auto) |

### Start Monitoring Stack
```bash
cd monitoring/
docker compose up -d
# Prometheus: http://localhost:9090
```

### Useful PromQL Queries
```promql
sum(kidney_predictions_total)
kidney_confidence_score_sum / kidney_confidence_score_count
rate(kidney_inference_latency_seconds_sum[5m]) / rate(kidney_inference_latency_seconds_count[5m]) * 1000
rate(http_requests_total[5m]) * 60
```

---

## 🚀 Full Reproduction Guide

```bash
# 1. Clone and enter project
git clone <repo-url>
cd kidney-stone-cnn

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets into data/external/

# 5. Phase 1 — Data pipeline
python scripts/organize_data.py
python scripts/preprocess_data.py
python scripts/split_data.py
python scripts/generate_annotations.py
python scripts/verify_labels.py

# 6. Phase 2 — Train (~90 min on Apple MPS)
python scripts/train.py

# 7. Phase 3 — Evaluate
# Run notebooks/03_gradcam.ipynb
python scripts/generate_report.py

# 8. Phase 4 — API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 9. Phase 5 — Dashboard
`http://localhost:8000`

# 10. Phase 6 — Monitoring
cd monitoring && docker compose up -d
```

---

## ⚠️ Known Limitations

1. **No patient-level split** — Kaggle dataset has no patient IDs. Sequential CT slices may appear in both train and test, potentially inflating metrics. External validation recommended before clinical use.
2. **AUC = 1.0 caveat** — Likely reflects CT slice similarity between splits. Not indicative of true generalisation on unseen scanner data.
3. **Low stone image count** — Only 952 stone training images. Rare variants (< 3mm) may be underdetected.
4. **No API authentication** — Do not expose port 8000 publicly without adding auth middleware.
5. **No bounding box annotations** — Classification only. Localisation deferred to a future phase.
6. **CT-heavy dataset** — Model performance on ultrasound should be evaluated on a dedicated ultrasound test set.

---

## 📄 License

Datasets used under CC BY 4.0. Model weights and code — Internal Research Project.

---

*NephroScan AI · Kidney Stone Detection CNN · Devaguru · February 2026*