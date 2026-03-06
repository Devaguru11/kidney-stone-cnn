# 🫘 NephroScan AI
### Unified Kidney Analysis — 3-Model Deep Learning Pipeline

> **Author:** Devaguru · **March 2026** · **Status:** ✅ Complete  
> **Stack:** EfficientNet-B4 · FastAPI · Grad-CAM++ · ReportLab · Apple MPS

---

## What This Does

Upload one kidney CT scan. Get results from 3 AI models simultaneously.

```
CT Scan
  ↓
v1 · Stone Detector          binary · 99.2% accuracy · AUC 1.0000
  ↓
v2 · 4-Class Classifier      Normal / Cyst / Stone / Tumour · 97.0% accuracy
  ↓
v3 · Cancer Detector         Cancer / Not Cancer · AUC 0.9999 · Precision 100%
  ↓
Risk Level + Grad-CAM Heatmap + Clinical PDF Report
```

---

## Results at a Glance

| Model | Task | Accuracy | AUC | Key Metric |
|-------|------|----------|-----|------------|
| **v1** | Stone Detection | **99.2%** | **1.0000** | 0 missed stones |
| **v2** | 4-Class Classification | **97.0%** | **0.9984** | Tumour recall 92.7% |
| **v3** | Cancer Detection | **99.6%** | **0.9999** | Precision **100%** |

### v2 — 4-Class Training
| Epoch | Acc | AUC | Tumour Recall | Note |
|-------|-----|-----|--------------|------|
| 1 | 71.4% | 0.9041 | 67.0% | Backbone frozen |
| 3 | 76.0% | 0.9278 | 63.4% | Backbone frozen |
| 4 | 92.0% | 0.9929 | 96.7% | ← Backbone unfrozen |
| 5 | 95.6% | 0.9960 | 88.8% | Fine-tuning |
| **6** | **97.2%** | **0.9986** | **92.7%** | ← **Best checkpoint** |

### v3 — Cancer Detector Training
| Epoch | Acc | AUC | Cancer Recall | Precision | Note |
|-------|-----|-----|--------------|-----------|------|
| 1 | 94.4% | 0.9983 | 99.4% | 77.9% | Backbone frozen |
| 4 | 99.4% | 0.9998 | 97.2% | 99.4% | ← Backbone unfrozen |
| **5** | **99.6%** | **0.9999** | **98.0%** | **100.0%** | ← **Best checkpoint** |

---

## Quick Start

```bash
# 1. Activate environment
cd '/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn'
source .venv/bin/activate

# 2. Start the unified API
uvicorn api.unified_main:app --port 8000 --reload
```

Expected output:
```
Loading 3 models on mps...
  v1 stone detector loaded ✅
  v2 4-class classifier loaded ✅
  v3 cancer detector loaded ✅
All 3 models ready ✅
```

```bash
# 3. Open the dashboard
open nephroscan_unified.html
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | CT scan → all 3 model results + Grad-CAM heatmap |
| `POST` | `/report` | Generate downloadable clinical PDF |
| `GET` | `/health` | Server status + device (MPS/CUDA/CPU) |
| `GET` | `/model-info` | Architecture, parameters, accuracy per model |
| `GET` | `/docs` | Interactive Swagger UI |

### Sample Response

```json
{
  "v1": {
    "prediction": "no_stone",
    "has_stone": false,
    "confidence": 0.9821,
    "probabilities": { "stone": 0.0179, "no_stone": 0.9821 }
  },
  "v2": {
    "prediction": "normal",
    "confidence": 0.9614,
    "color": "#2E7D32",
    "clinical_note": "No abnormality detected. Routine follow-up recommended.",
    "probabilities": { "normal": 0.9614, "cyst": 0.021, "stone": 0.011, "tumour": 0.006 }
  },
  "v3": {
    "prediction": "not_cancer",
    "is_cancer": false,
    "cancer_prob": 0.0012,
    "confidence": 0.9988
  },
  "risk_level": "NORMAL",
  "gradcam_heatmap": "data:image/png;base64,..."
}
```

---

## Architecture

All 3 models share the same backbone:

```
EfficientNet-B4 (pretrained ImageNet)
  → BatchNorm1d(1792)
  → Dropout(0.4)
  → Linear(1792 → 512)
  → GELU
  → BatchNorm1d(512)
  → Dropout(0.3)
  → Linear(512 → N)      ← N=2 for v1/v3, N=4 for v2
```

**Training strategy (v2 & v3):**
- Epochs 1–3: backbone frozen, head only (`lr=1e-3`)
- Epochs 4+: full fine-tune (backbone `lr=1e-4`, head `lr=1e-3`)
- Loss: Focal Loss (`γ=2.0`)
- Optimiser: AdamW + WeightedRandomSampler
- Inference: Temperature scaling (`T=0.5`)
- v3 backbone initialised from v2 weights → faster convergence

---

## Dataset

**Source:** [CT Kidney Dataset — Normal, Cyst, Tumor, Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) · CC BY 4.0

| Split | Normal | Cyst | Stone | Tumour | Total |
|-------|--------|------|-------|--------|-------|
| Train | 5,077 | 1,800 | 952 | 2,079 | **9,908** |
| Val | 1,089 | 386 | 204 | 446 | **2,125** |
| Test | 1,089 | 386 | 224 | 446 | **2,145** |
| **Total** | **7,255** | **2,572** | **1,380** | **2,971** | **12,446** |

**Preprocessing:** Resize 224×224 (Lanczos) → CLAHE (`clipLimit=4.0`) → BGR→RGB  
**Split:** Deterministic MD5 filename hash — 70/15/15, no random seed dependency

**v3 label mapping:**

| v2 Class | v3 Label |
|----------|----------|
| Tumour | ✅ Cancer |
| Normal | ❌ Not Cancer |
| Cyst | ❌ Not Cancer |
| Stone | ❌ Not Cancer |

---

## Project Structure

```
kidney-stone-cnn/
├── api/
│   ├── unified_main.py        ← FastAPI app · port 8000 · all 3 models
│   ├── unified_inference.py   ← Loads 3 models + Grad-CAM++
│   ├── unified_report.py      ← Clinical PDF generation (ReportLab)
│   ├── main.py                ← Legacy v1 API
│   └── inference.py           ← Legacy v1 inference
│
├── checkpoints/
│   ├── best_model.pth         ← v1 · binary stone · 99.2% acc
│   ├── best_model_v2.pth      ← v2 · 4-class · 97.0% acc
│   └── best_model_v3.pth      ← v3 · cancer · AUC 0.9999
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   ├── 03_gradcam.ipynb
│   ├── 05_train_v2.ipynb
│   └── 06_cancer_detection.ipynb
│
├── src/                       ← data · models · training · evaluation
├── scripts/                   ← preprocess · split · verify · export
├── data/                      ← processed 224×224 CT images
├── reports/                   ← grad-cam · calibration · model card
├── monitoring/                ← Prometheus + Grafana
│
├── nephroscan_unified.html    ← Unified dark dashboard (v3)
├── nephroscan.html            ← Legacy v1 dashboard
└── requirements.txt
```

---

## Dashboard Features

- Dark UI — navy/teal design, animated scan ring loader
- Drag & drop CT scan upload
- 3 model result cards side by side — confidence bars, probability breakdown
- Risk banner — `NORMAL` / `LOW` / `MEDIUM` / `HIGH`
- Grad-CAM heatmap — original scan vs AI focus area
- Prediction history table
- Download clinical PDF report per scan

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| Research only | Not validated for clinical use. Findings must be reviewed by a clinician. |
| CT scans only | Trained on CT images. Performance on ultrasound/MRI not validated. |
| No patient split | Dataset has no patient IDs — CT slices may appear in train and test. |
| No API auth | Do not expose port 8000 publicly without authentication middleware. |
| Kidney only | Single organ. Pan-cancer detection requires organ-specific models. |

---

## License

Dataset: CC BY 4.0  
Code & weights: Internal Research Project

---

> ⚠️ **This project is for research and portfolio purposes only.**  
> It is not approved for clinical use and must not be used to make medical decisions.  
> All AI outputs require review by a qualified healthcare professional.

---

*NephroScan AI · Devaguru · March 2026 · 3 Models · 12,446 Images · EfficientNet-B4*