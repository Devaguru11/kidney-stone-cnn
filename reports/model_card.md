reports/model_card.md content
# Model Card — Kidney Stone Detection CNN

## Model Details
- **Architecture:** EfficientNet-B4 + custom classification head
- **Task:** Binary image classification (stone / no_stone)
- **Version:** 1.0.0
- **Checkpoint:** checkpoints/best_model.pth
- **Parameters:** 18,471,242
- **Training date:** February 2026

## Intended Use
- **Primary use:** Screening aid for radiologists reviewing CT and ultrasound
- **Intended users:** Radiologists and clinical AI researchers
- **Out of scope:** Autonomous diagnosis, emergency triage, paediatric cases

## Training Data
- Kaggle CT Kidney Dataset: 12,446 images (Stone/Cyst/Normal/Tumor)
- Kaggle Kidney Stone Ultrasound Dataset
- Train/Val/Test split: 70 / 15 / 15 (by filename hash)
- Stone images in training set: 952
- Imbalance ratio: 8.1:1 (handled with Focal Loss + WeightedRandomSampler)

## Performance — Test Set (1,904 images)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sensitivity | 1.0000 | >= 0.92 | PASSED |
| Specificity | 0.9917 | >= 0.88 | PASSED |
| AUC-ROC | 1.0000 | >= 0.95 | PASSED |
| F2-Score | 0.9877 | >= 0.90 | PASSED |
| False Negatives | 0 | Minimise | ZERO |

## Limitations
1. No patient-level split — sequential CT slices may appear in train and test
2. AUC=1.0 may reflect data leakage — requires external validation
3. Only 952 stone training images — small stone variants may be underdetected
4. Not validated on DICOM data from real hospital scanners
5. No paediatric or pregnancy cases in training data

## Ethical Considerations
- Model is advisory only — all predictions require radiologist confirmation
- No patient data is stored in model weights
- Training data is fully anonymised Kaggle datasets
- Bias audit across stone size subgroups not yet performed

## How to Use
```python
from src.models.efficientnet import KidneyStoneClassifier
import torch
model = KidneyStoneClassifier()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()
# Pass preprocessed 224x224 tensor, get logits
```

