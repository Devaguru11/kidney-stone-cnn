# api/inference_v2.py
import torch, cv2, numpy as np, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import timm
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Class definitions ─────────────────────────────────────────────────────────
# Temperature scaling — lower = more confident (0.3 very sharp, 1.0 = raw softmax)
TEMPERATURE = 0.5

CLASS_NAMES = ['normal', 'cyst', 'stone', 'tumour']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES  = 4

CLINICAL_NOTES = {
    'normal': 'No abnormality detected. Routine follow-up recommended.',
    'cyst':   'Kidney cyst detected. Usually benign. Bosniak classification advised.',
    'stone':  'Kidney stone detected. Urology referral recommended.',
    'tumour': 'Renal mass detected. Urgent urology and oncology referral recommended.',
}

CLASS_COLORS = {
    'normal': '#2E7D32',
    'cyst':   '#1565C0',
    'stone':  '#E65100',
    'tumour': '#B71C1C',
}

# ── Model ─────────────────────────────────────────────────────────────────────
class KidneyClassifierV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── Predictor ─────────────────────────────────────────────────────────────────
class KidneyPredictorV2:
    def __init__(self, checkpoint='checkpoints/best_model_v2.pth', img_size=224):
        self.img_size  = img_size
        self.transform = A.Compose([
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f'Loading v2 model on {self.device}...')
        self.model = KidneyClassifierV2().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()
        print('V2 model ready.')

    def predict(self, image_bytes: bytes) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Could not decode image')

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE
        lab = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

        tensor = self.transform(image=img)['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()

        top_idx   = int(probs.argmax())
        top_class = CLASS_NAMES[top_idx]

        return {
            'prediction':    top_class,
            'confidence':    round(float(probs[top_idx]), 4),
            'probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)},
            'clinical_note': CLINICAL_NOTES[top_class],
            'color':         CLASS_COLORS[top_class],
            'model_version': 'efficientnet_b4_v2_4class',
        }