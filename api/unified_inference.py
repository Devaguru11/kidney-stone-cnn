# api/unified_inference.py
# Loads all 3 models and runs unified inference

import torch, cv2, numpy as np, sys, base64
from pathlib import Path
from io import BytesIO
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import timm
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Shared transform ──────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_transform():
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_device():
    if torch.backends.mps.is_available():  return torch.device('mps')
    if torch.cuda.is_available():          return torch.device('cuda')
    return torch.device('cpu')

# ── Model 1 — Stone Detector (v1) ────────────────────────────────────────────
# Architecture matches original v1 saved weights exactly:
# head = [0:Flatten, 1:Dropout, 2:BatchNorm1d, 3:Dropout, 4:Linear, 5:ReLU, 6:Dropout, 7:Linear]
class StoneDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features  # 1792
        self.head = nn.Sequential(
            nn.Flatten(),                       # 0
            nn.Dropout(p=0.3),                  # 1
            nn.BatchNorm1d(in_features),         # 2 ← matches head.2.*
            nn.Dropout(p=0.3),                  # 3
            nn.Linear(in_features, 512),         # 4 ← matches head.4.*
            nn.ReLU(),                           # 5
            nn.Dropout(p=0.3),                  # 6
            nn.Linear(512, 2),                   # 7 ← matches head.7.*
        )
    def forward(self, x): return self.head(self.backbone(x))

# ── Model 2 — 4-Class Classifier (v2) ────────────────────────────────────────
class KidneyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features), nn.Dropout(p=0.4),
            nn.Linear(in_features, 512), nn.GELU(),
            nn.BatchNorm1d(512), nn.Dropout(p=0.3),
            nn.Linear(512, 4),
        )
    def forward(self, x): return self.classifier(self.backbone(x))

# ── Model 3 — Cancer Detector (v3) ───────────────────────────────────────────
class CancerDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features), nn.Dropout(p=0.4),
            nn.Linear(in_features, 512), nn.GELU(),
            nn.BatchNorm1d(512), nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )
    def forward(self, x): return self.classifier(self.backbone(x))

# ── Grad-CAM helper ───────────────────────────────────────────────────────────
def generate_gradcam(model, tensor, img_rgb, class_idx):
    try:
        model_cpu = type(model)()
        model_cpu.load_state_dict(model.state_dict())
        model_cpu.eval()
        target_layers = [model_cpu.backbone.blocks[-1][-1]]
        with GradCAMPlusPlus(model=model_cpu, target_layers=target_layers) as cam:
            grayscale = cam(
                input_tensor=tensor.cpu(),
                targets=[ClassifierOutputTarget(class_idx)]
            )[0]
        vis = show_cam_on_image(img_rgb.astype(np.float32) / 255.0, grayscale, use_rgb=True)
        buf = BytesIO()
        Image.fromarray(vis).save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f'Grad-CAM error: {e}')
        return None

# ── Unified Predictor ─────────────────────────────────────────────────────────
class UnifiedPredictor:
    V1_CLASSES = ['no_stone', 'stone']
    V2_CLASSES = ['normal', 'cyst', 'stone', 'tumour']
    V3_CLASSES = ['not_cancer', 'cancer']

    V2_COLORS = {
        'normal': '#00D4AA', 'cyst':   '#2D9CDB',
        'stone':  '#FF8C42', 'tumour': '#FF4B6E',
    }
    CLINICAL = {
        'normal':     'No abnormality detected. Routine follow-up recommended.',
        'cyst':       'Kidney cyst detected. Usually benign. Bosniak classification advised.',
        'stone':      'Kidney stone detected. Urology referral recommended.',
        'tumour':     'Renal mass detected. Urgent urology and oncology referral.',
        'cancer':     'Malignancy suspected. Urgent oncology referral recommended.',
        'not_cancer': 'No malignancy detected. Routine follow-up recommended.',
    }

    def __init__(self, ckpt_dir='checkpoints', img_size=224, temperature=0.5):
        self.img_size    = img_size
        self.temperature = temperature
        self.transform   = get_transform()
        self.device      = get_device()
        print(f'Loading 3 models on {self.device}...')

        def load(ModelClass, filename):
            m = ModelClass().to(self.device)
            m.load_state_dict(
                torch.load(f'{ckpt_dir}/{filename}', map_location=self.device)
            )
            m.eval()
            return m

        self.model_v1 = load(StoneDetector,    'best_model.pth')
        print('  v1 stone detector loaded ✅')
        self.model_v2 = load(KidneyClassifier, 'best_model_v2.pth')
        print('  v2 4-class classifier loaded ✅')
        self.model_v3 = load(CancerDetector,   'best_model_v3.pth')
        print('  v3 cancer detector loaded ✅')
        print('All 3 models ready ✅')

    def _preprocess(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Could not decode image')
        img = cv2.resize(img, (self.img_size, self.img_size))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        tensor = self.transform(image=img)['image'].unsqueeze(0).to(self.device)
        return img, tensor

    def _infer(self, model, tensor):
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()
        return probs

    def predict(self, image_bytes: bytes, include_gradcam: bool = True) -> dict:
        img_rgb, tensor = self._preprocess(image_bytes)

        # ── V1: Stone detection ──────────────────────────────────────────────
        p1     = self._infer(self.model_v1, tensor)
        v1_idx = int(p1.argmax())
        v1_result = {
            'prediction':    self.V1_CLASSES[v1_idx],
            'has_stone':     bool(v1_idx == 1),
            'confidence':    round(float(p1[v1_idx]), 4),
            'probabilities': {
                self.V1_CLASSES[i]: round(float(p1[i]), 4) for i in range(2)
            },
        }

        # ── V2: 4-class classification ───────────────────────────────────────
        p2      = self._infer(self.model_v2, tensor)
        v2_idx  = int(p2.argmax())
        v2_name = self.V2_CLASSES[v2_idx]
        v2_result = {
            'prediction':    v2_name,
            'confidence':    round(float(p2[v2_idx]), 4),
            'color':         self.V2_COLORS[v2_name],
            'clinical_note': self.CLINICAL[v2_name],
            'probabilities': {
                self.V2_CLASSES[i]: round(float(p2[i]), 4) for i in range(4)
            },
        }

        # ── V3: Cancer detection ─────────────────────────────────────────────
        p3      = self._infer(self.model_v3, tensor)
        v3_idx  = int(p3.argmax())
        v3_name = self.V3_CLASSES[v3_idx]
        v3_result = {
            'prediction':    v3_name,
            'is_cancer':     bool(v3_idx == 1),
            'confidence':    round(float(p3[v3_idx]), 4),
            'cancer_prob':   round(float(p3[1]), 4),
            'clinical_note': self.CLINICAL[v3_name],
            'probabilities': {
                self.V3_CLASSES[i]: round(float(p3[i]), 4) for i in range(2)
            },
        }

        # ── Grad-CAM on v2 (most informative) ───────────────────────────────
        gradcam = None
        if include_gradcam:
            gradcam = generate_gradcam(self.model_v2, tensor, img_rgb, v2_idx)

        # ── Risk level ───────────────────────────────────────────────────────
        if v3_result['is_cancer']:   risk = 'HIGH'
        elif v2_name == 'tumour':    risk = 'HIGH'
        elif v2_name == 'stone':     risk = 'MEDIUM'
        elif v2_name == 'cyst':      risk = 'LOW'
        else:                        risk = 'NORMAL'

        return {
            'v1':              v1_result,
            'v2':              v2_result,
            'v3':              v3_result,
            'risk_level':      risk,
            'gradcam_heatmap': gradcam,
            'model_versions':  ['v1_stone_detector', 'v2_4class', 'v3_cancer'],
        }