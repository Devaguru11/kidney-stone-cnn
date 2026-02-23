# api/inference.py
import torch, cv2, numpy as np, base64, io
from pathlib import Path
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.efficientnet import KidneyStoneClassifier
from src.data.augmentations import get_val_transforms

class KidneyStonePredictor:
    """
    Loads model once at startup.
    Thread-safe for concurrent API requests.
    """
    def __init__(self,
                 checkpoint_path: str = 'checkpoints/best_model.pth',
                 threshold: float = 0.5,
                 img_size: int = 224):
        self.threshold = threshold
        self.img_size  = img_size
        self.transform = get_val_transforms(img_size)
        self.label_map = {0: 'no_stone', 1: 'stone'}

        # Device — MPS on Mac, CUDA on NVIDIA, CPU fallback
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f'Loading model on {self.device}...')
        self.model = KidneyStoneClassifier().to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print('Model ready.')

    def preprocess(self, image_bytes: bytes) -> tuple:
        """Convert raw image bytes → model-ready tensor + original float array."""
        # Decode bytes → numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Could not decode image — unsupported format')

        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_float = img_resized.astype(np.float32) / 255.0

        tensor = self.transform(image=img_resized)['image'].unsqueeze(0)
        return tensor.to(self.device), img_float

    def predict(self, image_bytes: bytes,
                include_gradcam: bool = True) -> dict:
        """
        Run full inference pipeline on image bytes.
        Returns prediction dict matching PredictionResponse schema.
        """
        tensor, img_float = self.preprocess(image_bytes)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            prob_stone    = probs[1].item()
            prob_no_stone = probs[0].item()

        pred_class = 1 if prob_stone >= self.threshold else 0
        prediction = self.label_map[pred_class]
        confidence = prob_stone if pred_class == 1 else prob_no_stone

        gradcam_b64 = None
        if include_gradcam:
            gradcam_b64 = self._generate_gradcam(tensor, img_float, pred_class)

        return {
            'prediction':           prediction,
            'confidence':           round(confidence, 4),
            'probability_stone':    round(prob_stone, 4),
            'probability_no_stone': round(prob_no_stone, 4),
            'gradcam_heatmap':      gradcam_b64,
            'model_version':        'efficientnet_b4_v1',
            'threshold_used':       self.threshold,
        }

    def _generate_gradcam(self, tensor, img_float, target_class: int) -> str:
        """Generate Grad-CAM++ heatmap, return as base64 PNG string."""
        try:
            target_layer = [self.model.backbone.blocks[-1]]
            cam = GradCAMPlusPlus(model=self.model, target_layers=target_layer)
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
            overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

            # Convert numpy array → base64 PNG
            pil_img = Image.fromarray(overlay)
            buffer  = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f'Grad-CAM failed: {e}')
            return None

