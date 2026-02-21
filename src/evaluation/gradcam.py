# src/evaluation/gradcam.py
import cv2, torch, numpy as np
from pathlib import Path
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.models.efficientnet import KidneyStoneClassifier
from src.data.augmentations import get_val_transforms

def load_model(checkpoint_path: str, device) -> KidneyStoneClassifier:
    model = KidneyStoneClassifier().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def generate_gradcam(model, img_path: str, device,
                     target_class: int = 1) -> dict:
    """
    Generate Grad-CAM++ heatmap for a single image.
    target_class: 1=stone, 0=no_stone
    Returns dict with: heatmap, overlay, prediction, confidence
    """
    # Load original image for overlay
    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, (224, 224))
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig_float = orig_rgb.astype(np.float32) / 255.0

    # Preprocess for model
    transform = get_val_transforms(224)
    tensor = transform(image=orig_rgb)['image'].unsqueeze(0).to(device)

    # Target layer — last block of EfficientNet backbone
    target_layer = [model.backbone.blocks[-1]]

    # Generate Grad-CAM++
    cam = GradCAMPlusPlus(model=model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    # Create coloured overlay
    overlay = show_cam_on_image(orig_float, grayscale_cam, use_rgb=True)

    # Get model prediction
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    label_map = {0: 'no_stone', 1: 'stone'}
    return {
        'heatmap':    grayscale_cam,
        'overlay':    overlay,
        'prediction': label_map[pred_class],
        'confidence': round(confidence, 4),
        'true_class': target_class,
    }

