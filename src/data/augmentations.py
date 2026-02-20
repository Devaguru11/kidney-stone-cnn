# src/data/augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet mean and std — used because EfficientNet was pre-trained on ImageNet
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Augmentations applied ONLY during training."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16,
                        max_width=16, p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms(img_size: int = 224) -> A.Compose:
    """No augmentation for val/test — only resize and normalize."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
