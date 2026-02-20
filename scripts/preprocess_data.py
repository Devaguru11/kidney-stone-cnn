# scripts/preprocess_data.py
import cv2, numpy as np, os
from pathlib import Path
from tqdm import tqdm

SRC_DIR    = Path('data/processed/train')
TARGET_SIZE = (224, 224)   # standard CNN input size

def preprocess_image(img_path: Path) -> np.ndarray:
    """Load, resize, and normalize a single image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None  # skip corrupt files

    # Resize to 224x224
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # Apply CLAHE for contrast enhancement (helps see small stones)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img

def run():
    for label in ['stone', 'no_stone']:
        folder = SRC_DIR / label
        files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        print(f'Processing {len(files)} images in {label}...')
        skipped = 0
        for f in tqdm(files):
            result = preprocess_image(f)
            if result is None:
                skipped += 1
                f.unlink()  # delete corrupt file
                continue
            # Overwrite in place (already in processed/)
            cv2.imwrite(str(f), result)
        print(f'  Done. Skipped (corrupt): {skipped}')

if __name__ == '__main__':
    run()

