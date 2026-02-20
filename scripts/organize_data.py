import os, shutil
from pathlib import Path

KAGGLE_DIR = Path('data/external/kidney_kaggle')
OUT_DIR    = Path('data/processed')

# Map: folder name → binary label
LABEL_MAP = {
    'Stone': 'stone',      # positive class
    'Cyst':  'no_stone',   # negative
    'Normal':'no_stone',   # negative
    'Tumor': 'no_stone',   # negative (different condition)
}

def organize():
    counts = {'stone': 0, 'no_stone': 0}
    for folder_name, label in LABEL_MAP.items():
        src = KAGGLE_DIR / folder_name
        if not src.exists():
            print(f'  Warning: {src} not found, skipping')
            continue
        # We'll split into train/val/test later (Step 5)
        # For now put everything into train/ for inspection
        dst = OUT_DIR / 'train' / label
        dst.mkdir(parents=True, exist_ok=True)
        images = list(src.glob('*.jpg')) + list(src.glob('*.png'))
        for img in images:
            shutil.copy2(img, dst / img.name)
            counts[label] += 1
    print('Done!')
    print(f'  Stone images:    {counts["stone"]}')
    print(f'  No-stone images: {counts["no_stone"]}')
    ratio = counts['no_stone'] / max(counts['stone'], 1)
    print(f'  Imbalance ratio: {ratio:.1f}:1 (no_stone:stone)')

if __name__ == '__main__':
    organize()
