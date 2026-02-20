# scripts/split_data.py
import os, shutil, hashlib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC_DIR = Path('data/processed/train')
OUT_DIR = Path('data/processed')
SPLITS_CSV = Path('data/labels/splits.csv')

# Split ratios
VAL_RATIO  = 0.15
TEST_RATIO = 0.15

def stable_hash(filename: str) -> float:
    """Deterministic hash → float [0,1] for reproducible splits."""
    h = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0

def run():
    records = []
    for label in ['stone', 'no_stone']:
        for f in (SRC_DIR / label).glob('*'):
            h = stable_hash(f.name)
            if h < TEST_RATIO:
                split = 'test'
            elif h < TEST_RATIO + VAL_RATIO:
                split = 'val'
            else:
                split = 'train'
            records.append({'filename': f.name, 'label': label, 'split': split})

    df = pd.DataFrame(records)
    print('Split summary:')
    print(df.groupby(['split','label']).size().unstack(fill_value=0))

    # Move files to correct split folders
    for _, row in df.iterrows():
        src = SRC_DIR / row['label'] / row['filename']
        dst = OUT_DIR / row['split'] / row['label'] / row['filename']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))

    df.to_csv(SPLITS_CSV, index=False)
    print(f'Splits saved to {SPLITS_CSV}')

if __name__ == '__main__':
    run()

