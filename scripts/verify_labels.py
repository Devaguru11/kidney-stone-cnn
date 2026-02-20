# scripts/verify_labels.py
import os, hashlib, cv2, json, pandas as pd, numpy as np
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image
from tqdm import tqdm

DATA_DIR   = Path('data/processed')
SPLITS_CSV = Path('data/labels/splits.csv')
REPORT_DIR = Path('data/labels/label_verification')
REPORT_DIR.mkdir(exist_ok=True)

issues = []   # collect all problems found

# ── CHECK 1: Class Balance ──────────────────────────────────────────
print('Check 1/5: Class balance...')
df = pd.read_csv(SPLITS_CSV)
balance = df.groupby(['split', 'label']).size().unstack(fill_value=0)
print(balance)
for split in ['train', 'val', 'test']:
    stone    = balance.loc[split, 'stone']
    no_stone = balance.loc[split, 'no_stone']
    ratio    = no_stone / max(stone, 1)
    if ratio > 10:
        issues.append(f'CRITICAL: {split} imbalance {ratio:.1f}:1 — fix before training')
    elif ratio > 5:
        issues.append(f'WARNING: {split} imbalance {ratio:.1f}:1 — monitor closely')

# ── CHECK 2: Duplicate Detection via pHash ──────────────────────────
print('Check 2/5: Duplicate detection (pHash)...')
hashes = defaultdict(list)
all_images = list(DATA_DIR.rglob('*.jpg')) + list(DATA_DIR.rglob('*.png'))
for img_path in tqdm(all_images):
    try:
        h = str(imagehash.phash(Image.open(img_path)))
        hashes[h].append(str(img_path))
    except Exception:
        issues.append(f'CORRUPT: {img_path}')

dup_groups = {h: paths for h, paths in hashes.items() if len(paths) > 1}
if dup_groups:
    issues.append(f'WARNING: {len(dup_groups)} duplicate image groups found')
    # Write duplicates report
    with open(REPORT_DIR / 'duplicates.json', 'w') as f:
        json.dump(dup_groups, f, indent=2)

# ── CHECK 3: Corrupt / Blank Images ────────────────────────────────
print('Check 3/5: Corrupt / blank images...')
blank_count = 0
for img_path in tqdm(all_images):
    img = cv2.imread(str(img_path))
    if img is None:
        issues.append(f'CORRUPT: {img_path}')
        continue
    mean_val = img.mean()
    if mean_val < 5:    # nearly all-black
        issues.append(f'BLANK(black): {img_path}')
        blank_count += 1
    if mean_val > 250:  # nearly all-white
        issues.append(f'BLANK(white): {img_path}')
        blank_count += 1
print(f'  Blank/corrupt images found: {blank_count}')

# ── CHECK 4: Train/Test Leakage ─────────────────────────────────────
print('Check 4/5: Data leakage check...')
train_names = set(df[df.split=='train'].filename)
test_names  = set(df[df.split=='test'].filename)
leaked = train_names & test_names
if leaked:
    issues.append(f'CRITICAL LEAKAGE: {len(leaked)} files in both train and test!')
else:
    print('  No leakage detected.')

# ── CHECK 5: Size / Dimension Check ────────────────────────────────
print('Check 5/5: Image size consistency...')
sizes = defaultdict(int)
for img_path in tqdm(list(all_images)[:500]):  # sample 500
    img = cv2.imread(str(img_path))
    if img is not None:
        sizes[img.shape[:2]] += 1
print('  Image sizes found:', dict(sizes))
if len(sizes) > 1:
    issues.append('WARNING: Mixed image sizes detected — preprocessing may be incomplete')

# ── GENERATE REPORT ─────────────────────────────────────────────────
report_path = REPORT_DIR / 'qa_report.txt'
with open(report_path, 'w') as f:
    f.write('KIDNEY STONE DATASET — LABEL VERIFICATION REPORT\n')
    f.write('='*60 + '\n\n')
    f.write(str(balance) + '\n\n')
    f.write(f'Total images: {len(all_images)}\n')
    f.write(f'Duplicate groups: {len(dup_groups)}\n')
    f.write(f'Issues found: {len(issues)}\n\n')
    for issue in issues:
        f.write(f'  [{"PASS" if not issue else "ISSUE"}] {issue}\n')
    if not issues:
        f.write('  ALL CHECKS PASSED — dataset is clean\n')

print(f'\nQA Report saved to {report_path}')
print(f'Issues found: {len(issues)}')
for i in issues: print(f'  → {i}')


