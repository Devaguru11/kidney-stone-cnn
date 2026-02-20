# scripts/generate_annotations.py
import json, os
from pathlib import Path
import pandas as pd

SPLITS_CSV = Path('data/labels/splits.csv')
OUT_JSON   = Path('data/labels/annotations.json')

df = pd.read_csv(SPLITS_CSV)

coco = {
    'info': {'description': 'Kidney Stone Detection Dataset', 'version': '1.0'},
    'categories': [
        {'id': 0, 'name': 'no_stone'},
        {'id': 1, 'name': 'stone'}
    ],
    'images': [],
    'annotations': []
}

label_to_id = {'no_stone': 0, 'stone': 1}
for img_id, row in enumerate(df.itertuples()):
    coco['images'].append({
        'id': img_id,
        'file_name': f"{row.split}/{row.label}/{row.filename}",
        'split': row.split,
        'label': row.label,
        'category_id': label_to_id[row.label]
    })
    # Bounding box annotation: to be filled during labeling (Step 7)
    # For classification, annotation is just the image-level label above

with open(OUT_JSON, 'w') as f:
    json.dump(coco, f, indent=2)
print(f'Saved {len(coco["images"])} image records to {OUT_JSON}')
