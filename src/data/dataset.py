# src/data/dataset.py
import cv2, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class KidneyStoneDataset(Dataset):
    """
    Loads kidney stone images from data/processed/.
    Label: 1 = stone present, 0 = no stone
    """
    def __init__(self,
                 splits_csv: str,
                 data_dir: str,
                 split: str,          # 'train', 'val', or 'test'
                 transform=None):
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.label_map = {'stone': 1, 'no_stone': 0}

        df = pd.read_csv(splits_csv)
        self.df = df[df['split'] == split].reset_index(drop=True)
        print(f'[{split}] Loaded {len(self.df)} images')
        print(f'  stone={int((self.df.label=="stone").sum())}  '
              f'no_stone={int((self.df.label=="no_stone").sum())}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = self.label_map[row['label']]

        # Build path:  data/processed/<split>/<label>/<filename>
        img_path = (self.data_dir / row['split'] /
                    row['label'] / row['filename'])

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f'Image not found: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label
