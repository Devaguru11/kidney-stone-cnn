# src/data/datamodule.py
# src/data/datamodule.py
import torch, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataset import KidneyStoneDataset
from src.data.augmentations import get_train_transforms, get_val_transforms

class KidneyStoneDataModule:
    def __init__(self,
                 splits_csv:  str = 'data/labels/splits.csv',
                 data_dir:    str = 'data/processed',
                 img_size:    int = 224,
                 batch_size:  int = 32,
                 num_workers: int = 4):
        self.splits_csv  = splits_csv
        self.data_dir    = data_dir
        self.img_size    = img_size
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self):
        self.train_ds = KidneyStoneDataset(
            self.splits_csv, self.data_dir, 'train',
            get_train_transforms(self.img_size))
        self.val_ds = KidneyStoneDataset(
            self.splits_csv, self.data_dir, 'val',
            get_val_transforms(self.img_size))
        self.test_ds = KidneyStoneDataset(
            self.splits_csv, self.data_dir, 'test',
            get_val_transforms(self.img_size))

    def _make_weighted_sampler(self, dataset):
        """Oversamples minority class so batches are ~50/50."""
        labels = [dataset.label_map[row['label']]
                  for _, row in dataset.df.iterrows()]
        labels = np.array(labels)
        class_counts = np.bincount(labels)          # [no_stone_count, stone_count]
        class_weights = 1.0 / class_counts          # higher weight for rarer class
        sample_weights = class_weights[labels]      # weight per image
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )

    def train_loader(self) -> DataLoader:
        sampler = self._make_weighted_sampler(self.train_ds)
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          sampler=sampler, num_workers=self.num_workers,
                          pin_memory=True)

    def val_loader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_loader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

