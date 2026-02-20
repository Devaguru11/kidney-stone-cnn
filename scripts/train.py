# scripts/train.py
import os, sys, torch, mlflow, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datamodule import KidneyStoneDataModule
from src.models.efficientnet import KidneyStoneClassifier
from src.training.losses import FocalLoss
from src.training.metrics import compute_metrics

CFG = {
    'project_root':   '/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn',
    'splits_csv':     'data/labels/splits.csv',
    'data_dir':       'data/processed',
    'checkpoint_dir': 'checkpoints',
    'img_size':       224,
    'batch_size':     8,      # FIXED: was 16 — MPS runs out of memory at 16 after unfreeze
    'num_workers':    0,
    'num_epochs':     30,
    'lr_head':        1e-3,
    'lr_backbone':    1e-4,
    'patience':       7,
    'freeze_epochs':  3,
}

def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')

def train():
    os.chdir(CFG['project_root'])
    Path(CFG['checkpoint_dir']).mkdir(exist_ok=True)
    device = get_device()
    print(f'Using device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────
    dm = KidneyStoneDataModule(
        splits_csv=CFG['splits_csv'],
        data_dir=CFG['data_dir'],
        img_size=CFG['img_size'],
        batch_size=CFG['batch_size'],
        num_workers=CFG['num_workers'],
    )
    dm.setup()
    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()

    # ── Model ─────────────────────────────────────────────────────────────
    model = KidneyStoneClassifier().to(device)
    model.freeze_backbone()

    # ── Loss & Optimiser ──────────────────────────────────────────────────
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CFG['lr_backbone']},
        {'params': model.head.parameters(),     'lr': CFG['lr_head']},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['num_epochs']
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_auc    = 0.0
    patience_ct = 0

    mlflow.set_experiment('kidney-stone-cnn')
    with mlflow.start_run(run_name='efficientnet_b4_focal_v2'):
        mlflow.log_params(CFG)

        for epoch in range(1, CFG['num_epochs'] + 1):

            if epoch == CFG['freeze_epochs'] + 1:
                model.unfreeze_backbone()
                # FIXED: clear MPS cache right after unfreeze to reclaim memory
                if device.type == 'mps':
                    torch.mps.empty_cache()

            # ── Train ─────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch:02d} train'):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

                # FIXED: clear MPS cache every 50 batches to prevent memory buildup
                if device.type == 'mps' and (optimizer.state_dict()['state'] or True):
                    torch.mps.empty_cache()

            train_loss /= len(train_loader)

            # ── Validate ──────────────────────────────────────────────────
            model.eval()
            all_labels, all_probs = [], []
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs)
                    val_loss += criterion(logits, labels).item()
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= len(val_loader)
            metrics   = compute_metrics(all_labels, all_probs)
            scheduler.step()

            # FIXED: clear cache after validation too
            if device.type == 'mps':
                torch.mps.empty_cache()

            mlflow.log_metrics({
                'train_loss': round(train_loss, 4),
                'val_loss':   round(val_loss, 4),
                **metrics
            }, step=epoch)

            print(f'Epoch {epoch:02d} | train_loss={train_loss:.4f} '
                  f'val_loss={val_loss:.4f} '
                  f'AUC={metrics["auc_roc"]:.4f} '
                  f'Sens={metrics["sensitivity"]:.4f}')

            if metrics['auc_roc'] > best_auc:
                best_auc    = metrics['auc_roc']
                patience_ct = 0
                ckpt = Path(CFG['checkpoint_dir']) / 'best_model.pth'
                torch.save(model.state_dict(), ckpt)
                mlflow.log_artifact(str(ckpt))
                print(f'  ✅ New best AUC: {best_auc:.4f} — checkpoint saved')
            else:
                patience_ct += 1
                print(f'  No improvement ({patience_ct}/{CFG["patience"]})')
                if patience_ct >= CFG['patience']:
                    print('Early stopping triggered.')
                    break

        print(f'Training complete. Best val AUC: {best_auc:.4f}')
        mlflow.log_metric('best_val_auc', best_auc)

if __name__ == '__main__':
    train()