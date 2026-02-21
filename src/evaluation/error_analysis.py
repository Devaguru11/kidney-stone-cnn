# src/evaluation/error_analysis.py
import os, sys, torch, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datamodule import KidneyStoneDataModule
from src.models.efficientnet import KidneyStoneClassifier
from src.evaluation.gradcam import generate_gradcam

def find_errors(model, dm, device, threshold=0.5):
    """Find all false positives and false negatives in the test set."""
    model.eval()
    df = pd.read_csv('data/labels/splits.csv')
    test_df = df[df.split == 'test'].reset_index(drop=True)

    all_labels, all_probs, all_paths = [], [], []

    with torch.no_grad():
        for imgs, labels in dm.test_loader():
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = (np.array(all_probs) >= threshold).astype(int)
    all_labels = np.array(all_labels)

    # Build results dataframe
    results = test_df.copy()
    results['prob_stone'] = all_probs
    results['pred']  = all_preds
    results['label_int'] = all_labels
    results['correct'] = (results.pred == results.label_int)

    false_positives = results[
        (results.label_int == 0) & (results.pred == 1)  # no_stone predicted as stone
    ]
    false_negatives = results[
        (results.label_int == 1) & (results.pred == 0)  # stone predicted as no_stone
    ]

    print(f'False Positives: {len(false_positives)}')
    print(f'False Negatives: {len(false_negatives)}')
    return false_positives, false_negatives, results

def plot_false_positives(false_positives, model, device, save_path):
    """Show all FP images with Grad-CAM and confidence scores."""
    n = len(false_positives)
    if n == 0:
        print('No false positives to show!')
        return

    cols = min(n, 7)
    rows = ((n - 1) // cols + 1) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f'False Positives — {n} no_stone images predicted as stone',
                 fontsize=13, fontweight='bold', color='red')

    for idx, (_, row) in enumerate(false_positives.iterrows()):
        img_path = f"data/processed/test/no_stone/{row['filename']}"
        if not Path(img_path).exists(): continue

        result = generate_gradcam(model, img_path, device, target_class=1)
        orig   = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        r_idx = (idx // cols) * 2
        c_idx = idx % cols

        if rows == 2:
            axes[0][c_idx].imshow(orig)
            axes[0][c_idx].set_title(f"{row['filename'][:15]}", fontsize=7)
            axes[0][c_idx].axis('off')
            axes[1][c_idx].imshow(result['overlay'])
            axes[1][c_idx].set_title(f"conf: {result['confidence']:.2%}", fontsize=7, color='red')
            axes[1][c_idx].axis('off')
        else:
            axes[r_idx][c_idx].imshow(orig)
            axes[r_idx][c_idx].set_title(f"{row['filename'][:15]}", fontsize=7)
            axes[r_idx][c_idx].axis('off')
            axes[r_idx+1][c_idx].imshow(result['overlay'])
            axes[r_idx+1][c_idx].set_title(f"conf: {result['confidence']:.2%}", fontsize=7, color='red')
            axes[r_idx+1][c_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Saved to {save_path}')
