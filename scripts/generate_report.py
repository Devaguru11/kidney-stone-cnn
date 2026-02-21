# scripts/generate_report.py
import os, sys, json, base64
from pathlib import Path
from datetime import datetime

os.chdir('/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn')
sys.path.insert(0, '.')

def img_to_base64(path):
    if not Path(path).exists(): return ''
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

METRICS = {
    'sensitivity':  1.0,
    'specificity':  0.9917,
    'auc_roc':      1.0,
    'f2_score':     0.9877,
    'precision':    0.9412,
    'tp': 224, 'fp': 14, 'tn': 1666, 'fn': 0
}

roc_b64  = img_to_base64('data/labels/label_verification/test_results.png')
gcam_b64 = img_to_base64('reports/gradcam_stone.png')
fp_b64   = img_to_base64('reports/false_positives.png')
cal_b64  = img_to_base64('reports/calibration_curve.png')
thr_b64  = img_to_base64('reports/threshold_curve.png')

# Pre-build conditional image HTML to avoid backslashes inside f-string expressions
roc_html  = f"<img src='data:image/png;base64,{roc_b64}' />"  if roc_b64  else "<p><em>Chart not found — run the evaluation notebook first.</em></p>"
gcam_html = f"<img src='data:image/png;base64,{gcam_b64}' />" if gcam_b64 else "<p><em>Run notebooks/03_gradcam.ipynb first.</em></p>"
fp_html   = f"<img src='data:image/png;base64,{fp_b64}' />"  if fp_b64   else "<p><em>Run error analysis notebook first.</em></p>"
thr_html  = f"<img src='data:image/png;base64,{thr_b64}' />" if thr_b64  else "<p><em>Run calibration notebook first.</em></p>"
cal_html  = f"<img src='data:image/png;base64,{cal_b64}' />" if cal_b64  else ""

html = f'''<!DOCTYPE html>
<html><head><meta charset='UTF-8'>
<title>Kidney Stone CNN — Clinical Evaluation Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 40px; background: #f8f9fa; color: #1a1a2e; }}
  h1   {{ color: #0d1b2a; border-bottom: 3px solid #1E88E5; padding-bottom: 12px; }}
  h2   {{ color: #1565C0; margin-top: 40px; }}
  h3   {{ color: #1E88E5; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .kpi {{ background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 8px #0001; }}
  .kpi .value {{ font-size: 2.2em; font-weight: bold; color: #1E88E5; }}
  .kpi .label {{ font-size: 0.85em; color: #546E7A; margin-top: 6px; }}
  .kpi.green .value {{ color: #2E7D32; }}
  .kpi.red .value {{ color: #B71C1C; }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px #0001; }}
  th {{ background: #0d1b2a; color: white; padding: 12px 16px; text-align: left; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #E8ECF0; }}
  tr:nth-child(even) td {{ background: #F4F6F9; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 16px 0; box-shadow: 0 2px 12px #0002; }}
  .pass {{ color: #2E7D32; font-weight: bold; }}
  .warn {{ color: #E65100; font-weight: bold; }}
  .section {{ background: white; border-radius: 12px; padding: 28px; margin: 24px 0; box-shadow: 0 2px 8px #0001; }}
  .note {{ background: #F4F6F9; border-left: 4px solid #1E88E5; padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 0.95em; color: #546E7A; }}
</style></head><body>

<h1>🫘 Kidney Stone Detection CNN — Clinical Evaluation Report</h1>
<p><strong>Model:</strong> EfficientNet-B4 with custom classification head<br>
<strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
<strong>Dataset:</strong> 12,446 images (CT + Ultrasound) | Test set: 1,904 images<br>
<strong>Status:</strong> <span class='pass'>✅ All clinical KPI targets exceeded</span></p>

<div class='section'>
<h2>Key Performance Indicators</h2>
<div class='kpi-grid'>
  <div class='kpi green'><div class='value'>{METRICS['sensitivity']:.1%}</div><div class='label'>Sensitivity<br>(Target ≥ 92%)</div></div>
  <div class='kpi green'><div class='value'>{METRICS['specificity']:.1%}</div><div class='label'>Specificity<br>(Target ≥ 88%)</div></div>
  <div class='kpi green'><div class='value'>{METRICS['auc_roc']:.4f}</div><div class='label'>AUC-ROC<br>(Target ≥ 0.95)</div></div>
  <div class='kpi red'><div class='value'>{METRICS['fn']}</div><div class='label'>False Negatives<br>(Missed Stones)</div></div>
</div></div>

<div class='section'>
<h2>Confusion Matrix & ROC Curve</h2>
<p>The ROC curve shows perfect discrimination (AUC = 1.0). The confusion matrix confirms zero missed stone cases across all 224 stone images in the test set.</p>
{roc_html}
</div>

<div class='section'>
<h2>Grad-CAM Visual Explanations</h2>
<p>Grad-CAM++ heatmaps show which image regions drove each prediction. Red/yellow areas = high model attention. These should correspond to kidney and urinary tract anatomy for valid predictions.</p>
{gcam_html}
</div>

<div class='section'>
<h2>False Positive Analysis ({METRICS['fp']} cases)</h2>
<p>All {METRICS['fp']} false positives are shown below with Grad-CAM overlays. These are no_stone images incorrectly predicted as stone. Common causes: cysts mimicking stones, vascular calcifications, image compression artifacts.</p>
{fp_html}
<div class='note'>Clinical impact: These 14 cases would trigger unnecessary follow-up imaging but would not cause harm. Zero false negatives means zero missed stones — which is the primary safety requirement.</div>
</div>

<div class='section'>
<h2>Threshold Calibration</h2>
<p>Decision threshold was optimised using F2-score on the validation set. F2 weights recall (sensitivity) twice as heavily as precision, appropriate for a screening tool where missing a stone is more harmful than a false alarm.</p>
{thr_html}
{cal_html}
</div>

<div class='section'>
<h2>Detailed Metrics Table</h2>
<table><tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>
<tr><td>Sensitivity (Recall)</td><td>{METRICS['sensitivity']:.4f}</td><td>≥ 0.92</td><td class='pass'>✅ PASSED</td></tr>
<tr><td>Specificity</td><td>{METRICS['specificity']:.4f}</td><td>≥ 0.88</td><td class='pass'>✅ PASSED</td></tr>
<tr><td>AUC-ROC</td><td>{METRICS['auc_roc']:.4f}</td><td>≥ 0.95</td><td class='pass'>✅ PASSED</td></tr>
<tr><td>Precision</td><td>{METRICS['precision']:.4f}</td><td>≥ 0.85</td><td class='pass'>✅ PASSED</td></tr>
<tr><td>F2-Score</td><td>{METRICS['f2_score']:.4f}</td><td>≥ 0.90</td><td class='pass'>✅ PASSED</td></tr>
<tr><td>False Negatives</td><td>{METRICS['fn']}</td><td>Minimise</td><td class='pass'>✅ ZERO</td></tr>
<tr><td>False Positives</td><td>{METRICS['fp']}</td><td>&lt; 5% of negatives</td><td class='pass'>✅ 0.83%</td></tr>
</table></div>

<div class='section'>
<h2>Known Limitations</h2>
<ul>
<li>Dataset is Kaggle CT + Ultrasound only — no TCIA DICOM data. Performance on different scanner manufacturers is unknown.</li>
<li>Patient-level split was not possible due to missing patient IDs in source dataset. Sequential CT slices may appear in both train and test.</li>
<li>Only 952 stone training images — model may underperform on rare stone variants (&lt;3mm, faint calcifications).</li>
<li>AUC = 1.0 on the test set may indicate some data leakage via sequential CT slices. Results should be validated on an independent external dataset before clinical deployment.</li>
</ul></div>

<div class='section'>
<h2>Next Steps — Phase 4</h2>
<p>Model is ready for API development and deployment. Phase 4 will wrap this model in a FastAPI REST endpoint, containerise with Docker, and serve predictions with Grad-CAM heatmaps via HTTP.</p>
</div>

<hr><p style='color:#999;font-size:0.85em'>Generated automatically by scripts/generate_report.py · {datetime.now().strftime('%Y-%m-%d %H:%M')} · Kidney Stone CNN v1.0</p>
</body></html>'''

out = Path('reports/clinical_report.html')
out.write_text(html)
print(f'Report saved to {out}')
print('Open in browser: open reports/clinical_report.html')
