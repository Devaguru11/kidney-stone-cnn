# scripts/test_api.py
# Run this while the API is running in another terminal
import requests, os, time, json, base64
from pathlib import Path

BASE_URL  = 'http://localhost:8000'
DATA_DIR  = Path('data/processed/test')
PASS = '✅'; FAIL = '❌'

def test_health():
    print('\n--- Test: /health ---')
    r = requests.get(f'{BASE_URL}/health')
    assert r.status_code == 200, f'Expected 200, got {r.status_code}'
    data = r.json()
    assert data['status'] == 'healthy', 'API not healthy'
    assert data['model_loaded'] == True, 'Model not loaded'
    print(f'{PASS} /health — status: {data["status"]}, device: {data["device"]}')

def test_model_info():
    print('\n--- Test: /model-info ---')
    r = requests.get(f'{BASE_URL}/model-info')
    assert r.status_code == 200
    data = r.json()
    print(f'{PASS} /model-info — arch: {data["architecture"]}')
    print(f'     params: {data["parameters"]:,} | AUC: {data["training_auc"]}')

def test_predict_stone():
    print('\n--- Test: /predict — stone image ---')
    stone_images = list((DATA_DIR/'stone').glob('*.jpg'))[:3]
    for img_path in stone_images:
        with open(img_path, 'rb') as f:
            t0 = time.time()
            r  = requests.post(f'{BASE_URL}/predict?include_gradcam=false',
                               files={'file': (img_path.name, f, 'image/jpeg')})
            ms = round((time.time()-t0)*1000)
        assert r.status_code == 200, f'Got {r.status_code}: {r.text}'
        data = r.json()
        correct = data['prediction'] == 'stone'
        icon = PASS if correct else FAIL
        print(f'{icon} {img_path.name[:30]:30s} | pred={data["prediction"]:10s} | conf={data["confidence"]:.4f} | {ms}ms')

def test_predict_no_stone():
    print('\n--- Test: /predict — no_stone image ---')
    imgs = list((DATA_DIR/'no_stone').glob('*.jpg'))[:3]
    for img_path in imgs:
        with open(img_path, 'rb') as f:
            r = requests.post(f'{BASE_URL}/predict?include_gradcam=false',
                              files={'file': (img_path.name, f, 'image/jpeg')})
        assert r.status_code == 200
        data = r.json()
        correct = data['prediction'] == 'no_stone'
        icon = PASS if correct else FAIL
        print(f'{icon} {img_path.name[:30]:30s} | pred={data["prediction"]:10s} | conf={data["confidence"]:.4f}')

def test_predict_with_gradcam():
    print('\n--- Test: /predict — with Grad-CAM heatmap ---')
    img_path = list((DATA_DIR/'stone').glob('*.jpg'))[0]
    with open(img_path, 'rb') as f:
        r = requests.post(f'{BASE_URL}/predict?include_gradcam=true',
                          files={'file': (img_path.name, f, 'image/jpeg')})
    assert r.status_code == 200
    data = r.json()
    has_heatmap = data['gradcam_heatmap'] is not None
    icon = PASS if has_heatmap else FAIL
    print(f'{icon} Grad-CAM heatmap present: {has_heatmap}')
    if has_heatmap:
        # Save heatmap to file for inspection
        img_bytes = base64.b64decode(data['gradcam_heatmap'])
        with open('reports/api_test_heatmap.png', 'wb') as f:
            f.write(img_bytes)
        print(f'{PASS} Heatmap saved → reports/api_test_heatmap.png')

def test_batch_predict():
    print('\n--- Test: /predict/batch ---')
    imgs = list((DATA_DIR/'stone').glob('*.jpg'))[:4]
    files = [('files', (p.name, open(p,'rb'), 'image/jpeg')) for p in imgs]
    r = requests.post(f'{BASE_URL}/predict/batch?include_gradcam=false', files=files)
    assert r.status_code == 200
    data = r.json()
    print(f'{PASS} Batch: {data["total"]} predictions returned')
    for pred in data['predictions']:
        print(f'     {pred["filename"][:30]:30s} → {pred["prediction"]} ({pred["confidence"]:.4f})')

def test_invalid_file():
    print('\n--- Test: invalid file type ---')
    r = requests.post(f'{BASE_URL}/predict',
                      files={'file': ('test.txt', b'not an image', 'text/plain')})
    assert r.status_code == 400, f'Expected 400, got {r.status_code}'
    print(f'{PASS} Invalid file correctly rejected with 400')

if __name__ == '__main__':
    os.chdir('/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn')
    print('=== Kidney Stone API — Full Test Suite ===')
    test_health()
    test_model_info()
    test_predict_stone()
    test_predict_no_stone()
    test_predict_with_gradcam()
    test_batch_predict()
    test_invalid_file()
    print('\n=== All tests complete ===')
