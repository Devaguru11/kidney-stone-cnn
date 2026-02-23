# 🚀 Kidney Stone Detection — Phase 4: FastAPI Inference Server

> **Status:** ✅ Complete  
> **Duration:** ~1 Day  
> **Last Updated:** February 2026  
> **Author:** Devaguru

---

## 📋 Phase Overview

Phase 4 wraps the trained EfficientNet-B4 model in a production-ready REST API using FastAPI. The API accepts CT scan images and returns predictions, confidence scores, and optional Grad-CAM heatmaps in real time.

> The model is only useful if it can be queried. Phase 4 turns the checkpoint into a living service that any frontend, hospital system, or downstream pipeline can call over HTTP.

---

## 📁 Final Folder Structure (After Phase 4)

```
kidney-stone-cnn/
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app — 5 endpoints
│   ├── inference.py          # KidneyStonePredictor class
│   └── schemas.py            # Pydantic request/response schemas
│
├── scripts/
│   ├── export_onnx.py        # Export model to ONNX format
│   └── test_api.py           # Automated end-to-end test suite
│
├── checkpoints/
│   ├── best_model.pth        # PyTorch checkpoint (Phase 2 output)
│   └── best_model.onnx       # ONNX export (Phase 4 output)
│
├── reports/
│   └── api_test_heatmap.png  # Grad-CAM output saved during test run
│
├── requirements_api.txt      # Minimal deps for serving only
└── requirements.txt          # Full deps (training + serving)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/batch` | Multiple images in one request |
| `GET` | `/health` | Server + model health check |
| `GET` | `/model-info` | Architecture, params, metrics |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## 📤 Request / Response

### `POST /predict`

**Request:** `multipart/form-data`
- `file` — JPEG/PNG image
- `include_gradcam` — boolean query param (default: `false`)

**Response:**
```json
{
  "prediction": "stone",
  "confidence": 0.9988,
  "probabilities": {
    "stone": 0.9988,
    "no_stone": 0.0012
  },
  "gradcam_heatmap": "<base64-encoded PNG or null>",
  "inference_time_ms": 30.4
}
```

### `POST /predict/batch`

**Request:** Multiple files in `multipart/form-data`

**Response:**
```json
[
  { "filename": "Stone-(817).jpg", "prediction": "stone", "confidence": 0.9988 },
  { "filename": "Normal-(529).jpg", "prediction": "no_stone", "confidence": 0.9757 }
]
```

### `GET /health`
```json
{ "status": "healthy", "device": "mps" }
```

### `GET /model-info`
```json
{
  "architecture": "EfficientNet-B4 + custom classification head",
  "parameters": 18471242,
  "auc": 1.0,
  "sensitivity": 1.0,
  "specificity": 0.9917,
  "f2_score": 0.9877
}
```

---

## ⚙️ Implementation Details

### `KidneyStonePredictor` (`api/inference.py`)
- Model loaded **once at startup** via FastAPI lifespan — not per request
- Runs on **Apple MPS** (M-series Mac) automatically, falls back to CPU
- Applies identical preprocessing to training: resize 224×224, CLAHE, ImageNet normalisation
- Grad-CAM++ heatmap generation using the final EfficientNet conv layer
- Thread-safe — single model instance shared across requests

### Pydantic Schemas (`api/schemas.py`)
- Full type validation on all inputs and outputs
- Invalid file types (non-image uploads) rejected with HTTP 400 before reaching the model

---

## ⚡ ONNX Export

The model was also exported to ONNX for CPU-optimised inference:

```bash
python scripts/export_onnx.py
```

| Runtime | Latency per image |
|---------|-------------------|
| PyTorch (MPS) | 482.6ms |
| ONNX (CPU) | 23.6ms |
| **Speedup** | **20.4×** |

Saved to: `checkpoints/best_model.onnx`

---

## 🧪 Test Results

All 7 tests passed via `scripts/test_api.py`:

| Test | Result |
|------|--------|
| `/health` | ✅ status: healthy, device: mps |
| `/model-info` | ✅ arch + params correct |
| Stone images (3) | ✅ All predicted `stone`, conf 96–100% |
| No-stone images (3) | ✅ All predicted `no_stone`, conf 97–99% |
| Grad-CAM heatmap | ✅ Generated and saved |
| Batch prediction (4) | ✅ All correct |
| Invalid file type | ✅ Rejected with HTTP 400 |

---

## 🚀 How to Start the Server

```bash
cd '/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn'

/Users/devaguru/Kidney\ Stone\ CNN/.venv/bin/uvicorn api.main:app \
  --reload --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

---

## 🧪 How to Run the Test Suite

With the server running in Terminal 1, open Terminal 2:

```bash
cd '/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn'

/Users/devaguru/Kidney\ Stone\ CNN/.venv/bin/python scripts/test_api.py
```

---

## 🔁 Quick curl Test

```bash
cp "/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn/data/processed/test/stone/Stone- (1004).jpg" /tmp/test_stone.jpg

curl -X POST "http://localhost:8000/predict?include_gradcam=false" \
  -F "file=@/tmp/test_stone.jpg"
```

---

## 📦 API Dependencies (`requirements_api.txt`)

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.16
Pillow>=10.0.0
numpy>=1.26.0
opencv-python-headless>=4.9.0
pydantic>=2.0.0
onnxscript
onnx
onnxruntime
```

---

## ⚠️ Known Limitations

1. **No authentication** — The API has no API key or token validation. Do not expose port 8000 publicly without adding auth middleware first.
2. **MPS only on Apple Silicon** — The server defaults to MPS on M-series Macs. On Linux/cloud the device falls back to CPU (or CUDA if available).
3. **Grad-CAM adds latency** — Heatmap generation adds ~200–400ms per request. Keep `include_gradcam=false` for high-throughput use.
4. **No request queuing** — Under concurrent load, multiple large batch requests may compete for GPU memory. A task queue (Celery, ARQ) is recommended for production.
5. **ONNX model not yet wired into API** — `best_model.onnx` was exported and benchmarked but the API still uses the PyTorch checkpoint. Switching the predictor to ONNX Runtime would give ~20× CPU speedup.

---

## ➡️ Next Phase

**Phase 5 — Docker + Deployment:**
- Write `Dockerfile` for the FastAPI app
- `docker build` + `docker run` locally
- Deploy to Railway / Render / EC2

---

*Kidney Stone Detection CNN — Internal Research Project*