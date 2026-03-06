# api/main_v2.py
import os, sys, time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.inference_v2 import KidneyPredictorV2

# ── Global predictor ──────────────────────────────────────────────────────────
predictor: KidneyPredictorV2 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    os.chdir(str(Path(__file__).parent.parent))
    predictor = KidneyPredictorV2(checkpoint='checkpoints/best_model_v2.pth')
    yield
    print('Shutting down v2 API...')


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title='NephroScan AI v2',
    description='4-class kidney condition classifier: Normal / Cyst / Stone / Tumour',
    version='2.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Response schemas ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:    str
    confidence:    float
    probabilities: dict
    clinical_note: str
    color:         str
    model_version: str


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    version:     str
    device:      str
    classes:     list


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get('/health', response_model=HealthResponse, tags=['System'])
async def health():
    device = str(predictor.device) if predictor else 'not loaded'
    return HealthResponse(
        status='healthy' if predictor else 'model not loaded',
        model_loaded=predictor is not None,
        version='v2',
        device=device,
        classes=['normal', 'cyst', 'stone', 'tumour'],
    )


@app.get('/model-info', tags=['System'])
async def model_info():
    return {
        'architecture':  'EfficientNet-B4 + 4-class head',
        'parameters':    18_471_244,
        'classes':       ['normal', 'cyst', 'stone', 'tumour'],
        'input_size':    '224x224 RGB',
        'test_accuracy': 0.9698,
        'test_auc':      0.9984,
        'per_class': {
            'normal': {'recall': 0.9960, 'precision': 0.9614},
            'cyst':   {'recall': 0.9769, 'precision': 0.9801},
            'stone':  {'recall': 0.9135, 'precision': 0.9314},
            'tumour': {'recall': 0.9388, 'precision': 0.9919},
        }
    }


@app.post('/predict', response_model=PredictionResponse, tags=['Inference'])
async def predict(
    file: UploadFile = File(..., description='CT scan image (JPG or PNG)'),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type: {file.content_type}. Use JPG or PNG.'
        )

    try:
        start  = time.time()
        result = predictor.predict(await file.read())
        elapsed = round(time.time() - start, 3)
        print(f'Predicted: {result["prediction"]} ({result["confidence"]:.2%}) in {elapsed}s')
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference error: {str(e)}')


# ── Run ───────────────────────────────────────────────────────────────────────
# uvicorn api.main_v2:app --host 0.0.0.0 --port 8001 --reload