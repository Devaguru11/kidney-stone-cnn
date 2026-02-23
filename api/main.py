# api/main.py
import os, sys, time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.inference import KidneyStonePredictor
from api.schemas import PredictionResponse, HealthResponse, ModelInfoResponse

# ── Global predictor instance (loaded once at startup) ────────────
predictor: KidneyStonePredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when API starts, clean up when it stops."""
    global predictor
    os.chdir(str(Path(__file__).parent.parent))  # set project root
    print('Starting Kidney Stone CNN API...')
    predictor = KidneyStonePredictor(
        checkpoint_path='checkpoints/best_model.pth',
        threshold=0.5
    )
    yield  # API is running here
    print('Shutting down...')

# ── Create FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title='Kidney Stone Detection API',
    description='EfficientNet-B4 model for detecting kidney stones in CT/ultrasound images. Returns prediction, confidence, and Grad-CAM heatmap.',
    version='1.0.0',
    lifespan=lifespan,
)

# Allow requests from any origin (needed for browser-based testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Endpoints ─────────────────────────────────────────────────────

@app.get('/health', response_model=HealthResponse, tags=['System'])
async def health_check():
    """Check if the API and model are running correctly."""
    device = str(predictor.device) if predictor else 'not loaded'
    return HealthResponse(
        status='healthy' if predictor else 'model not loaded',
        model_loaded=predictor is not None,
        model_version='efficientnet_b4_v1',
        device=device,
    )

@app.get('/model-info', response_model=ModelInfoResponse, tags=['System'])
async def model_info():
    """Get model architecture details and training performance."""
    return ModelInfoResponse(
        architecture='EfficientNet-B4 + custom classification head',
        parameters=18_471_242,
        input_size='224x224 RGB',
        classes=['no_stone', 'stone'],
        training_auc=1.0,
        training_sensitivity=1.0,
        threshold=predictor.threshold if predictor else 0.5,
    )

@app.post('/predict', response_model=PredictionResponse, tags=['Inference'])
async def predict(
    file: UploadFile = File(..., description='CT or ultrasound image (JPG/PNG)'),
    include_gradcam: bool = Query(True, description='Include Grad-CAM heatmap in response'),
):
    """
    Predict whether an image contains a kidney stone.

    - **file**: Upload a JPG or PNG image
    - **include_gradcam**: Set to false for faster response without heatmap
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    # Validate file type
    if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type: {file.content_type}. Use JPG or PNG.'
        )

    try:
        image_bytes = await file.read()
        start = time.time()
        result = predictor.predict(image_bytes, include_gradcam=include_gradcam)
        elapsed = round(time.time() - start, 3)
        print(f'Predicted: {result["prediction"]} ({result["confidence"]:.2%}) in {elapsed}s')
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference error: {str(e)}')

@app.post('/predict/batch', tags=['Inference'])
async def predict_batch(
    files: List[UploadFile] = File(..., description='Multiple images'),
    include_gradcam: bool = Query(False, description='Include Grad-CAM (slower)'),
):
    """Run prediction on multiple images at once (max 10)."""
    if predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    if len(files) > 10:
        raise HTTPException(status_code=400, detail='Max 10 images per batch')

    results = []
    for f in files:
        try:
            image_bytes = await f.read()
            result = predictor.predict(image_bytes, include_gradcam=include_gradcam)
            result['filename'] = f.filename
            results.append(result)
        except Exception as e:
            results.append({'filename': f.filename, 'error': str(e)})
    return {'predictions': results, 'total': len(results)}


    
    
