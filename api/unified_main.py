# api/unified_main.py
import os, sys, time, json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.unified_inference import UnifiedPredictor
from api.unified_report    import generate_report

predictor: UnifiedPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    os.chdir(str(Path(__file__).parent.parent))
    predictor = UnifiedPredictor(ckpt_dir='checkpoints')
    yield

app = FastAPI(
    title='NephroScan AI — Unified',
    description='3-model kidney analysis: Stone Detection + 4-Class + Cancer Detection',
    version='3.0.0',
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

class UnifiedResponse(BaseModel):
    v1:              dict
    v2:              dict
    v3:              dict
    risk_level:      str
    gradcam_heatmap: Optional[str] = None
    model_versions:  list

@app.get('/health')
async def health():
    return {
        'status':        'healthy' if predictor else 'loading',
        'models_loaded': predictor is not None,
        'version':       '3.0.0',
        'device':        str(predictor.device) if predictor else 'none',
        'models':        ['v1_stone', 'v2_4class', 'v3_cancer'],
    }

@app.get('/model-info')
async def model_info():
    return {
        'v1': {'name':'Stone Detector',     'classes':['no_stone','stone'],              'accuracy':0.992},
        'v2': {'name':'4-Class Classifier', 'classes':['normal','cyst','stone','tumour'],'accuracy':0.970},
        'v3': {'name':'Cancer Detector',    'classes':['not_cancer','cancer'],           'auc':0.9999},
    }

@app.post('/predict', response_model=UnifiedResponse)
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(True),
):
    if predictor is None:
        raise HTTPException(503, 'Models not loaded yet')
    if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise HTTPException(400, f'Unsupported file type: {file.content_type}')
    try:
        t      = time.time()
        result = predictor.predict(await file.read(), include_gradcam=include_gradcam)
        print(f'Predict: risk={result["risk_level"]} v2={result["v2"]["prediction"]} in {time.time()-t:.2f}s')
        return UnifiedResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f'Error: {str(e)}')

@app.post('/report')
async def report(
    file:   UploadFile = File(...),
    result: str        = Form(...),
):
    """Generate and return a clinical PDF report."""
    try:
        result_dict  = json.loads(result)
        image_bytes  = await file.read()
        pdf_bytes    = generate_report(result_dict, image_bytes, file.filename)

        # Return as PDF if reportlab available, else HTML
        is_pdf = pdf_bytes[:4] == b'%PDF'
        media  = 'application/pdf' if is_pdf else 'text/html'
        ext    = 'pdf' if is_pdf else 'html'

        return Response(
            content=pdf_bytes,
            media_type=media,
            headers={
                'Content-Disposition': f'attachment; filename="nephroscan_report_{int(time.time())}.{ext}"'
            }
        )
    except Exception as e:
        raise HTTPException(500, f'Report generation failed: {str(e)}')

# ── Start ─────────────────────────────────────────────────────────────────────
# cd '/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn'
# /Users/devaguru/Kidney\ Stone\ CNN/.venv/bin/uvicorn api.unified_main:app --port 8000 --reload