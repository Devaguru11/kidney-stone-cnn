# api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class PredictionResponse(BaseModel):
    """Response from POST /predict"""
    prediction:   str   = Field(..., description="'stone' or 'no_stone'")
    confidence:   float = Field(..., description="Model confidence 0.0–1.0")
    probability_stone:    float = Field(..., description="P(stone)")
    probability_no_stone: float = Field(..., description="P(no_stone)")
    gradcam_heatmap: Optional[str] = Field(None, description="Base64 PNG heatmap")
    model_version:   str = Field(default='efficientnet_b4_v1')
    threshold_used:  float = Field(default=0.5)

class HealthResponse(BaseModel):
    """Response from GET /health"""
    status:        str
    model_loaded:  bool
    model_version: str
    device:        str

class ModelInfoResponse(BaseModel):
    """Response from GET /model-info"""
    architecture:   str
    parameters:     int
    input_size:     str
    classes:        list
    training_auc:   float
    training_sensitivity: float
    threshold:      float
