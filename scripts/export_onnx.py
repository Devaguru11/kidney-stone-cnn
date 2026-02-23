# scripts/export_onnx.py
import os, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir('/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn')

from src.models.efficientnet import KidneyStoneClassifier

CHECKPOINT = 'checkpoints/best_model.pth'
ONNX_OUT   = 'checkpoints/best_model.onnx'

# Load PyTorch model
model = KidneyStoneClassifier()
model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
model.eval()

# Create dummy input — batch of 1 image, 3 channels, 224x224
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUT,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={
        'image':  {0: 'batch_size'},
        'logits': {0: 'batch_size'},
    }
)
print(f'ONNX model saved to: {ONNX_OUT}')

# Verify ONNX model is valid
import onnx
onnx_model = onnx.load(ONNX_OUT)
onnx.checker.check_model(onnx_model)
print('ONNX model validation: PASSED')

# Quick speed comparison
import onnxruntime as ort, numpy as np, time
sess = ort.InferenceSession(ONNX_OUT)
inp  = dummy_input.numpy()

# PyTorch timing
t0 = time.time()
for _ in range(50): model(dummy_input)
pytorch_ms = (time.time()-t0)/50*1000

# ONNX timing
t0 = time.time()
for _ in range(50): sess.run(None, {'image': inp})
onnx_ms = (time.time()-t0)/50*1000

print(f'PyTorch CPU latency: {pytorch_ms:.1f}ms per image')
print(f'ONNX CPU latency:    {onnx_ms:.1f}ms per image')
print(f'Speedup:             {pytorch_ms/onnx_ms:.1f}x')
