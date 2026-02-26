# api/metrics.py
# Custom Prometheus metrics for Kidney Stone Detection API


from prometheus_client import Counter, Histogram, Gauge


# Total predictions made
PREDICTIONS_TOTAL = Counter(
    'kidney_predictions_total',
    'Total number of predictions made',
    ['prediction']   # label: 'stone' or 'no_stone'
)


# Confidence score histogram
CONFIDENCE_HISTOGRAM = Histogram(
    'kidney_confidence_score',
    'Distribution of confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)


# Inference latency
INFERENCE_LATENCY = Histogram(
    'kidney_inference_latency_seconds',
    'Time taken for model inference',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)


# Model status gauge (1 = loaded, 0 = not loaded)
MODEL_STATUS = Gauge(
    'kidney_model_loaded',
    'Whether the model is currently loaded'
)


# Active requests
ACTIVE_REQUESTS = Gauge(
    'kidney_active_requests',
    'Number of requests currently being processed'
)
