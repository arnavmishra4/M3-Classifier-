"""
inference.py — M3 Inference (Progression Classifier: XGBoost + MLP, Platt-calibrated)

FastAPI usage:
    from inference import run_m3, preload_m3_model

    # optional: call once at app startup to avoid first-request disk-load latency
    preload_m3_model()

    result = run_m3(delta_mu_r, delta_mu_d, delta_gamma, volume_delta, edema_delta)
"""

import numpy as np
import joblib

LABEL_INV = {0: 'True_Progression', 1: 'Treatment_Working', 2: 'Treatment_Failing'}
CONF_LOW  = 0.55
CONF_HIGH = 0.75
N_CLASSES = 3

# ---------------------------------------------------------------------------
# Model cache — load once, reuse across requests instead of hitting disk
# on every call. Keyed by model_path so multiple checkpoints can coexist
# (e.g. swapping models without restarting the process).
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_path: str):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = joblib.load(model_path)
    return _MODEL_CACHE[model_path]


def preload_m3_model(model_path: str = "m3_calibrated_model.pkl") -> None:
    """Call once at FastAPI startup (lifespan) to warm the cache ahead of the first request."""
    _get_model(model_path)


def _entropy(probs):
    raw = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)))
    return float(raw / np.log(N_CLASSES))

def _gate(max_prob, entropy):
    if max_prob < CONF_LOW:
        return "LOW",  "ESCALATE"
    elif max_prob < CONF_HIGH or entropy > 0.70:
        return "MID",  "TENTATIVE"
    else:
        return "HIGH", "FULL_PASS"


def run_m3(
    delta_mu_r:   float,
    delta_mu_d:   float,
    delta_gamma:  float,
    volume_delta: float,
    edema_delta:  float,
    model_path:   str = "m3_calibrated_model.pkl",
) -> dict:
    calibrated = _get_model(model_path)

    x     = np.array([[delta_mu_r, delta_mu_d, delta_gamma, volume_delta, edema_delta]])
    probs = calibrated.predict_proba(x)[0]

    pred_idx = int(np.argmax(probs))
    max_prob = float(probs[pred_idx])
    entropy  = _entropy(probs)
    band, action = _gate(max_prob, entropy)

    return {
        "progression_class":   LABEL_INV[pred_idx] if action != "ESCALATE" else None,
        "confidence":          round(max_prob, 4),
        "entropy":             round(entropy, 4),
        "class_probabilities": {LABEL_INV[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "confidence_band":     band,   # LOW | MID | HIGH
        "action":              action, # ESCALATE | TENTATIVE | FULL_PASS
    }