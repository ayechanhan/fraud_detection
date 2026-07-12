"""
Flask API that serves the registered fraud model.

Endpoints:
    GET  /health   liveness check + which model version is loaded (no key needed)
    POST /predict  send one application's fields, get a fraud probability back
    POST /reload   reload the latest registered model (use after a retrain)

We load the LATEST model from the mlflow registry ourselves and serve it with a
plain Flask route - deliberately NOT mlflow's own server - so the serving path is
simple code we fully own and can explain. /predict and /reload require the API key.

Run from the repo root (after training a model):
    python service/app.py
"""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

from auth import require_api_key

MODEL_NAME = "fraud_model"
MODEL_URI = f"models:/{MODEL_NAME}/latest"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# The exact fields an incoming application must provide (the model's inputs).
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type",
]
FRAUD_THRESHOLD = 0.5  # probability at/above which we label the application "fraud"

app = Flask(__name__)

# The model is loaded once at startup and cached in these module globals.
_model = None
_model_version = None


def load_model():
    """Fetch the latest registered model from mlflow and cache it in memory.

    Called at startup and by /reload, so the running service can pick up a newly
    retrained model version without a restart.
    """
    global _model, _model_version
    mlflow.set_tracking_uri(TRACKING_URI)
    _model = mlflow.sklearn.load_model(MODEL_URI)
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    _model_version = max((int(v.version) for v in versions), default=None)
    return _model_version


@app.get("/health")
def health():
    """Cheap liveness check - also reports which model version is serving."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
    })


@app.post("/predict")
@require_api_key
def predict():
    """Score a single application for fraud probability."""
    if _model is None:
        return jsonify({"error": "No model loaded. Train and register a model first."}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with the application fields."}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Build the one-row frame the model expects, then read the fraud probability.
    row = pd.DataFrame([{f: data[f] for f in FEATURES}])
    probability = float(_model.predict_proba(row)[0, 1])

    return jsonify({
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= FRAUD_THRESHOLD,
        "threshold": FRAUD_THRESHOLD,
        "model_version": _model_version,
    })


@app.post("/reload")
@require_api_key
def reload():
    """Reload the latest registered model (call this right after a retrain)."""
    try:
        version = load_model()
        return jsonify({"status": "reloaded", "model_version": version})
    except Exception as exc:  # noqa: BLE001 - surface any load error to the caller
        return jsonify({"error": f"Reload failed: {exc}"}), 500


# Try to load a model at startup, but still start if the registry is empty
# (so /health works and the error is a clear 503 instead of a crash).
try:
    load_model()
except Exception as exc:  # noqa: BLE001
    app.logger.warning("Could not load a model at startup: %s", exc)


if __name__ == "__main__":
    # Port 8000 to avoid clashing with the mlflow UI (5000) and macOS AirPlay.
    app.run(host="127.0.0.1", port=8000, debug=False)
