"""
DemandIQ - REST API
===================

Flask backend for the Intelligent Demand Forecasting Agent dashboard.

Endpoints
---------
GET  /                        the dashboard
GET  /api/health              service + model status
POST /api/upload              multipart CSV upload -> dataset_id
POST /api/sample              load the built-in demo dataset -> dataset_id
GET  /api/datasets/<id>       dataset profile (columns, range, products)
POST /api/forecast            run the pipeline, return dashboard payload
GET  /api/products/<id>       list products in a dataset
GET  /api/product/<id>/<sku>  deep dive on one product
GET  /api/export/<id>.csv     inventory recommendations as CSV
GET  /api/sample.csv          download the demo CSV

Run:  python app.py       (http://127.0.0.1:5000)
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

try:                                   # optional: only needed if the UI is served elsewhere
    from flask_cors import CORS
except ImportError:
    CORS = None

from forecasting_engine import (
    DemandForecaster,
    ForecastConfig,
    MODEL_LABELS,
    detect_columns,
    summarize,
    to_table,
)
from sample_data import generate_sales_data

BASE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Configuration (all overridable via environment variables, so the same
# image/codebase runs the same way locally and on a real host)
# --------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 25))
MAX_DATASETS = int(os.environ.get("MAX_DATASETS", 20))
DEBUG = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

logging.basicConfig(level=logging.INFO if not DEBUG else logging.DEBUG)
logger = logging.getLogger("demandiq")

app = Flask(__name__, static_folder=str(BASE_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["DEBUG"] = DEBUG

if CORS:
    # With no ALLOWED_ORIGINS set, CORS is left at its permissive default,
    # which is fine when the dashboard is served from the same Flask app.
    # Set ALLOWED_ORIGINS if the frontend is ever hosted separately.
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS or "*"}})

# In-memory dataset store. Swap for Redis or a database if you need persistence
# across processes - the API contract does not change. Note: with multiple
# gunicorn workers, each worker has its own copy of this dict (see README).
DATASETS: Dict[str, Dict[str, Any]] = {}
CACHE: Dict[str, Dict[str, Any]] = {}


def _store(df: pd.DataFrame, name: str) -> str:
    if len(DATASETS) >= MAX_DATASETS:                     # evict the oldest
        oldest = min(DATASETS, key=lambda k: DATASETS[k]["created"])
        DATASETS.pop(oldest, None)
        CACHE.pop(oldest, None)
    dataset_id = uuid.uuid4().hex[:12]
    DATASETS[dataset_id] = {"df": df, "name": name, "created": time.time()}
    return dataset_id


def _profile(dataset_id: str) -> Dict[str, Any]:
    entry = DATASETS[dataset_id]
    df, mapping = entry["df"], detect_columns(entry["df"])
    dates = pd.to_datetime(df[mapping["date"]], errors="coerce") if mapping["date"] else None
    products = (df[mapping["product"]].astype(str).nunique() if mapping["product"] else 1)
    return {
        "dataset_id": dataset_id,
        "name": entry["name"],
        "rows": int(len(df)),
        "columns": list(df.columns),
        "detected_columns": mapping,
        "products": int(products),
        "date_range": ([str(dates.min().date()), str(dates.max().date())]
                       if dates is not None and dates.notna().any() else None),
        "uploaded_at": datetime.fromtimestamp(entry["created"], timezone.utc).isoformat(),
    }


def _error(message: str, code: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), code


# --------------------------------------------------------------------------
# Static
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "demand_forecasting_frontend.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "service": "DemandIQ forecasting API",
        "version": "1.0.0",
        "time": datetime.now(timezone.utc).isoformat(),
        "models": MODEL_LABELS,
        "datasets_in_memory": len(DATASETS),
        "capabilities": [
            "historical sales analysis", "time-series processing", "demand prediction",
            "trend detection", "seasonality detection", "anomaly handling",
            "inventory recommendation",
        ],
    })


# --------------------------------------------------------------------------
# Data intake
# --------------------------------------------------------------------------

@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return _error("No file part in the request. Send the CSV as form field 'file'.")
    f = request.files["file"]
    if not f.filename:
        return _error("No file selected.")
    if not f.filename.lower().endswith((".csv", ".txt", ".tsv")):
        return _error("Only CSV files are supported.")

    try:
        raw = f.read()
        sep = "\t" if f.filename.lower().endswith(".tsv") else None
        df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
    except Exception as exc:
        return _error(f"Could not read the CSV: {exc}")

    if df.empty:
        return _error("The file has no rows.")

    mapping = detect_columns(df)
    if not mapping["date"] or not mapping["quantity"]:
        return _error(
            "Could not identify a date column and a quantity column.",
            columns=list(df.columns),
            hint="Rename your columns to something like date, product_id, units_sold.",
        )

    dataset_id = _store(df, f.filename)
    return jsonify({"ok": True, **_profile(dataset_id)})


@app.post("/api/sample")
def sample():
    days = int(request.json.get("days", 730)) if request.is_json and request.json else 730
    df = generate_sales_data(days=min(max(days, 60), 1460))
    dataset_id = _store(df, "sample_sales_history.csv")
    return jsonify({"ok": True, **_profile(dataset_id)})


@app.get("/api/sample.csv")
def sample_csv():
    csv = generate_sales_data().to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_sales_history.csv"})


@app.get("/api/datasets/<dataset_id>")
def dataset_info(dataset_id: str):
    if dataset_id not in DATASETS:
        return _error("Unknown dataset id. Upload a file or load the sample data first.", 404)
    return jsonify({"ok": True, **_profile(dataset_id)})


@app.get("/api/products/<dataset_id>")
def products(dataset_id: str):
    if dataset_id not in DATASETS:
        return _error("Unknown dataset id.", 404)
    fc = DemandForecaster(DATASETS[dataset_id]["df"])
    return jsonify({"ok": True, "products": fc.products, "count": len(fc.products)})


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

def _config_from_request(payload: Dict[str, Any]) -> ForecastConfig:
    horizon = payload.get("horizon", 30)
    if isinstance(horizon, str):                          # "30 days" from the dropdown
        horizon = int("".join(ch for ch in horizon if ch.isdigit()) or 30)
    confidence = payload.get("confidence", 0.90)
    if isinstance(confidence, str):
        confidence = float("".join(ch for ch in confidence if ch.isdigit() or ch == ".") or 90)
    if confidence > 1:
        confidence /= 100.0

    return ForecastConfig(
        horizon=int(min(max(horizon, 1), 365)),
        confidence=min(max(float(confidence), 0.5), 0.99),
        frequency=payload.get("frequency", "D").upper()[:1],
        model=str(payload.get("model", "auto")).lower().strip().replace(" ", "_"),
        lead_time_days=int(payload.get("lead_time_days", 7)),
        review_period_days=int(payload.get("review_period_days", 7)),
        anomaly_threshold=float(payload.get("anomaly_threshold", 3.5)),
        clean_anomalies=bool(payload.get("clean_anomalies", True)),
    )


@app.post("/api/forecast")
def forecast():
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("dataset_id")
    if not dataset_id or dataset_id not in DATASETS:
        return _error("Unknown or missing dataset_id. Upload a CSV or load the sample data first.", 404)

    cfg = _config_from_request(payload)
    started = time.time()

    try:
        forecaster = DemandForecaster(DATASETS[dataset_id]["df"], cfg)
        run = forecaster.analyze_all(limit=payload.get("limit"))
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:                              # unexpected: report, do not crash
        return _error(f"Forecast failed: {exc}", 500)

    results = run["results"]
    if not results:
        return _error("No product had enough history to forecast.", 422, details=run["errors"])

    table = to_table(results)
    overview = summarize(results)

    focus_name = payload.get("product") or table[0]["product"]
    focus = next((r for r in results if r["product"] == focus_name), results[0])

    response = {
        "ok": True,
        "dataset_id": dataset_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 2),
        "config": {
            "horizon": cfg.horizon,
            "confidence_pct": round(cfg.confidence * 100, 1),
            "model_requested": cfg.model,
            "frequency": cfg.frequency,
            "lead_time_days": cfg.lead_time_days,
        },
        "preparation": run["preparation"],
        "overview": overview,
        "chart": {
            "product": focus["product"],
            "history_dates": focus["history"]["dates"][-90:],
            "history": focus["history"]["actual"][-90:],
            "forecast_dates": focus["forecast"]["dates"],
            "forecast": focus["forecast"]["values"],
            "lower": focus["forecast"]["lower"],
            "upper": focus["forecast"]["upper"],
            "model": focus["model_label"],
        },
        "table": table,
        "anomalies": [
            {"product": r["product"], **item}
            for r in results for item in r["anomalies"]["items"]
        ][:100],
        "products": [r["product"] for r in results],
        "skipped": run["errors"],
    }
    CACHE[dataset_id] = response
    return jsonify(response)


@app.get("/api/product/<dataset_id>/<path:product>")
def product_detail(dataset_id: str, product: str):
    if dataset_id not in DATASETS:
        return _error("Unknown dataset id.", 404)
    cfg = _config_from_request(request.args.to_dict())
    try:
        detail = DemandForecaster(DATASETS[dataset_id]["df"], cfg).analyze_product(product)
    except ValueError as exc:
        return _error(str(exc), 404)
    return jsonify({"ok": True, "detail": detail})


@app.get("/api/export/<dataset_id>.csv")
def export(dataset_id: str):
    cached = CACHE.get(dataset_id)
    if not cached:
        return _error("Run a forecast for this dataset before exporting.", 404)
    df = pd.DataFrame(cached["table"])
    stamp = datetime.now().strftime("%Y%m%d")
    return Response(
        df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=demand_forecast_{stamp}.csv"},
    )


@app.errorhandler(413)
def too_large(_):
    return _error(f"File is larger than {MAX_UPLOAD_MB} MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return _error("Endpoint not found.", 404)


@app.errorhandler(500)
def server_error(exc):
    logger.exception("Unhandled server error")
    return _error("Internal server error.", 500)


if __name__ == "__main__":
    # Local development only. In production this app is served by gunicorn
    # (see Procfile / Dockerfile) which imports `app` directly and never
    # runs this block.
    port = int(os.environ.get("PORT", 5000))
    print(f"DemandIQ API  ->  http://127.0.0.1:{port}  (debug={DEBUG})")
    app.run(host="0.0.0.0", port=port, debug=DEBUG)
