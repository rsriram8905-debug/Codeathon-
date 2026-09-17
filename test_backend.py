"""
Tests for the DemandIQ backend.  Run:  python test_backend.py   (or: pytest -q)

Each test checks a stated requirement, so a failure tells you which capability broke.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from app import app
from forecasting_engine import (
    DemandForecaster,
    ForecastConfig,
    detect_anomalies,
    detect_columns,
    detect_seasonality,
    detect_trend,
    inventory_recommendation,
    metrics,
    summarize,
    to_table,
)
from sample_data import generate_sales_data


def test_trend_detection():
    t = np.arange(120)
    assert detect_trend(50 + 2.0 * t)["direction"] == "Increasing"
    assert detect_trend(400 - 2.0 * t)["direction"] == "Decreasing"
    flat = 100 + np.random.default_rng(0).normal(0, 3, 120)
    assert detect_trend(flat)["direction"] == "Stable"


def test_seasonality_detection():
    t = np.arange(400)
    weekly = 100 + 30 * np.sin(2 * np.pi * t / 7)
    assert detect_seasonality(weekly)["period"] == 7
    noise = 100 + np.random.default_rng(1).normal(0, 5, 400)
    assert detect_seasonality(noise)["detected"] is False


def test_anomaly_detection_and_cleaning():
    rng = np.random.default_rng(2)
    y = 100 + rng.normal(0, 4, 300)
    y[50] = 400          # promotion spike
    y[120] = 2           # stockout
    out = detect_anomalies(y)
    assert 50 in out["indices"] and 120 in out["indices"]
    assert out["count"] < 15                          # no flood of false positives
    assert out["cleaned"][50] < 200                   # spike pulled back to the local level


def test_metrics_are_sane():
    actual = np.array([10.0, 20.0, 30.0])
    assert metrics(actual, actual)["accuracy_pct"] == 100.0
    assert metrics(actual, actual * 2)["accuracy_pct"] < 80


def test_inventory_policy():
    forecast = np.full(30, 10.0)
    low = inventory_recommendation(forecast, 2.0, current_stock=5, lead_time=7,
                                   review_period=7, confidence=0.9)
    assert low["stock_status"] == "Stockout risk" and low["suggested_order_qty"] > 0

    high = inventory_recommendation(forecast, 2.0, current_stock=2000, lead_time=7,
                                    review_period=7, confidence=0.9)
    assert high["stock_status"] == "Overstock" and high["suggested_order_qty"] == 0
    assert high["excess_units"] > 0


def test_column_detection_is_flexible():
    df = pd.DataFrame({"Order Date": ["2025-01-01"], "SKU": ["A"], "Units Sold": [5],
                       "On Hand": [10], "Unit Price": [99]})
    m = detect_columns(df)
    assert (m["date"], m["product"], m["quantity"], m["stock"], m["price"]) == \
           ("Order Date", "SKU", "Units Sold", "On Hand", "Unit Price")


def test_missing_days_are_filled():
    dates = pd.date_range("2025-01-01", periods=60).delete([5, 6, 7])
    df = pd.DataFrame({"date": dates, "sku": "A", "units": np.arange(len(dates)) + 10})
    fc = DemandForecaster(df)
    assert fc.prep_report["missing_periods_filled"] == 3
    assert len(fc.get_series("A")) == 60


def test_forecast_shape_and_intervals():
    fc = DemandForecaster(generate_sales_data(days=400), ForecastConfig(horizon=21))
    res = fc.analyze_product(fc.products[0])
    f = res["forecast"]
    assert len(f["values"]) == len(f["lower"]) == len(f["upper"]) == 21
    assert all(l <= v <= u for l, v, u in zip(f["lower"], f["values"], f["upper"]))
    assert all(v >= 0 for v in f["values"])           # demand is never negative
    assert res["accuracy"]["accuracy_pct"] > 50       # beats nonsense on clean data


def test_full_run_and_table():
    run = DemandForecaster(generate_sales_data(days=540)).analyze_all()
    assert run["errors"] == []
    rows = to_table(run["results"])
    assert len(rows) == len(run["results"])
    assert summarize(run["results"])["products_analyzed"] == len(rows)


def test_api_flow():
    c = app.test_client()
    assert c.get("/api/health").get_json()["ok"]

    loaded = c.post("/api/sample", json={"days": 400}).get_json()
    did = loaded["dataset_id"]

    fc = c.post("/api/forecast", json={"dataset_id": did, "horizon": "30 days",
                                       "confidence": "90%", "model": "auto"}).get_json()
    assert fc["ok"] and fc["table"] and fc["overview"]["products_analyzed"] == 6
    assert c.get(f"/api/export/{did}.csv").status_code == 200
    assert c.post("/api/forecast", json={"dataset_id": "missing"}).status_code == 404


def test_api_rejects_unusable_file():
    c = app.test_client()
    buf = io.BytesIO(b"foo,bar\n1,2\n")
    res = c.post("/api/upload", data={"file": (buf, "bad.csv")},
                 content_type="multipart/form-data")
    assert res.status_code == 400 and res.get_json()["ok"] is False


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
