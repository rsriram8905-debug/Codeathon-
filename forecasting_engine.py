"""
DemandIQ - forecasting engine
=============================

Pure numpy/pandas implementation of the demand-forecasting pipeline:

    raw CSV  ->  column mapping  ->  daily series per product
             ->  anomaly detection + cleaning
             ->  trend + seasonality analysis
             ->  model selection via rolling backtest
             ->  forecast + prediction intervals
             ->  inventory recommendation

No network calls and no heavyweight ML dependencies, so it runs anywhere
pandas runs. Every model here is implemented from scratch and is explainable,
which matters more for a supply-chain tool than raw model complexity.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------

DATE_ALIASES = ["date", "order_date", "sale_date", "day", "timestamp", "period", "ds", "week", "month"]
PRODUCT_ALIASES = ["product", "product_id", "sku", "item", "item_id", "product_name", "material", "article"]
QTY_ALIASES = ["quantity", "qty", "units", "units_sold", "sales", "demand", "sold", "volume", "y", "amount"]
PRICE_ALIASES = ["price", "unit_price", "selling_price", "mrp", "rate"]
STOCK_ALIASES = ["stock", "current_stock", "on_hand", "inventory", "stock_on_hand", "available_qty"]
LEADTIME_ALIASES = ["lead_time", "leadtime", "lead_time_days", "supplier_lead_time"]


def _match(columns: List[str], aliases: List[str]) -> Optional[str]:
    """Find the column whose normalised name best matches a list of aliases."""
    norm = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in columns}
    for alias in aliases:                         # exact match first
        for original, n in norm.items():
            if n == alias:
                return original
    for alias in aliases:                         # then substring match
        for original, n in norm.items():
            if alias in n:
                return original
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    mapping = {
        "date": _match(cols, DATE_ALIASES),
        "product": _match(cols, PRODUCT_ALIASES),
        "quantity": _match(cols, QTY_ALIASES),
        "price": _match(cols, PRICE_ALIASES),
        "stock": _match(cols, STOCK_ALIASES),
        "lead_time": _match(cols, LEADTIME_ALIASES),
    }
    # Fallbacks: first parseable datetime column, first numeric column.
    if mapping["date"] is None:
        for c in cols:
            col = df[c]
            if pd.api.types.is_numeric_dtype(col):
                # A plain number column is a measure, not a date - unless it is
                # packed like 20250131.
                looks_packed = col.dropna().between(19000101, 21001231).all() and not col.empty
                if not looks_packed:
                    continue
                col = col.astype("Int64").astype(str)
            parsed = _parse_dates(col)
            if parsed.notna().mean() > 0.8:
                mapping["date"] = c
                break
    if mapping["quantity"] is None:
        for c in cols:
            if c != mapping["date"] and pd.api.types.is_numeric_dtype(df[c]):
                mapping["quantity"] = c
                break
    return mapping


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class ForecastConfig:
    horizon: int = 30                  # days to forecast
    confidence: float = 0.90           # prediction-interval level
    frequency: str = "D"               # D | W | M
    model: str = "auto"                # auto | ridge | holt_winters | seasonal_naive | moving_average
    lead_time_days: int = 7            # supplier lead time for inventory maths
    review_period_days: int = 7        # how often stock is reviewed / reordered
    anomaly_threshold: float = 3.5     # robust z-score cut-off
    clean_anomalies: bool = True       # replace outliers before training
    max_products: int = 500            # guard rail for very wide files


Z_TABLE = {0.80: 1.2816, 0.85: 1.4395, 0.90: 1.6449, 0.95: 1.9600, 0.975: 2.2414, 0.99: 2.5758}


def z_score(conf: float) -> float:
    if conf in Z_TABLE:
        return Z_TABLE[conf]
    keys = sorted(Z_TABLE)
    conf = min(max(conf, keys[0]), keys[-1])
    return float(np.interp(conf, keys, [Z_TABLE[k] for k in keys]))


FREQ_PERIODS = {"D": 7, "W": 52, "M": 12}     # dominant season length per frequency
TREND_LOOKBACK = {"D": 180, "W": 52, "M": 24} # how far back "current trend" looks


# --------------------------------------------------------------------------
# Step 1 - time-series preparation
# --------------------------------------------------------------------------

def _parse_dates(col: pd.Series) -> pd.Series:
    """Parse dates tolerantly: ISO, US and day-first CSVs all show up in the wild."""
    attempts = []
    warnings.filterwarnings("ignore", message=".*dayfirst.*")
    for kwargs in ({"format": "mixed", "dayfirst": False},
                   {"format": "mixed", "dayfirst": True},
                   {"dayfirst": False},
                   {"dayfirst": True}):
        try:
            parsed = pd.to_datetime(col, errors="coerce", **kwargs)
        except Exception:
            continue
        attempts.append(parsed)
    if not attempts:
        return pd.to_datetime(col, errors="coerce")

    best_rate = max(p.notna().mean() for p in attempts)
    good = [p for p in attempts if p.notna().mean() >= best_rate - 0.01]

    def span(p: pd.Series) -> float:
        valid = p.dropna()
        if valid.empty:
            return float("inf")
        return (valid.max() - valid.min()).total_seconds()

    # 03/04/2025 is ambiguous. A wrong reading scatters dates across a wider,
    # gappier range, so the tightest span is the right one.
    return min(good, key=span)


def prepare_timeseries(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    frequency: str = "D",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw rows and aggregate them into one regular series per product.

    Returns a long dataframe with columns [product, date, quantity] on a gap-free
    calendar, plus a report describing what was repaired.
    """
    date_col, prod_col, qty_col = mapping["date"], mapping["product"], mapping["quantity"]
    if date_col is None or qty_col is None:
        raise ValueError(
            "Could not find a date column and a quantity column in the file. "
            "Expected something like 'date' and 'units_sold'."
        )

    work = df.copy()
    raw_rows = len(work)

    work["__date"] = _parse_dates(work[date_col])
    bad_dates = int(work["__date"].isna().sum())
    work = work.dropna(subset=["__date"])

    work["__qty"] = pd.to_numeric(
        work[qty_col].astype(str).str.replace(r"[,₹$\s]", "", regex=True), errors="coerce"
    )
    bad_qty = int(work["__qty"].isna().sum())
    work = work.dropna(subset=["__qty"])

    negatives = int((work["__qty"] < 0).sum())        # returns / corrections
    work["__qty"] = work["__qty"].clip(lower=0)

    if prod_col is None:
        work["__product"] = "All products"
    else:
        work["__product"] = work[prod_col].astype(str).str.strip().replace("", "Unknown")

    duplicates = int(work.duplicated(subset=["__product", "__date"]).sum())

    grouped = (
        work.groupby(["__product", pd.Grouper(key="__date", freq=frequency)])["__qty"]
        .sum()
        .reset_index()
        .rename(columns={"__product": "product", "__date": "date", "__qty": "quantity"})
    )

    # Reindex every product onto the same gap-free calendar; missing day = no sale.
    full_index = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq=frequency)
    frames, filled = [], 0
    for product, part in grouped.groupby("product"):
        s = part.set_index("date")["quantity"].reindex(full_index)
        filled += int(s.isna().sum())
        s = s.fillna(0.0)
        frames.append(pd.DataFrame({"product": product, "date": s.index, "quantity": s.values}))

    tidy = pd.concat(frames, ignore_index=True)

    report = {
        "raw_rows": raw_rows,
        "usable_rows": int(len(work)),
        "rows_dropped_bad_date": bad_dates,
        "rows_dropped_bad_quantity": bad_qty,
        "negative_quantities_clipped": negatives,
        "duplicate_rows_aggregated": duplicates,
        "missing_periods_filled": filled,
        "periods": int(len(full_index)),
        "products": int(tidy["product"].nunique()),
        "date_start": str(full_index.min().date()),
        "date_end": str(full_index.max().date()),
        "frequency": frequency,
    }
    return tidy, report


# --------------------------------------------------------------------------
# Step 2 - anomaly detection (robust, works on short series)
# --------------------------------------------------------------------------

def detect_anomalies(y: np.ndarray, threshold: float = 3.5, window: int = 7) -> Dict[str, Any]:
    """Rolling-median + MAD outlier detection.

    A median/MAD baseline is used instead of mean/std because a promotion spike
    would otherwise inflate the std and hide itself.
    """
    n = len(y)
    s = pd.Series(y, dtype="float64")
    window = max(5, min(window, n if n % 2 else n - 1))
    if window % 2 == 0:
        window += 1

    baseline = s.rolling(window, center=True, min_periods=3).median()
    baseline = baseline.bfill().ffill()
    resid = s - baseline

    # Scale locally as well as globally: retail volumes are heteroscedastic, so a
    # single global MAD would flag ordinary noise in the busy months and miss real
    # outliers in the quiet ones.
    global_mad = float(np.median(np.abs(resid - np.median(resid))))
    global_scale = 1.4826 * global_mad if global_mad > 1e-9 else float(resid.std(ddof=0)) or 1.0

    local_mad = resid.rolling(max(window * 4 + 1, 21), center=True, min_periods=5).apply(
        lambda v: np.median(np.abs(v - np.median(v))), raw=True
    )
    scale = (1.4826 * local_mad).bfill().ffill()
    floor = max(0.25 * global_scale, 0.05 * float(np.mean(np.abs(s))), 1e-6)
    scale = scale.clip(lower=floor)

    robust_z = (resid - np.median(resid)) / scale

    flags = np.abs(robust_z.values) > threshold
    cleaned = s.copy()
    cleaned[flags] = baseline[flags]                   # winsorise onto the local level

    items = []
    for i in np.flatnonzero(flags):
        items.append({
            "index": int(i),
            "actual": round(float(y[i]), 2),
            "expected": round(float(baseline.iloc[i]), 2),
            "deviation_pct": round(float((y[i] - baseline.iloc[i]) / max(baseline.iloc[i], 1) * 100), 1),
            "z_score": round(float(robust_z.iloc[i]), 2),
            "type": "spike" if y[i] > baseline.iloc[i] else "drop",
            "likely_cause": ("promotion or bulk order" if y[i] > baseline.iloc[i]
                             else "stockout or data gap"),
            "severity": "high" if abs(robust_z.iloc[i]) > threshold * 1.8 else "medium",
        })

    return {
        "count": int(flags.sum()),
        "rate_pct": round(float(flags.mean() * 100), 2),
        "indices": np.flatnonzero(flags).tolist(),
        "items": items,
        "cleaned": cleaned.values,
    }


# --------------------------------------------------------------------------
# Step 3 - trend & seasonality
# --------------------------------------------------------------------------

def detect_trend(y: np.ndarray, lookback: Optional[int] = None) -> Dict[str, Any]:
    """OLS slope with a t-test, so 'Stable' means statistically flat, not eyeballed."""
    seg = y if lookback is None else y[-lookback:]
    n = len(seg)
    if n < 4:
        return {"direction": "Stable", "slope_per_period": 0.0, "change_pct": 0.0,
                "t_stat": 0.0, "significant": False, "strength": "insufficient data"}

    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, seg, 1)
    fitted = slope * x + intercept
    resid = seg - fitted
    dof = max(n - 2, 1)
    se_slope = math.sqrt(max((resid @ resid) / dof, 1e-12) / max(((x - x.mean()) ** 2).sum(), 1e-12))
    t_stat = slope / se_slope if se_slope > 0 else 0.0

    level = max(float(np.mean(seg)), 1e-9)
    change_pct = slope * n / level * 100                 # % change across the window
    significant = abs(t_stat) > 2.0

    if not significant or abs(change_pct) < 5:
        direction = "Stable"
    elif slope > 0:
        direction = "Increasing"
    else:
        direction = "Decreasing"

    strength = "strong" if abs(change_pct) > 25 else "moderate" if abs(change_pct) > 10 else "mild"
    return {
        "direction": direction,
        "slope_per_period": round(float(slope), 4),
        "change_pct": round(float(change_pct), 2),
        "t_stat": round(float(t_stat), 2),
        "significant": bool(significant),
        "strength": strength if direction != "Stable" else "flat",
        "window": n,
    }


def _acf(y: np.ndarray, lag: int) -> float:
    n = len(y)
    if lag >= n:
        return 0.0
    c = y - y.mean()
    denom = float(c @ c)
    if denom < 1e-12:
        return 0.0
    return float((c[lag:] @ c[:-lag]) / denom)


def detect_seasonality(y: np.ndarray, frequency: str = "D") -> Dict[str, Any]:
    """Look for repeating structure at the calendar periods that matter in retail."""
    n = len(y)
    candidates = {"D": [7, 14, 30, 91, 365], "W": [4, 13, 26, 52], "M": [3, 6, 12]}[frequency]
    candidates = [p for p in candidates if n >= int(1.75 * p) + 2]  # need ~2 cycles, allow 1.75

    scores = []
    for p in candidates:
        a = _acf(y, p)
        # Strength = share of variance explained by the period-p seasonal means.
        idx = np.arange(n) % p
        seasonal = np.array([y[idx == k].mean() for k in range(p)])[idx]
        var_total = float(np.var(y))
        strength = float(np.var(seasonal) / var_total) if var_total > 1e-12 else 0.0
        scores.append({"period": p, "acf": round(a, 3), "strength": round(strength, 3)})

    detected = [s for s in scores if s["acf"] > 0.2 and s["strength"] > 0.05]
    best = max(detected, key=lambda s: s["acf"]) if detected else None

    labels = {7: "weekly", 14: "fortnightly", 30: "monthly", 91: "quarterly", 365: "yearly",
              4: "monthly", 13: "quarterly", 26: "half-yearly", 52: "yearly",
              3: "quarterly", 6: "half-yearly", 12: "yearly"}

    peak_label = None
    if best:
        p = best["period"]
        idx = np.arange(n) % p
        means = np.array([y[idx == k].mean() for k in range(p)])
        if p == 7 and frequency == "D":
            # Align bucket 0 with the weekday of the first observation downstream.
            peak_label = f"peaks every {p} periods (offset {int(np.argmax(means))})"
        else:
            peak_label = f"peaks at position {int(np.argmax(means))} of {p}"

    return {
        "detected": best is not None,
        "period": best["period"] if best else None,
        "label": labels.get(best["period"]) if best else None,
        "strength": best["strength"] if best else 0.0,
        "acf": best["acf"] if best else 0.0,
        "peak": peak_label,
        "candidates": scores,
    }


# --------------------------------------------------------------------------
# Step 4 - models
# --------------------------------------------------------------------------

def _design_matrix(t: np.ndarray, n_train: int, seasonal_periods: List[int]) -> np.ndarray:
    """Trend + Fourier seasonality features. Deterministic in t, so multi-step
    forecasting needs no recursion and accumulates no feedback error."""
    cols = [np.ones_like(t, dtype=float), t / max(n_train, 1)]
    for p in seasonal_periods:
        k = 3 if p <= 14 else 2
        for h in range(1, k + 1):
            cols.append(np.sin(2 * np.pi * h * t / p))
            cols.append(np.cos(2 * np.pi * h * t / p))
    return np.column_stack(cols)


def ridge_seasonal(y: np.ndarray, horizon: int, frequency: str = "D",
                   alpha: float = 1.0) -> np.ndarray:
    """Ridge regression on trend + Fourier terms (closed form, no sklearn needed)."""
    n = len(y)
    periods = [p for p in ([7, 30, 365] if frequency == "D" else
                           [52] if frequency == "W" else [12]) if n >= int(1.75 * p)]
    if not periods and frequency == "D" and n >= 14:
        periods = [7]

    t = np.arange(n, dtype=float)
    X = _design_matrix(t, n, periods)
    ridge = alpha * np.eye(X.shape[1])
    ridge[0, 0] = 0.0                                   # never penalise the intercept
    beta = np.linalg.solve(X.T @ X + ridge, X.T @ y)

    tf = np.arange(n, n + horizon, dtype=float)
    return _design_matrix(tf, n, periods) @ beta


def holt_winters(y: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Additive Holt-Winters with a small grid search over the smoothing weights."""
    n = len(y)
    if n < 2 * season + 2:
        season = 1

    def run(alpha, beta, gamma):
        if season > 1:
            seasonal = list(y[:season] - y[:season].mean())
            level = float(y[:season].mean())
        else:
            seasonal = [0.0]
            level = float(y[0])
        trend = float((y[season] - y[0]) / season) if n > season else 0.0
        sse, fitted = 0.0, []
        for i in range(n):
            s_idx = i % max(season, 1)
            pred = level + trend + seasonal[s_idx]
            err = y[i] - pred
            sse += err * err
            fitted.append(pred)
            last_level = level
            level = alpha * (y[i] - seasonal[s_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            if season > 1:
                seasonal[s_idx] = gamma * (y[i] - level) + (1 - gamma) * seasonal[s_idx]
        return sse, level, trend, list(seasonal)

    best = None
    for alpha in (0.1, 0.3, 0.5, 0.8):
        for beta in (0.0, 0.05, 0.2):
            for gamma in ((0.0, 0.2, 0.5) if season > 1 else (0.0,)):
                sse, level, trend, seasonal = run(alpha, beta, gamma)
                if best is None or sse < best[0]:
                    best = (sse, level, trend, seasonal)

    _, level, trend, seasonal = best
    return np.array([level + (h + 1) * trend + seasonal[(n + h) % max(season, 1)]
                     for h in range(horizon)])


def seasonal_naive(y: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last full season. A strong baseline any model must beat."""
    season = season if len(y) >= season else 1
    tail = y[-season:]
    return np.array([tail[i % season] for i in range(horizon)], dtype=float)


def moving_average(y: np.ndarray, horizon: int, window: int = 14) -> np.ndarray:
    window = min(window, len(y))
    return np.full(horizon, float(y[-window:].mean()))


MODELS = {
    "ridge": ridge_seasonal,
    "holt_winters": holt_winters,
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
}

MODEL_LABELS = {
    "ridge": "Ridge + Fourier seasonality",
    "holt_winters": "Holt-Winters (triple exponential smoothing)",
    "seasonal_naive": "Seasonal naive",
    "moving_average": "Moving average",
    "ensemble": "Weighted ensemble",
}


def _run_model(name: str, y: np.ndarray, horizon: int, frequency: str, season: int) -> np.ndarray:
    if name == "ridge":
        out = ridge_seasonal(y, horizon, frequency)
    elif name == "holt_winters":
        out = holt_winters(y, horizon, season)
    elif name == "seasonal_naive":
        out = seasonal_naive(y, horizon, season)
    else:
        out = moving_average(y, horizon)
    return np.clip(out, 0, None)                       # demand is never negative


# --------------------------------------------------------------------------
# Step 5 - accuracy metrics and backtesting
# --------------------------------------------------------------------------

def metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(actual) < 1e-9, np.nan, np.abs(actual))
    mape = float(np.nanmean(np.abs(err) / denom) * 100) if not np.all(np.isnan(denom)) else float("nan")
    smape_denom = (np.abs(actual) + np.abs(pred)) / 2
    smape = float(np.mean(np.abs(err) / np.where(smape_denom < 1e-9, 1, smape_denom)) * 100)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2) if not math.isnan(mape) else None,
        "smape": round(smape, 2),
        "accuracy_pct": round(max(0.0, 100.0 - smape), 2),   # sMAPE is bounded, so this is too
        "bias": round(float(np.mean(err)), 2),
    }


def backtest(y: np.ndarray, model: str, horizon: int, frequency: str,
             season: int, folds: int = 3) -> Dict[str, Any]:
    """Rolling-origin evaluation: train on the past, score on the held-out future."""
    n = len(y)
    h = max(1, min(horizon, max(n // 5, 1)))
    min_train = max(10, min(2 * season, n // 2))      # long seasons must not starve the folds
    results, usable = [], 0
    for f in range(folds, 0, -1):
        cut = n - f * h
        if cut < min_train:
            continue
        pred = _run_model(model, y[:cut], h, frequency, season)
        results.append(metrics(y[cut:cut + h], pred))
        usable += 1
    if not usable:
        return {"folds": 0, "accuracy_pct": None, "smape": None, "rmse": None, "mae": None}
    agg = {k: round(float(np.mean([r[k] for r in results])), 2)
           for k in ("mae", "rmse", "smape", "accuracy_pct", "bias")}
    agg["folds"] = usable
    return agg


def select_model(y: np.ndarray, horizon: int, frequency: str, season: int) -> Tuple[str, Dict[str, Any]]:
    """Pick whichever model wins the backtest; ties go to the simpler model."""
    scores = {}
    for name in MODELS:
        bt = backtest(y, name, horizon, frequency, season)
        if bt.get("smape") is not None:
            scores[name] = bt
    if not scores:
        return "moving_average", {"folds": 0, "accuracy_pct": None}
    best = min(scores, key=lambda k: scores[k]["smape"])
    scores[best]["compared"] = {k: v["smape"] for k, v in scores.items()}
    return best, scores[best]


# --------------------------------------------------------------------------
# Step 6 - inventory policy
# --------------------------------------------------------------------------

def inventory_recommendation(
    forecast: np.ndarray,
    resid_std: float,
    current_stock: float,
    lead_time: int,
    review_period: int,
    confidence: float,
    unit_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Classic (R, S) periodic-review policy driven by the forecast.

    safety stock = z * sigma * sqrt(lead time + review period)
    order-up-to  = expected demand over that window + safety stock
    """
    z = z_score(confidence)
    protection = lead_time + review_period
    demand_lt = float(np.sum(forecast[:min(lead_time, len(forecast))]))
    demand_protection = float(np.sum(forecast[:min(protection, len(forecast))]))
    avg_daily = float(np.mean(forecast)) if len(forecast) else 0.0

    safety_stock = z * resid_std * math.sqrt(max(protection, 1))
    reorder_point = demand_lt + z * resid_std * math.sqrt(max(lead_time, 1))
    order_up_to = demand_protection + safety_stock
    suggested_order = max(0.0, order_up_to - current_stock)
    days_cover = current_stock / avg_daily if avg_daily > 1e-9 else float("inf")

    if current_stock < demand_lt:
        status, action, risk = "Stockout risk", "Restock urgently", "high"
    elif current_stock < reorder_point:
        status, action, risk = "Low stock", "Restock", "medium"
    elif current_stock > order_up_to * 1.6:
        status, action, risk = "Overstock", "Reduce / pause ordering", "medium"
    else:
        status, action, risk = "Optimal", "Maintain", "low"

    excess = max(0.0, current_stock - order_up_to)
    rec = {
        "current_stock": round(current_stock, 1),
        "avg_daily_demand": round(avg_daily, 2),
        "demand_over_lead_time": round(demand_lt, 1),
        "safety_stock": round(safety_stock, 1),
        "reorder_point": round(reorder_point, 1),
        "order_up_to_level": round(order_up_to, 1),
        "suggested_order_qty": int(round(suggested_order)),
        "excess_units": int(round(excess)),
        "days_of_cover": round(days_cover, 1) if math.isfinite(days_cover) else None,
        "stock_status": status,
        "action": action,
        "risk_level": risk,
        "service_level_pct": round(confidence * 100, 1),
    }
    if unit_price:
        rec["tied_up_capital"] = round(excess * unit_price, 2)
        rec["order_value"] = round(suggested_order * unit_price, 2)
    return rec


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

class DemandForecaster:
    """End-to-end agent: give it a dataframe, get forecasts and stock advice."""

    def __init__(self, df: pd.DataFrame, config: Optional[ForecastConfig] = None,
                 mapping: Optional[Dict[str, Optional[str]]] = None):
        self.config = config or ForecastConfig()
        self.raw = df
        self.mapping = mapping or detect_columns(df)
        self.series, self.prep_report = prepare_timeseries(df, self.mapping, self.config.frequency)
        self._static = self._collect_static_attributes()

    # -- per-product static info (stock, price, lead time) -----------------
    def _collect_static_attributes(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        prod_col = self.mapping.get("product")
        for key, col in (("stock", self.mapping.get("stock")),
                         ("price", self.mapping.get("price")),
                         ("lead_time", self.mapping.get("lead_time"))):
            if not col:
                continue
            vals = pd.to_numeric(self.raw[col], errors="coerce")
            if prod_col:
                grouped = vals.groupby(self.raw[prod_col].astype(str).str.strip()).last()
            else:
                grouped = pd.Series({"All products": vals.dropna().iloc[-1] if vals.notna().any() else np.nan})
            for product, v in grouped.items():
                if pd.notna(v):
                    out.setdefault(str(product), {})[key] = float(v)
        return out

    @property
    def products(self) -> List[str]:
        return sorted(self.series["product"].unique().tolist())

    def get_series(self, product: str) -> pd.Series:
        part = self.series[self.series["product"] == product]
        return pd.Series(part["quantity"].values, index=pd.DatetimeIndex(part["date"]))

    # -- the main call ----------------------------------------------------
    def analyze_product(self, product: str) -> Dict[str, Any]:
        cfg = self.config
        s = self.get_series(product)
        y_raw = s.values.astype(float)
        dates = s.index
        season = FREQ_PERIODS[cfg.frequency]

        if len(y_raw) < 8:
            raise ValueError(f"'{product}' has only {len(y_raw)} periods; need at least 8 to forecast.")

        # 1. anomalies
        anom = detect_anomalies(y_raw, cfg.anomaly_threshold)
        y = anom["cleaned"] if cfg.clean_anomalies else y_raw

        # 2. structure
        trend = detect_trend(y, lookback=min(len(y), TREND_LOOKBACK[cfg.frequency]))
        seasonality = detect_seasonality(y, cfg.frequency)
        if seasonality["detected"] and seasonality["period"]:
            season = seasonality["period"] if seasonality["period"] <= len(y) // 2 else season

        # 3. model choice + fit
        if cfg.model in ("auto", "ensemble"):
            chosen, bt = select_model(y, cfg.horizon, cfg.frequency, season)
        else:
            chosen = cfg.model if cfg.model in MODELS else "ridge"
            bt = backtest(y, chosen, cfg.horizon, cfg.frequency, season)

        if cfg.model == "ensemble":
            preds, weights = [], []
            for name in MODELS:
                b = backtest(y, name, cfg.horizon, cfg.frequency, season)
                if b.get("smape") is None:
                    continue
                preds.append(_run_model(name, y, cfg.horizon, cfg.frequency, season))
                weights.append(1.0 / max(b["smape"], 1e-6))
            forecast = (np.average(preds, axis=0, weights=weights) if preds
                        else _run_model("ridge", y, cfg.horizon, cfg.frequency, season))
            model_name = "ensemble"
        else:
            forecast = _run_model(chosen, y, cfg.horizon, cfg.frequency, season)
            model_name = chosen

        forecast = np.clip(forecast, 0, None)

        # 4. in-sample residual spread -> prediction intervals that widen with horizon
        fitted_hist = _run_model(model_name if model_name in MODELS else "ridge",
                                 y[:-season] if len(y) > 2 * season else y,
                                 season if len(y) > 2 * season else 1,
                                 cfg.frequency, season)
        ref = y[-len(fitted_hist):]
        resid_std = float(np.std(ref - fitted_hist)) or float(np.std(y)) or 1.0
        z = z_score(cfg.confidence)
        widen = np.sqrt(1.0 + np.arange(cfg.horizon) / max(len(y), 1))
        lower = np.clip(forecast - z * resid_std * widen, 0, None)
        upper = forecast + z * resid_std * widen

        # 5. inventory
        stat = self._static.get(product, {})
        stock_estimated = "stock" not in stat
        current_stock = stat.get("stock", float(np.mean(y[-min(len(y), 14):]) * cfg.lead_time_days * 1.2))
        lead_time = int(stat.get("lead_time", cfg.lead_time_days))
        inventory = inventory_recommendation(
            forecast, resid_std, current_stock, lead_time,
            cfg.review_period_days, cfg.confidence, stat.get("price"),
        )
        inventory["stock_source"] = "estimated from recent demand" if stock_estimated else "from file"

        hist_avg = float(np.mean(y[-min(len(y), cfg.horizon):]))
        fc_avg = float(np.mean(forecast))
        change_pct = (fc_avg - hist_avg) / hist_avg * 100 if hist_avg > 1e-9 else 0.0

        future_dates = pd.date_range(dates[-1], periods=cfg.horizon + 1, freq=cfg.frequency)[1:]

        return {
            "product": product,
            "model": model_name,
            "model_label": MODEL_LABELS.get(model_name, model_name),
            "periods_analyzed": int(len(y)),
            "history": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "actual": [round(float(v), 2) for v in y_raw],
                "cleaned": [round(float(v), 2) for v in y],
            },
            "forecast": {
                "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "values": [round(float(v), 2) for v in forecast],
                "lower": [round(float(v), 2) for v in lower],
                "upper": [round(float(v), 2) for v in upper],
                "total_units": int(round(float(forecast.sum()))),
                "avg_per_period": round(fc_avg, 2),
                "change_vs_history_pct": round(change_pct, 1),
                "change_units": int(round(fc_avg * cfg.horizon - hist_avg * cfg.horizon)),
                "confidence_pct": round(cfg.confidence * 100, 1),
            },
            "trend": trend,
            "seasonality": {k: v for k, v in seasonality.items() if k != "candidates"},
            "seasonality_detail": seasonality["candidates"],
            "anomalies": {k: v for k, v in anom.items() if k != "cleaned"},
            "accuracy": bt,
            "inventory": inventory,
            "history_stats": {
                "total_units": int(round(float(y_raw.sum()))),
                "avg": round(float(np.mean(y_raw)), 2),
                "std": round(float(np.std(y_raw)), 2),
                "min": round(float(np.min(y_raw)), 2),
                "max": round(float(np.max(y_raw)), 2),
                "zero_periods": int((y_raw == 0).sum()),
            },
        }

    def analyze_all(self, limit: Optional[int] = None) -> Dict[str, Any]:
        results, errors = [], []
        for product in self.products[: (limit or self.config.max_products)]:
            try:
                results.append(self.analyze_product(product))
            except Exception as exc:                      # one bad SKU must not kill the run
                errors.append({"product": product, "error": str(exc)})
        return {"results": results, "errors": errors, "preparation": self.prep_report}


# --------------------------------------------------------------------------
# Portfolio roll-up used by the dashboard
# --------------------------------------------------------------------------

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    acc = [r["accuracy"]["accuracy_pct"] for r in results if r["accuracy"].get("accuracy_pct") is not None]
    anomalies = sum(r["anomalies"]["count"] for r in results)
    high_sev = sum(1 for r in results for i in r["anomalies"]["items"] if i["severity"] == "high")
    restock = [r for r in results if r["inventory"]["action"].startswith("Restock")]
    reduce_ = [r for r in results if r["inventory"]["action"].startswith("Reduce")]
    urgent = [r for r in results if r["inventory"]["risk_level"] == "high"]

    excess_units = sum(r["inventory"]["excess_units"] for r in results)
    tied_capital = sum(r["inventory"].get("tied_up_capital", 0) or 0 for r in results)
    seasonal = [r for r in results if r["seasonality"]["detected"]]

    alerts = []
    if urgent:
        alerts.append({
            "level": "critical", "icon": "⚠️",
            "title": f"{len(urgent)} product(s) need restocking",
            "detail": "Projected demand over the lead time exceeds stock on hand: "
                      + ", ".join(r["product"] for r in urgent[:3])
                      + ("…" if len(urgent) > 3 else ""),
        })
    if reduce_:
        alerts.append({
            "level": "success", "icon": "✅",
            "title": "Overstock can be released",
            "detail": f"{excess_units:,} units above the order-up-to level across "
                      f"{len(reduce_)} product(s)."
                      + (f" About ₹{tied_capital:,.0f} of working capital." if tied_capital else ""),
        })
    if seasonal:
        labels = {}
        for r in seasonal:
            labels[r["seasonality"]["label"]] = labels.get(r["seasonality"]["label"], 0) + 1
        top = max(labels, key=labels.get)
        alerts.append({
            "level": "info", "icon": "💡",
            "title": f"{top.capitalize()} seasonality detected",
            "detail": f"{labels[top]} product(s) repeat on a {top} cycle — align promotions "
                      f"and replenishment to that rhythm.",
        })
    if anomalies:
        alerts.append({
            "level": "info", "icon": "🔎",
            "title": f"{anomalies} anomalies handled",
            "detail": f"{high_sev} were severe enough to review. Outliers were smoothed before "
                      f"training so one promotion does not distort the baseline.",
        })

    return {
        "products_analyzed": len(results),
        "avg_accuracy_pct": round(float(np.mean(acc)), 1) if acc else None,
        "anomalies_detected": anomalies,
        "anomalies_high_severity": high_sev,
        "restock_count": len(restock),
        "reduce_count": len(reduce_),
        "urgent_count": len(urgent),
        "total_forecast_units": int(sum(r["forecast"]["total_units"] for r in results)),
        "excess_units": excess_units,
        "tied_up_capital": round(tied_capital, 2) if tied_capital else None,
        "seasonal_products": len(seasonal),
        "alerts": alerts,
    }


def to_table(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flat rows for the dashboard table and the CSV export."""
    rows = []
    for r in results:
        inv, fc, tr = r["inventory"], r["forecast"], r["trend"]
        rows.append({
            "product": r["product"],
            "forecast_units": fc["total_units"],
            "change_units": fc["change_units"],
            "change_pct": fc["change_vs_history_pct"],
            "trend": tr["direction"],
            "trend_strength": tr["strength"],
            "seasonality": r["seasonality"]["label"] or "none",
            "anomalies": r["anomalies"]["count"],
            "accuracy_pct": r["accuracy"].get("accuracy_pct"),
            "model": r["model_label"],
            "current_stock": inv["current_stock"],
            "reorder_point": inv["reorder_point"],
            "safety_stock": inv["safety_stock"],
            "suggested_order_qty": inv["suggested_order_qty"],
            "days_of_cover": inv["days_of_cover"],
            "stock_status": inv["stock_status"],
            "action": inv["action"],
            "risk_level": inv["risk_level"],
        })
    rows.sort(key=lambda r: abs(r["change_units"]), reverse=True)
    return rows
