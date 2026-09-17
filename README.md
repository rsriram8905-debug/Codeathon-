# DemandIQ — Intelligent Demand Forecasting Agent

Python backend for the DemandIQ dashboard. It takes raw retail sales history and returns
demand forecasts, trend and seasonality analysis, flagged anomalies, and stock
recommendations for every product.

## Run it locally

```bash
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:5000
```

Open the address in a browser, click **Load sample data**, then **Run forecast**.
To use your own file, drop a CSV on the upload panel.

```bash
python test_backend.py        # 11 checks covering each requirement
python sample_data.py         # writes sample_sales_history.csv
```

Configuration is via environment variables (see `.env.example`): `PORT`,
`FLASK_DEBUG`, `MAX_UPLOAD_MB`, `MAX_DATASETS`, `ALLOWED_ORIGINS`. Locally
these all have sane defaults, so plain `python app.py` works with none set.

## Deploying for real

The app is served by **gunicorn**, not Flask's dev server, whenever it isn't
run directly with `python app.py`. Two ready-made entry points are included:

- **Procfile** — for platforms that read one directly (Render, Railway, Heroku):
  ```
  web: gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
  ```
- **Dockerfile** — for anywhere that runs containers (Fly.io, Railway, ECS,
  Cloud Run, a VPS):
  ```bash
  docker build -t demandiq .
  docker run -p 5000:5000 -e PORT=5000 demandiq
  ```

### Render / Railway (no Docker needed)
1. Push this project to a GitHub repo.
2. Create a new **Web Service** from the repo.
3. Build command: `pip install -r requirements.txt`. Start command: leave it
   to auto-detect the `Procfile`, or set it explicitly to the line above.
4. Set `FLASK_DEBUG=0` (default) and any other variables from `.env.example`
   you want to override. The platform sets `PORT` for you automatically.

### Fly.io / a plain VPS (Docker)
```bash
fly launch          # or: docker build/run as above, behind nginx/caddy for TLS
```

### A note on scaling past one worker
Uploaded datasets and cached forecasts live **in memory** (see
`forecasting_engine.py` / `DATASETS` in `app.py`), scoped to a single process.
With `--workers 2` or more, or with multiple machines, a request can land on
a worker that never saw the upload. For a single small deployment this is
usually invisible; for anything with real concurrent users, either:
- run a single worker (`--workers 1`), which is fine for light traffic and
  keeps the current in-memory design, or
- swap `DATASETS`/`CACHE` for Redis or a database — the rest of the API
  contract (`app.py`) does not need to change to do this.

## Files

| File | What it holds |
| --- | --- |
| `forecasting_engine.py` | The whole pipeline: cleaning, anomalies, trend, seasonality, models, backtesting, inventory policy |
| `app.py` | Flask REST API and static hosting for the dashboard |
| `sample_data.py` | Synthetic two-year retail history with trends, seasons, promotions and stockouts |
| `test_backend.py` | Test suite |
| `demand_forecasting_frontend.html` | The dashboard, wired to the API |

## Input format

Column names are auto-detected, so `date` / `Order Date` / `sale_date` all work, as do
`units_sold` / `qty` / `demand`. Only a date column and a quantity column are required.

```csv
date,product_id,units_sold,unit_price,current_stock,lead_time_days
2025-01-01,A-101 Rice 5kg,132,420,1850,7
2025-01-02,A-101 Rice 5kg,119,420,1850,7
```

Without a `product_id` column everything is treated as one series. Without
`current_stock`, stock on hand is estimated from recent demand and the response says so.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Service status and available models |
| POST | `/api/upload` | Multipart CSV upload → `dataset_id` |
| POST | `/api/sample` | Load the built-in demo dataset |
| GET | `/api/datasets/<id>` | Row count, date range, detected columns |
| GET | `/api/products/<id>` | Products found in the dataset |
| POST | `/api/forecast` | Run the pipeline, return the dashboard payload |
| GET | `/api/product/<id>/<sku>` | Full detail for one product |
| GET | `/api/export/<id>.csv` | Recommendations as CSV |

```bash
curl -X POST localhost:5000/api/forecast -H 'Content-Type: application/json' \
  -d '{"dataset_id":"<id>","horizon":30,"confidence":0.9,"model":"auto","lead_time_days":7}'
```

Forecast request fields: `horizon` (1–365), `confidence` (0.5–0.99 or `"90%"`),
`model`, `frequency` (`D`/`W`/`M`), `lead_time_days`, `review_period_days`,
`anomaly_threshold`, `clean_anomalies`, `product`, `limit`.

## How each requirement is met

**Historical sales analysis.** Rows are parsed tolerantly — ISO, US and day-first dates,
numbers carrying commas or currency symbols — then deduplicated and aggregated per
product and period. Returns totals, averages, spread and zero-sale periods, plus a
preparation report saying exactly what was repaired.

**Time-series processing.** Every product is reindexed onto a gap-free calendar; a missing
day means no sale, not an unknown. Daily, weekly and monthly aggregation are supported.

**Demand prediction.** Four models are implemented from scratch: ridge regression on trend
plus Fourier seasonality, Holt-Winters triple exponential smoothing, seasonal naive, and a
moving average. `model: "auto"` runs a rolling-origin backtest and keeps the winner per
product; `"ensemble"` blends them weighted by inverse error. Prediction intervals come from
residual spread and widen with the horizon.

**Trend detection.** OLS slope with a t-test, so "Stable" means statistically flat rather
than eyeballed, reported with direction, percentage change and strength.

**Seasonality detection.** Autocorrelation at retail-relevant lags (7 / 14 / 30 / 91 / 365
days), confirmed by how much variance the seasonal profile explains. Reports the period,
its strength and where the peak falls.

**Anomaly handling.** Rolling-median and MAD z-scores, scaled locally so a quiet season
isn't judged by a busy one's noise. Outliers are classified as spikes (promotion, bulk
order) or drops (stockout, data gap) and winsorised onto the local level before training,
so one promotion doesn't permanently inflate the baseline.

**Inventory recommendation.** A periodic-review (R, S) policy: safety stock =
z × σ × √(lead time + review period), with reorder point, order-up-to level, suggested
order quantity, days of cover, and a status of Stockout risk / Low stock / Optimal /
Overstock. Where a price column exists, excess stock is priced as tied-up capital.

## Accuracy

Accuracy is `100 − sMAPE` measured on held-out periods across three rolling folds, never
on data the model trained on. On the sample dataset it lands around 85%, and the response
carries MAE, RMSE, sMAPE, bias and fold count alongside it so the number can be checked.

## Notes and limits

- Datasets live in memory and are evicted after 20 uploads. Swap `DATASETS` for Redis or
  a database for real deployment, and put this behind gunicorn rather than the dev server.
- A product needs at least 8 periods of history; shorter series are skipped and listed
  under `skipped` in the response.
- Confidence intervals assume roughly symmetric residuals, which holds less well for very
  low-volume, intermittent SKUs.
