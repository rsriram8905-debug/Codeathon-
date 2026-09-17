"""Generate a realistic retail sales history so the dashboard has something to chew on.

Each SKU gets its own base level, growth trend, weekly rhythm, yearly season,
promotion spikes and stockout dips - i.e. all the patterns the engine claims to handle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SKUS = [
    # name,                   base, growth/yr, weekly amp, yearly amp, price, lead time
    ("A-101 Rice 5kg",         120,  0.35, 0.30, 0.10,  420,  7),
    ("B-205 Cooking Oil 1L",    85,  0.05, 0.22, 0.15,  180,  5),
    ("C-312 Winter Jacket",     40, -0.25, 0.15, 0.85, 1899, 21),
    ("D-418 Bluetooth Buds",    65,  0.55, 0.35, 0.30, 2499, 14),
    ("E-521 Detergent 2kg",     95,  0.02, 0.18, 0.08,  320,  7),
    ("F-604 Sunscreen SPF50",   30,  0.18, 0.12, 0.70,  549, 10),
]


def generate_sales_data(days: int = 730, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    t = np.arange(len(dates))
    rows = []

    for name, base, growth, w_amp, y_amp, price, lead in SKUS:
        trend = base * (1 + growth * t / 365)
        weekly = 1 + w_amp * np.sin(2 * np.pi * (dates.dayofweek.values + 1) / 7)
        yearly = 1 + y_amp * np.sin(2 * np.pi * (dates.dayofyear.values - 80) / 365)
        qty = np.clip(trend * weekly * yearly * rng.normal(1.0, 0.12, len(dates)), 0, None)

        # Promotions: short, sharp multipliers.
        for start in rng.choice(len(dates) - 4, size=max(days // 120, 2), replace=False):
            qty[start:start + 3] *= rng.uniform(2.2, 3.4)

        # Stockouts: demand recorded as near zero.
        for start in rng.choice(len(dates) - 3, size=max(days // 200, 1), replace=False):
            qty[start:start + 2] *= rng.uniform(0.0, 0.15)

        # Festive lift in late October.
        qty[(dates.month == 10) & (dates.day > 18)] *= 1.6

        qty = np.round(qty).astype(int)
        stock = int(max(np.mean(qty[-14:]) * rng.uniform(4, 30), 1))

        for d, q in zip(dates, qty):
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "product_id": name,
                "units_sold": int(q),
                "unit_price": price,
                "current_stock": stock,
                "lead_time_days": lead,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv("sample_sales_history.csv", index=False)
    print(f"Wrote sample_sales_history.csv  rows={len(df):,}  skus={df.product_id.nunique()}")
