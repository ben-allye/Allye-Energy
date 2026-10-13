"""Pull London ambient temperature (Open-Meteo historical archive, ERA5-based, no API
key required) for the same date range as the fleet deployment, as a comparison baseline
for the cell-temperature analysis. Writes parquet to data/.

Daily (avg/min/max) and hourly (point-in-time) - hourly is the finest resolution the
archive offers (no true per-minute ambient product exists publicly; even Open-Meteo's
own minutely_15 is forecast-only, not available on historical archive queries).
"""
from pathlib import Path

import pandas as pd
import requests

from analyze import load_unit_metrics

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LONDON_LAT, LONDON_LON = 51.5074, -0.1278
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

if __name__ == "__main__":
    u = load_unit_metrics()
    start_date = u["ts_min"].min().strftime("%Y-%m-%d")
    end_date = u["ts_min"].max().strftime("%Y-%m-%d")
    print(f"Fetching London weather {start_date} to {end_date}...")

    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": LONDON_LAT,
            "longitude": LONDON_LON,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
            "hourly": "temperature_2m",
            "timezone": "Europe/London",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    daily, hourly = payload["daily"], payload["hourly"]

    daily_df = pd.DataFrame({
        "day": pd.to_datetime(daily["time"]),
        "ambient_temp_max_c": daily["temperature_2m_max"],
        "ambient_temp_min_c": daily["temperature_2m_min"],
        "ambient_temp_mean_c": daily["temperature_2m_mean"],
    })
    print(daily_df.shape)
    daily_df.to_parquet(DATA_DIR / "london_weather_daily.parquet")

    hourly_df = pd.DataFrame({
        "ts_hour": pd.to_datetime(hourly["time"]),
        "ambient_temp_c": hourly["temperature_2m"],
    })
    print(hourly_df.shape)
    hourly_df.to_parquet(DATA_DIR / "london_weather_hourly.parquet")
