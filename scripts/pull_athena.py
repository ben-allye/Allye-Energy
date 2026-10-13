"""Pull SCE-* BESS data from Athena (allye_historical): 1-min resampled unit/energy-meter
metrics for the full history, plus daily pack_metrics rollups. Writes parquet to data/.
Always stops at the end of yesterday (UTC). Usage: pull_athena.py [--since]  (--since = extend
existing parquets instead of re-pulling the full history).
"""
import os
import sys
import tempfile
from pathlib import Path

import boto3
import awswrangler as wr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB = os.environ["ATHENA_DATABASE"]
S3_OUTPUT = os.environ["ATHENA_S3_OUTPUT"]
PREFIX = os.environ["ENDPOINT_PREFIX"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

session = boto3.Session()


def run(sql, name, output_dir=DATA_DIR):
    print(f"--- {name} ---")
    df = wr.athena.read_sql_query(
        sql=sql, database=DB, s3_output=S3_OUTPUT, boto3_session=session, ctas_approach=False
    )
    print(df.shape)
    df.to_parquet(output_dir / f"{name}.parquet")
    return df


UNIT_METRICS_SQL = f"""
SELECT
    endpoint_id,
    date_trunc('minute', timestamp) AS ts_min,
    arbitrary(status_mode) AS status_mode,
    avg(soc_scaled) AS soc_avg,
    min(soc_scaled) AS soc_min,
    max(soc_scaled) AS soc_max,
    avg(avg_load) AS avg_load,
    max(peak_load) AS peak_load,
    avg(ac_in_pwr) AS ac_in_pwr,
    avg(ac_out_pwr) AS ac_out_pwr,
    avg(ac_in_current) AS ac_in_current,
    avg(ac_out_current) AS ac_out_current,
    max(total_kwh_in) AS total_kwh_in,
    max(total_kwh_out) AS total_kwh_out,
    max(total_runtime) AS total_runtime,
    max(operational_runtime) AS operational_runtime
FROM {DB}.unit_metrics
WHERE endpoint_id LIKE '{PREFIX}%'
  AND timestamp < current_date  -- complete days only (up to yesterday, UTC)
GROUP BY endpoint_id, date_trunc('minute', timestamp)
"""

ENERGY_METER_SQL = f"""
SELECT
    endpoint_id,
    meter,
    date_trunc('minute', timestamp) AS ts_min,
    arbitrary(status_mode) AS status_mode,
    avg(frequency) AS frequency,
    avg(powertotal) AS powertotal,
    avg(powerl1) AS powerl1,
    avg(powerl2) AS powerl2,
    avg(powerl3) AS powerl3,
    avg(currentl1) AS currentl1,
    avg(currentl2) AS currentl2,
    avg(currentl3) AS currentl3,
    avg(voltagel1) AS voltagel1,
    avg(voltagel2) AS voltagel2,
    avg(voltagel3) AS voltagel3
FROM {DB}.energy_meter_metrics
WHERE endpoint_id LIKE '{PREFIX}%'
  AND timestamp < current_date  -- complete days only (up to yesterday, UTC)
GROUP BY endpoint_id, meter, date_trunc('minute', timestamp)
"""

PACK_METRICS_SQL = f"""
SELECT
    endpoint_id,
    pack_number,
    date(timestamp) AS day,
    avg(voltage) AS voltage_avg,
    avg(current) AS current_avg,
    avg(soc) AS soc_avg,
    avg(celltempmax) AS celltempmax_avg,
    max(celltempmax) AS celltempmax_max,
    avg(celltempmin) AS celltempmin_avg,
    min(celltempmin) AS celltempmin_min,
    avg(cellvoltagemax) AS cellvoltagemax_avg,
    max(cellvoltagemax) AS cellvoltagemax_max,
    avg(cellvoltagemin) AS cellvoltagemin_avg,
    min(cellvoltagemin) AS cellvoltagemin_min,
    avg(energyfullpacknominal) AS energy_full_nominal_avg,
    avg(energyremainingnominal) AS energy_remaining_nominal_avg,
    avg(energybuffer) AS energy_buffer_avg,
    max(error_fault_flag_count) - min(error_fault_flag_count) AS error_faults_delta,
    max(alarm_fault_flag_count) - min(alarm_fault_flag_count) AS alarm_faults_delta,
    max(critical_fault_flag_count) - min(critical_fault_flag_count) AS critical_faults_delta
FROM {DB}.pack_metrics
WHERE endpoint_id LIKE '{PREFIX}%'
  AND timestamp < current_date  -- complete days only (up to yesterday, UTC)
GROUP BY endpoint_id, pack_number, date(timestamp)
"""

PACK_METRICS_1MIN_SQL = f"""
SELECT
    endpoint_id,
    pack_number,
    date_trunc('minute', timestamp) AS ts_min,
    avg(voltage) AS voltage_avg,
    avg(current) AS current_avg,
    avg(soc) AS soc_avg,
    avg(energyfullpacknominal) AS energy_full_nominal_avg,
    avg(celltempmax) AS celltempmax_avg,
    max(celltempmax) AS celltempmax_max
FROM {DB}.pack_metrics
WHERE endpoint_id LIKE '{PREFIX}%'
  AND timestamp < current_date  -- complete days only (up to yesterday, UTC)
GROUP BY endpoint_id, pack_number, date_trunc('minute', timestamp)
"""

UPPER_BOUND = "AND timestamp < current_date"
OVERLAP_DAYS = 1  # re-pull from the start of the last stored day: last bucket may be partial / late rows


def run_incremental(sql, name, keys, time_col):
    """Pull only rows from the last stored day onward, merge into the existing parquet."""
    path = DATA_DIR / f"{name}.parquet"
    old = pd.read_parquet(path)
    cutoff = pd.Timestamp(old[time_col].max()).normalize() - pd.Timedelta(days=OVERLAP_DAYS - 1)
    print(f"{name}: existing data to {old[time_col].max()}, re-pulling from {cutoff}")
    sql = sql.replace(UPPER_BOUND, f"AND timestamp >= timestamp '{cutoff:%Y-%m-%d %H:%M:%S}'\n  {UPPER_BOUND}")
    with tempfile.TemporaryDirectory(prefix="athena-incr-", dir=DATA_DIR) as staging:
        new = run(sql, name, Path(staging))
    if list(new.columns) != list(old.columns):
        raise RuntimeError(f"{name}: schema changed, run a full pull (without --since)")
    keep = old[pd.to_datetime(old[time_col]) < cutoff]
    out = pd.concat([keep, new], ignore_index=True).drop_duplicates(keys, keep="last")
    out = out.sort_values(keys).reset_index(drop=True)
    out.to_parquet(path)
    print(f"{name}: {len(old):,} -> {len(out):,} rows")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (UNIT_METRICS_SQL, "unit_metrics_1min", ["endpoint_id", "ts_min"], "ts_min"),
        (ENERGY_METER_SQL, "energy_meter_1min", ["endpoint_id", "meter", "ts_min"], "ts_min"),
        (PACK_METRICS_SQL, "pack_metrics_daily", ["endpoint_id", "pack_number", "day"], "day"),
        (PACK_METRICS_1MIN_SQL, "pack_metrics_1min", ["endpoint_id", "pack_number", "ts_min"], "ts_min"),
    ]
    if "--since" in sys.argv:  # incremental: extend existing parquets up to yesterday
        for sql, name, keys, time_col in jobs:
            run_incremental(sql, name, keys, time_col)
    else:  # full history
        with tempfile.TemporaryDirectory(prefix="athena-refresh-", dir=DATA_DIR) as staging:
            staging_dir = Path(staging)
            for sql, name, *_ in jobs:
                run(sql, name, staging_dir)
            for _, name, *_ in jobs:
                os.replace(staging_dir / f"{name}.parquet", DATA_DIR / f"{name}.parquet")
