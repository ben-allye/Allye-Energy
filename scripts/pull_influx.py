"""Pull SCE-* live-only signals from InfluxDB (skbd-bucket, 90-day retention):
DC-charger connector state (EV/plant charge sessions), vehicle-side VCU fields,
and unit location (lat/lon). 1-min windows for charger/control, 5-min for location.
Writes parquet to data/.
"""
import io
import json
import os
from pathlib import Path

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

INFLUX_HOST = os.environ["INFLUX_HOST"]
INFLUX_PORT = os.environ["INFLUX_PORT"]
INFLUX_ORG = os.environ["INFLUX_ORG"]
INFLUX_BUCKET = os.environ["INFLUX_BUCKET"]
INFLUX_SECRET_ID = os.environ["INFLUX_SECRET_ID"]
ENDPOINT_PREFIX = os.environ["ENDPOINT_PREFIX"]
RANGE_START = "-90d"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DC_CHARGER_FIELDS = [
    "ChargerAvailability", "ConnectionStatus",
    "Connector0StatusCode", "Connector0ErrorCode",
    "Connector1StatusCode", "Connector1ErrorCode",
    "Connector2StatusCode", "Connector2ErrorCode",
    "CurrentImport", "PowerActiveImport",
    "EnergyActiveImportInterval", "FaultStatus", "Temperature",
]

CONTROL_EV_FIELDS = [
    "VCU_EVSOC", "VCU_EVEnergyRequest", "VCU_EVEnergyCapacity",
    "VCU_EVMaximumCurrentLimit", "VCU_EVMaximumPowerLimit", "VCU_EVMaximumVoltageLimit",
    "VCU_VehicleVoltage", "VCU_HVReady", "VCU_ChargePermission", "StateOfCharge",
]

LOCATION_FIELDS = ["lat", "lon"]


def influx_session():
    secret = json.loads(
        boto3.client("secretsmanager").get_secret_value(SecretId=INFLUX_SECRET_ID)["SecretString"]
    )
    s = requests.Session()
    r = s.post(
        f"https://{INFLUX_HOST}:{INFLUX_PORT}/api/v2/signin",
        auth=(secret["username"], secret["password"]),
        timeout=15,
    )
    r.raise_for_status()
    return s


def parse_flux_csv(text):
    tables = []
    for block in text.split("\r\n\r\n"):
        lines = [l for l in block.splitlines() if l and not l.startswith("#")]
        if len(lines) < 2:
            continue
        tables.append(pd.read_csv(io.StringIO("\n".join(lines))))
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def query_flux(s, flux):
    r = s.post(
        f"https://{INFLUX_HOST}:{INFLUX_PORT}/api/v2/query?org={INFLUX_ORG}",
        headers={"Content-Type": "application/vnd.flux", "Accept": "text/csv"},
        data=flux,
        timeout=180,
    )
    r.raise_for_status()
    return parse_flux_csv(r.text)


METADATA_COLS = ["result", "table", "_start", "_stop", "_measurement"]


def clean(df, value_cols):
    df = df.drop(columns=[c for c in METADATA_COLS if c in df.columns])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns={"_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    for c in value_cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df.sort_values(["endpoint_id", "timestamp"]).reset_index(drop=True)


def pull_measurement(s, measurement, fields, window, name):
    field_filter = " or ".join(f'r._field == "{f}"' for f in fields)
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {RANGE_START})
  |> filter(fn: (r) => r._measurement == "{measurement}" and r.endpoint_id =~ /^{ENDPOINT_PREFIX}/)
  |> filter(fn: (r) => {field_filter})
  |> aggregateWindow(every: {window}, fn: last, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
'''
    df = clean(query_flux(s, flux), fields)
    print(f"--- {name} ---")
    print(df.shape)
    df.to_parquet(DATA_DIR / f"{name}.parquet")
    return df


if __name__ == "__main__":
    s = influx_session()
    pull_measurement(s, "dc_charger", DC_CHARGER_FIELDS, "1m", "dc_charger_1min")
    pull_measurement(s, "control", CONTROL_EV_FIELDS, "1m", "control_ev_1min")
    pull_measurement(s, "unit_status", LOCATION_FIELDS, "5m", "location_5min")
