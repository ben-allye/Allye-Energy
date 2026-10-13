"""Analyse SCE-* BESS usage: capacity/utilization, availability/uptime, discharge-cycle
detection, AC-output "session" counting (proxy for plant/vehicle loads served per cycle),
and location. Reads parquet from data/, prints a structured findings report.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ACTIVE_MODES = {"OnGrid", "OffGrid","Active", "ActiveDcRecharge"}
IDLE_MODES = { "LowPowerMode", "Standby"}
DOWN_MODES = {"Shutdown"}

SOC_SWING_THRESHOLD = 5.0   # % SoC change to count as a new charge/discharge leg
LOAD_ON_THRESHOLD_KW = 1.0  # AC output power (kW) above this = "session" active
SESSION_GAP_MIN = 5          # bridge gaps shorter than this many minutes
MAX_KWH_PER_MIN = 10.0       # cap on plausible per-minute energy (fleet peak_load max ~467kW)
GAP_TRUST_MIN = 5            # don't let the per-minute cap scale up past this many minutes of
                              # gap - a long comms blackout followed by one big counter jump
                              # isn't "N minutes at max plausible rate", it's an unknown amount
                              # of energy over an unknown path; treat it like the gap_affected
                              # legs in detect_soc_cycles rather than trusting it at face value


AVG_LOAD_SANE_MAX = 1000.0  # avg_load has sporadic sensor-glitch outliers in the millions


def mask_zero_dropouts(s):
    """soc_avg/total_kwh_in/out never legitimately read exactly 0.0 in this fleet (SoC's
    own 5th-percentile floor is 7-14%) - it's a telemetry dropout defaulting to 0, in runs
    of a few minutes. Mask and forward-fill rather than leaving the sensor glitch in."""
    return s.mask(s == 0.0).ffill().bfill()


def load_unit_metrics():
    df = pd.read_parquet(DATA_DIR / "unit_metrics_1min.parquet")
    df = df.sort_values(["endpoint_id", "ts_min"]).reset_index(drop=True)
    df.loc[df["avg_load"].abs() > AVG_LOAD_SANE_MAX, "avg_load"] = np.nan
    for col in ["soc_avg", "total_kwh_in", "total_kwh_out"]:
        df[col] = df.groupby("endpoint_id", group_keys=False)[col].apply(mask_zero_dropouts)
    return df


def load_energy_meter_out():
    df = pd.read_parquet(DATA_DIR / "energy_meter_1min.parquet")
    df = df[df["meter"] == "energyMeterOut_derived"]
    return df.sort_values(["endpoint_id", "ts_min"]).reset_index(drop=True)


def load_location():
    df = pd.read_parquet(DATA_DIR / "location_5min.parquet")
    df = df[(df["lat"] != 0) | (df["lon"] != 0)]
    return df.sort_values(["endpoint_id", "timestamp"]).reset_index(drop=True)


def load_pack_daily():
    return pd.read_parquet(DATA_DIR / "pack_metrics_daily.parquet")


PACK_CURRENT_SANE_MAX = 450.0  # fleet-wide 99th percentile is ~15A; anything past this is
                                # the same telemetry-spike corruption seen in avg_load/kwh


def load_pack_1min():
    df = pd.read_parquet(DATA_DIR / "pack_metrics_1min.parquet")
    df = df.sort_values(["endpoint_id", "pack_number", "ts_min"]).reset_index(drop=True)
    df.loc[df["current_avg"].abs() > PACK_CURRENT_SANE_MAX, "current_avg"] = np.nan
    # same zero-defaults-when-idle telemetry glitch as celltempmax_avg at daily resolution -
    # confirmed here too: current_avg is ~0 whenever celltempmax_avg reads exactly 0.0.
    for col in ["celltempmax_avg", "celltempmax_max"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    # 5.8M rows - float32 halves the footprint of this (and every frame derived from it)
    # with no real precision loss on sensor readings quantized to 0.5A/V/%/degC anyway.
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def clean_cumulative_diff(g, col):
    """Positive-increment-only diff of a cumulative counter column (total_kwh_out/in etc.),
    capped per-minute at MAX_KWH_PER_MIN and gap-limited at GAP_TRUST_MIN so a diff spanning
    a long comms blackout can't pass through as a single huge "plausible" jump."""
    dt_min = g["ts_min"].diff().dt.total_seconds() / 60.0
    cap = MAX_KWH_PER_MIN * dt_min.clip(lower=1, upper=GAP_TRUST_MIN)
    return g[col].diff().clip(lower=0).clip(upper=cap).fillna(0)


# ---------------------------------------------------------------- utilization

def utilization_summary(u):
    rows = []
    for ep, g in u.groupby("endpoint_id"):
        peak_ceiling = g["peak_load"].max()
        near_ceiling_frac = (g["avg_load"] > 0.8 * peak_ceiling).mean()
        rows.append({
            "endpoint_id": ep,
            "avg_load_mean": g["avg_load"].mean(),
            "avg_load_p90": g["avg_load"].quantile(0.9),
            "peak_load_max": peak_ceiling,
            "peak_load_p90": g["peak_load"].quantile(0.9),
            "avg_over_peak_ratio": g["avg_load"].mean() / peak_ceiling if peak_ceiling else np.nan,
            "frac_time_over_80pct_own_peak": near_ceiling_frac,
            "soc_mean": g["soc_avg"].mean(),
            "soc_p05": g["soc_avg"].quantile(0.05),
            "soc_p95": g["soc_avg"].quantile(0.95),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ cycle detection

def detect_soc_cycles(u, endpoint_id):
    """Hysteresis peak/trough segmentation on soc_avg -> list of legs (charge/discharge)."""
    g = u[u["endpoint_id"] == endpoint_id].dropna(subset=["soc_avg"])
    if g.empty:
        return pd.DataFrame()
    ts = g["ts_min"].to_numpy()
    soc = g["soc_avg"].to_numpy()

    legs = []
    ext_idx = 0
    ext_val = soc[0]
    direction = None  # 'up' / 'down'
    for i in range(1, len(soc)):
        if direction in (None, "up"):
            if soc[i] >= ext_val:
                ext_val, ext_idx = soc[i], i
            elif ext_val - soc[i] >= SOC_SWING_THRESHOLD:
                legs.append((ts[ext_idx], ext_val, ts[i], soc[i], "discharge", i - ext_idx))
                direction, ext_val, ext_idx = "down", soc[i], i
        if direction in (None, "down"):
            if soc[i] <= ext_val:
                ext_val, ext_idx = soc[i], i
            elif soc[i] - ext_val >= SOC_SWING_THRESHOLD:
                legs.append((ts[ext_idx], ext_val, ts[i], soc[i], "charge", i - ext_idx))
                direction, ext_val, ext_idx = "up", soc[i], i

    legs_df = pd.DataFrame(
        legs, columns=["start_ts", "start_soc", "end_ts", "end_soc", "kind", "n_samples"]
    )
    legs_df["depth"] = (legs_df["end_soc"] - legs_df["start_soc"]).abs()
    legs_df["duration_h"] = (legs_df["end_ts"] - legs_df["start_ts"]) / np.timedelta64(1, "h")
    # a leg built from a couple of samples either side of a long reporting gap isn't a real
    # continuous charge/discharge - it's a comms blackout with an unknown SOC path in between.
    expected_samples = legs_df["duration_h"] * 60
    legs_df["coverage"] = legs_df["n_samples"] / expected_samples.clip(lower=1)
    legs_df["gap_affected"] = legs_df["coverage"] < 0.5
    legs_df["endpoint_id"] = endpoint_id
    return legs_df


# -------------------------------------------------------- AC-output sessions

def detect_ac_sessions(meter_df, endpoint_id):
    g = meter_df[meter_df["endpoint_id"] == endpoint_id].dropna(subset=["powertotal"])
    if g.empty:
        return pd.DataFrame()
    g = g.set_index("ts_min").sort_index()
    active = g["powertotal"].abs() > LOAD_ON_THRESHOLD_KW
    active = active.reindex(
        pd.date_range(g.index.min(), g.index.max(), freq="1min"), fill_value=False
    )
    # bridge short gaps: a False run no longer than SESSION_GAP_MIN gets flipped to True
    run_id = (active != active.shift()).cumsum()
    run_len = active.groupby(run_id).transform("size")
    bridged = active | (~active & (run_len <= SESSION_GAP_MIN))
    session_id = (bridged != bridged.shift(fill_value=False)).cumsum()
    sessions = (
        pd.DataFrame({"active": bridged, "session_id": session_id})
        .reset_index(names="ts")
        .groupby("session_id")
        .agg(start=("ts", "min"), end=("ts", "max"), active=("active", "first"))
    )
    sessions = sessions[sessions["active"]].copy()
    sessions["duration_h"] = (sessions["end"] - sessions["start"]) / np.timedelta64(1, "h") + (1 / 60)
    sessions["endpoint_id"] = endpoint_id
    return sessions.reset_index(drop=True)


# -------------------------------------------------------- recharge cycles

FULL_SOC_THRESHOLD = 85.0   # SoC% counted as "fully recharged"
FULL_SOC_REARM_BELOW = 50.0  # must drop back below this before the next full-charge re-triggers


def recharge_cycles(u, endpoint_id):
    """A cycle runs from one full-recharge event to the next. 'Full-recharge event' = SoC
    crossing back above FULL_SOC_THRESHOLD, hysteresis-gated so noise near the threshold
    can't retrigger it without first dropping meaningfully lower."""
    g = u[u["endpoint_id"] == endpoint_id].dropna(subset=["soc_avg"]).sort_values("ts_min")
    ts, soc = g["ts_min"].to_numpy(), g["soc_avg"].to_numpy()
    events, armed = [], True
    for i in range(len(soc)):
        if soc[i] < FULL_SOC_REARM_BELOW:
            armed = True
        elif armed and soc[i] >= FULL_SOC_THRESHOLD:
            events.append(ts[i])
            armed = False
    if len(events) < 2:
        return pd.DataFrame(columns=["start", "end", "duration_h"])
    events = np.array(events)
    cycles = pd.DataFrame({"start": events[:-1], "end": events[1:]})
    cycles["duration_h"] = (cycles["end"] - cycles["start"]) / np.timedelta64(1, "h")
    return cycles


def cycle_details(u, endpoint_id):
    """recharge_cycles, with each cycle's actual depth (SoC swing below the 85% start)."""
    g = u[u["endpoint_id"] == endpoint_id].dropna(subset=["soc_avg"]).sort_values("ts_min")
    cycles = recharge_cycles(u, endpoint_id)
    mins = []
    for _, row in cycles.iterrows():
        w = g[(g["ts_min"] >= row["start"]) & (g["ts_min"] < row["end"])]
        mins.append(w["soc_avg"].min() if len(w) else np.nan)
    cycles["min_soc"] = mins
    cycles["depth"] = FULL_SOC_THRESHOLD - cycles["min_soc"]
    cycles["endpoint_id"] = endpoint_id
    return cycles


def session_membership(ts, sessions):
    """Boolean per-timestamp: does an AC-output session cover this minute?"""
    if sessions.empty:
        return np.zeros(len(ts), dtype=bool)
    intervals = pd.IntervalIndex.from_arrays(sessions["start"], sessions["end"], closed="both")
    return intervals.get_indexer(pd.DatetimeIndex(ts)) >= 0


# ------------------------------------------------------------------ uptime

def uptime_summary(u):
    rows = []
    for ep, g in u.groupby("endpoint_id"):
        g = g.sort_values("ts_min")
        span_min = (g["ts_min"].max() - g["ts_min"].min()) / np.timedelta64(1, "m") + 1
        reporting_frac = len(g) / span_min
        mode_frac = g["status_mode"].value_counts(normalize=True, dropna=False)
        active_frac = mode_frac.reindex(ACTIVE_MODES, fill_value=0).sum()
        idle_frac = mode_frac.reindex(IDLE_MODES, fill_value=0).sum()
        down_frac = mode_frac.reindex(DOWN_MODES, fill_value=0).sum()
        rt_span = g["total_runtime"].max() - g["total_runtime"].min()
        op_span = g["operational_runtime"].max() - g["operational_runtime"].min()
        rows.append({
            "endpoint_id": ep,
            "reporting_frac_of_span": reporting_frac,
            "active_frac": active_frac,
            "idle_frac": idle_frac,
            "down_frac": down_frac,
            "operational_over_total_runtime": op_span / rt_span if rt_span else np.nan,
            "first_seen": g["ts_min"].min(),
            "last_seen": g["ts_min"].max(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- location

def location_summary(loc):
    rows = []
    for ep, g in loc.groupby("endpoint_id"):
        g = g.copy()
        g["cluster"] = (g["lat"].round(3).astype(str) + "," + g["lon"].round(3).astype(str))
        clusters = g.groupby("cluster").agg(
            lat=("lat", "mean"), lon=("lon", "mean"),
            first_seen=("timestamp", "min"), last_seen=("timestamp", "max"),
            n=("timestamp", "size"),
        ).sort_values("first_seen")
        rows.append({"endpoint_id": ep, "n_distinct_locations": len(clusters), "clusters": clusters})
    return rows


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    u = load_unit_metrics()
    meter = load_energy_meter_out()
    loc = load_location()
    pack = load_pack_daily()

    print("\n================ UTILIZATION ================")
    print(utilization_summary(u).to_string(index=False))

    print("\n================ UPTIME / AVAILABILITY ================")
    print(uptime_summary(u).to_string(index=False))

    print("\n================ SOC CYCLES (per endpoint) ================")
    print("(gap_affected legs = spans a reporting blackout >= half the leg's minutes missing;")
    print(" these are comms outages, not real continuous charge/discharge, and are excluded")
    print(" from the depth/duration stats below - see the blackout line instead)")
    all_legs = {}
    for ep in u["endpoint_id"].unique():
        legs = detect_soc_cycles(u, ep)
        all_legs[ep] = legs
        clean = legs[~legs["gap_affected"]]
        disch = clean[clean["kind"] == "discharge"]
        chg = clean[clean["kind"] == "charge"]
        gaps = legs[legs["gap_affected"]]
        print(f"\n-- {ep} --")
        print(f"discharge legs: n={len(disch)}, depth mean={disch['depth'].mean():.1f}%, "
              f"duration mean={disch['duration_h'].mean():.2f}h")
        print(f"charge legs:    n={len(chg)}, depth mean={chg['depth'].mean():.1f}%, "
              f"duration mean={chg['duration_h'].mean():.2f}h")
        print(f"blackout-spanning legs: n={len(gaps)}, total hours={gaps['duration_h'].sum():.1f}, "
              f"mean depth={gaps['depth'].mean():.1f}%")

    print("\n================ AC-OUTPUT SESSIONS (per endpoint) ================")
    all_sessions = {}
    for ep in u["endpoint_id"].unique():
        sess = detect_ac_sessions(meter, ep)
        all_sessions[ep] = sess
        print(f"\n-- {ep} --")
        print(f"sessions: n={len(sess)}, duration mean={sess['duration_h'].mean():.2f}h, "
              f"median={sess['duration_h'].median():.2f}h")

    print("\n================ RECHARGE CYCLES: SESSIONS SERVED & SELF-CONSUMPTION ================")
    print(f"(cycle = one full recharge (SoC>={FULL_SOC_THRESHOLD:.0f}%) to the next. energy split")
    print(" uses every minute's plausible kwh_out increment, classified by whether an AC-output")
    print(" session (energy-meter powertotal > 1kW) covers that minute - not tied to leg boundaries.)")
    for ep in u["endpoint_id"].unique():
        g = u[u["endpoint_id"] == ep].sort_values("ts_min").reset_index(drop=True)
        sess = all_sessions[ep]

        clean_diff = clean_cumulative_diff(g, "total_kwh_out")
        in_session = session_membership(g["ts_min"], sess)

        cycles = recharge_cycles(u, ep)
        ts_arr = g["ts_min"].to_numpy()
        counts = []
        for _, row in cycles.iterrows():
            n = ((sess["start"] < row["end"]) & (sess["end"] > row["start"])).sum()
            counts.append(n)
        cycles["n_sessions"] = counts

        print(f"\n-- {ep} --")
        print(f"recharge cycles: n={len(cycles)}, duration mean={cycles['duration_h'].mean():.1f}h, "
              f"median={cycles['duration_h'].median():.1f}h")
        if len(cycles):
            print(f"AC-output sessions per cycle: mean={cycles['n_sessions'].mean():.1f}, "
                  f"median={cycles['n_sessions'].median():.1f}, max={cycles['n_sessions'].max()}")
        energy_in = clean_diff[in_session].sum()
        energy_out = clean_diff[~in_session].sum()
        total = energy_in + energy_out
        print(f"energy while a session is active:    {energy_in:.0f} kWh ({100*energy_in/total:.0f}%)")
        print(f"energy while NO session is active:   {energy_out:.0f} kWh ({100*energy_out/total:.0f}%) "
              f"-> standby/self-consumption/unmetered draw")

    print("\n================ LOCATION ================")
    for row in location_summary(loc):
        print(f"\n-- {row['endpoint_id']} -- distinct locations: {row['n_distinct_locations']}")
        print(row["clusters"].to_string())

    print("\n================ PACK FAULT SUMMARY (secondary) ================")
    fault_summary = pack.groupby("endpoint_id")[
        ["error_faults_delta", "alarm_faults_delta", "critical_faults_delta"]
    ].sum()
    print(fault_summary.to_string())
