"""Sales case: are these units running at capacity, and is the fleet growing into a squeeze?
Three sections - saturation, unmet-demand proxy, growth trend - each printed separately so
weak evidence (esp. section 2, which has no direct queue/session data behind it) can be
dropped from the pitch without touching the rest.
"""
import numpy as np
import pandas as pd

from analyze import load_unit_metrics, cycle_details, utilization_summary
from summary import daily_throughput, ACTIVE_DAY_MIN_KWH

# soc_avg's own 5th-percentile floor is 7-14% fleet-wide (see analyze.py) - so "near-empty"
# is set just above that, not at some arbitrary round number.
LOW_SOC_THRESHOLD = 15.0


def soc_low_while_serving_load(g):
    """Contiguous spans where soc_avg is below LOW_SOC_THRESHOLD *and* the unit is still
    pushing AC output (ac_out_pwr > 0) - i.e. running near-empty rather than idly low."""
    low = (g["soc_avg"] < LOW_SOC_THRESHOLD) & (g["ac_out_pwr"] > 0)
    run_id = (low != low.shift(fill_value=False)).cumsum()
    spans = (
        pd.DataFrame({"ts": g["ts_min"], "low": low, "run_id": run_id})[low]
        .groupby("run_id")
        .agg(start=("ts", "min"), end=("ts", "max"), n=("ts", "size"))
    )
    return spans


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    u = load_unit_metrics()
    endpoints = sorted(u["endpoint_id"].unique())

    # ============================================================ 1. CAPACITY SATURATION
    print("\n================ 1. CAPACITY SATURATION ================")
    print("(frac_time_over_80pct_own_peak is low fleet-wide - units are NOT running up against")
    print(" their own ceiling most of the time. Weak support for a 'maxed out' argument on its own.)")

    util = utilization_summary(u)

    rows = []
    all_cycles = []
    for ep in endpoints:
        g = u[u["endpoint_id"] == ep].sort_values("ts_min")
        calendar_days = (g["ts_min"].max() - g["ts_min"].min()) / np.timedelta64(1, "D")
        daily = daily_throughput(g)
        active_days = (daily >= ACTIVE_DAY_MIN_KWH).sum()
        cycles = cycle_details(u, ep)
        all_cycles.append(cycles)
        rows.append({
            "endpoint_id": ep,
            "active_day_ratio": active_days / calendar_days,
        })
    per_unit = pd.DataFrame(rows).merge(util, on="endpoint_id")
    all_cycles = pd.concat(all_cycles, ignore_index=True)
    all_cycles["start_month"] = pd.to_datetime(all_cycles["start"]).dt.to_period("M")
    all_cycles["efc"] = all_cycles["depth"] / 100.0

    print("-- active-day ratio & time spent near own peak (little slack left if both are high) --")
    print(per_unit[[
        "endpoint_id", "active_day_ratio", "frac_time_over_80pct_own_peak", "soc_p05", "soc_p95",
    ]].round(2).to_string(index=False))

    print("\n-- avg EFC/day by month, per unit (climbing = getting worked harder over time) --")
    daily_efc = all_cycles.groupby(["endpoint_id", "start_month", pd.to_datetime(all_cycles["start"]).dt.floor("D")])["efc"].sum()
    daily_efc.index.set_names(["endpoint_id", "start_month", "day"], inplace=True)
    monthly_efc = daily_efc.groupby(["endpoint_id", "start_month"]).mean().unstack("endpoint_id")
    print(monthly_efc.round(2).to_string())

    # ============================================================ 2. UNMET DEMAND (PROXY)
    print("\n================ 2. UNMET DEMAND - PROXY, NOT DIRECT EVIDENCE ================")
    print("(no real charger/session queue data exists - dc_charger_1min and control_ev_1min")
    print(" are both dead in this dataset. This section infers squeeze from SoC running near-empty")
    print(" while still serving load, and output dropping when that happens. Treat as suggestive,")
    print(" not proof - drop this section from the pitch if the numbers below aren't convincing.")
    print(" peak_load is checked alongside ac_out_pwr: if peak_load (demand ceiling) stays high")
    print(" while delivered output falls, that's throttling under real demand - if peak_load falls")
    print(" too, demand simply dropped and the low-SoC reading isn't evidence of a squeeze. Most")
    print(" low-SoC-active minutes are logged status_mode=OffGrid - these units are islanded with")
    print(" no mains backup at the time, so this reads as 'standalone runtime ran out', not")
    print(" 'machinery queued for a charger'.)")

    for ep in endpoints:
        g = u[u["endpoint_id"] == ep].sort_values("ts_min").dropna(subset=["soc_avg"])
        spans = soc_low_while_serving_load(g)
        active = g[g["ac_out_pwr"] > 0]
        normal = active[active["soc_avg"] >= LOW_SOC_THRESHOLD]
        low = active[active["soc_avg"] < LOW_SOC_THRESHOLD]
        avg_out_normal, avg_out_low = normal["ac_out_pwr"].mean(), low["ac_out_pwr"].mean()
        peak_normal, peak_low = normal["peak_load"].mean(), low["peak_load"].mean()
        top_mode = low["status_mode"].value_counts(dropna=False).idxmax() if len(low) else None
        demand_held = peak_low >= 0.9 * peak_normal
        print(f"\n-- {ep} --")
        print(f"near-empty-while-serving-load events: n={len(spans)}, "
              f"total hours={spans['n'].sum() / 60:.1f} "
              f"({100 * len(low) / len(active):.1f}% of active minutes)")
        print(f"ac_out_pwr: normal={avg_out_normal:.1f} kW vs low-SoC={avg_out_low:.1f} kW "
              f"({100 * (avg_out_low / avg_out_normal - 1):+.0f}%)")
        print(f"peak_load:  normal={peak_normal:.1f} kW vs low-SoC={peak_low:.1f} kW "
              f"-> demand {'held up' if demand_held else 'also dropped'} "
              f"({'real throttling' if demand_held else 'weak evidence, demand fell too'})")
        print(f"dominant status_mode while low-SoC & active: {top_mode}")

    # ============================================================ 3. GROWTH TREND
    print("\n================ 3. GROWTH TREND ================")

    daily_all = []
    for ep in endpoints:
        d = daily_throughput(u[u["endpoint_id"] == ep].sort_values("ts_min")).rename("kwh").reset_index()
        d["endpoint_id"] = ep
        daily_all.append(d)
    daily_all = pd.concat(daily_all, ignore_index=True).rename(columns={"ts_min": "day"})
    daily_all["month"] = pd.to_datetime(daily_all["day"]).dt.to_period("M")

    print("-- fleet throughput (kWh) by month --")
    print(daily_all.groupby("month")["kwh"].sum().round(0).to_string())

    print("\n-- days per month with N units simultaneously active (fleet-wide) --")
    active_by_day = {
        ep: daily_throughput(u[u["endpoint_id"] == ep].sort_values("ts_min"))
        for ep in endpoints
    }
    active_by_day = {ep: s[s >= ACTIVE_DAY_MIN_KWH].index for ep, s in active_by_day.items()}
    all_days = sorted(set().union(*active_by_day.values()))
    overlap = pd.Series(
        {d: sum(d in active_by_day[ep] for ep in active_by_day) for d in all_days}
    )
    overlap.index = pd.to_datetime(overlap.index)
    month = overlap.index.to_period("M")
    print(overlap.groupby(month).value_counts().unstack(fill_value=0).sort_index(axis=1).to_string())
