"""Simple fleet summary: total throughput (ac_out_pwr -> kWh), plus how the fleet actually
cycles (depth, duration, recharges/day) - not just a bare cycle count, which hides whether
units are doing shallow top-ups or deep daily drains.
"""
import numpy as np
import pandas as pd

from analyze import load_unit_metrics, cycle_details

ACTIVE_DAY_MIN_KWH = 1.0  # a day below this delivered essentially nothing

EFC_BINS = [0, 0.5, 1.0, 1.5, 100]
EFC_LABELS = ["<0.5 (light)", "0.5-1 (moderate)", "1-1.5 (heavy)", "1.5+ (very heavy)"]


def daily_throughput(g):
    day = g["ts_min"].dt.floor("D")
    return g["ac_out_pwr"].clip(lower=0).groupby(day).sum() / 60.0  # kW * 1min samples -> kWh


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    u = load_unit_metrics()

    rows = []
    all_cycles = []
    for ep, g in u.groupby("endpoint_id"):
        g = g.sort_values("ts_min")
        calendar_days = (g["ts_min"].max() - g["ts_min"].min()) / np.timedelta64(1, "D")
        daily = daily_throughput(g)
        active_days = (daily >= ACTIVE_DAY_MIN_KWH).sum()
        throughput_kwh = daily.sum()

        cycles = cycle_details(u, ep)
        all_cycles.append(cycles)

        rows.append({
            "endpoint_id": ep,
            "calendar_days": calendar_days,
            "active_days": active_days,
            "throughput_kwh": throughput_kwh,
            "throughput_kwh_per_active_day": throughput_kwh / active_days,
            "throughput_kwh_per_cycle": throughput_kwh / len(cycles) if len(cycles) else np.nan,
        })

    per_unit = pd.DataFrame(rows)
    all_cycles = pd.concat(all_cycles, ignore_index=True)
    all_cycles["start_day"] = pd.to_datetime(all_cycles["start"]).dt.floor("D")
    # equivalent full cycles: each cycle's depth as a fraction of a full 0-100% swing,
    # attributed to the day it started - combines "how hard" and "how often" into one number.
    all_cycles["efc"] = all_cycles["depth"] / 100.0

    print("\n================ PER-UNIT ================")
    print(per_unit.to_string(index=False))

    print("\n================ FLEET TOTAL ================")
    total_throughput = per_unit["throughput_kwh"].sum()
    total_cycles = len(all_cycles)
    total_active_days = per_unit["active_days"].sum()
    total_calendar_days = per_unit["calendar_days"].sum()
    print(f"total throughput:   {total_throughput:,.0f} kWh")
    print(f"total cycles:        {total_cycles}")
    print(f"total active days:   {total_active_days:,.0f} (of {total_calendar_days:,.0f} calendar unit-days)")

    print("\n================ FLEET AVERAGE (per unit per active day) ================")
    print(f"avg throughput/unit/active day: {total_throughput/total_active_days:,.1f} kWh")
    print(f"avg throughput/cycle:           {total_throughput/total_cycles:,.1f} kWh")

    print("\n================ HOW HARD & HOW OFTEN: EQUIVALENT FULL CYCLES / DAY ================")
    print("(each cycle's depth as a fraction of a full 0-100% swing, summed per day it started -")
    print(" e.g. one 90%-deep cycle = 0.9 EFC that day; three 40%-deep cycles = 1.2 EFC that day)")
    daily_efc = all_cycles.groupby(["endpoint_id", "start_day"])["efc"].sum()

    print("\n-- avg EFC/day per unit (days with at least one cycle) --")
    print(daily_efc.groupby("endpoint_id").mean().rename("avg_efc_per_day").round(2))

    print("\n-- distribution of days by usage intensity --")
    efc_bucket = pd.cut(daily_efc, bins=EFC_BINS, labels=EFC_LABELS)
    print(efc_bucket.groupby("endpoint_id", observed=True).value_counts().unstack(fill_value=0)[EFC_LABELS])
