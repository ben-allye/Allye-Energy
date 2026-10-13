"""Round-trip efficiency, capacity fade, fleet overlap, and usage rhythm."""
import numpy as np
import pandas as pd

from analyze import load_unit_metrics, cycle_details, load_pack_daily, clean_cumulative_diff
from summary import daily_throughput, ACTIVE_DAY_MIN_KWH


def cycle_energy(u, ep, cycles):
    g = u[u["endpoint_id"] == ep].sort_values("ts_min").reset_index(drop=True)
    ts_arr = g["ts_min"].to_numpy()
    prefix_out = clean_cumulative_diff(g, "total_kwh_out").cumsum().to_numpy()
    prefix_in = clean_cumulative_diff(g, "total_kwh_in").cumsum().to_numpy()

    kwh_out, kwh_in = [], []
    for _, row in cycles.iterrows():
        lo = np.searchsorted(ts_arr, np.datetime64(row["start"]), side="left")
        hi = np.searchsorted(ts_arr, np.datetime64(row["end"]), side="right") - 1
        kwh_out.append(prefix_out[hi] - prefix_out[lo] if hi > lo else 0.0)
        kwh_in.append(prefix_in[hi] - prefix_in[lo] if hi > lo else 0.0)
    cycles = cycles.copy()
    cycles["kwh_out"], cycles["kwh_in"] = kwh_out, kwh_in
    return cycles


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    u = load_unit_metrics()

    # ---------------------------------------------------------- efficiency
    print("\n================ ROUND-TRIP EFFICIENCY (per cycle: kwh_out / kwh_in) ================")
    print("(cycles with negligible kwh_in or kwh_out excluded - e.g. a cycle that re-triggers")
    print(" almost immediately captures only the recharge's topping-off tail with barely any")
    print(" discharge in that window, which isn't a meaningful efficiency reading)")
    for ep in u["endpoint_id"].unique():
        cycles = cycle_details(u, ep)
        cycles = cycle_energy(u, ep, cycles)
        valid = cycles[(cycles["kwh_in"] > 5) & (cycles["kwh_out"] > 5)]
        eff = valid["kwh_out"] / valid["kwh_in"]
        print(f"{ep}: n={len(valid)}, mean eff={eff.mean():.2%}, median={eff.median():.2%}, "
              f"p10={eff.quantile(.1):.2%}, p90={eff.quantile(.9):.2%}")

    # ------------------------------------------------------- capacity fade
    print("\n================ CAPACITY FADE (energy_full_nominal, sum of 6 packs) ================")
    pack = load_pack_daily()
    pack["day"] = pd.to_datetime(pack["day"])
    unit_capacity_daily = (
        pack.groupby(["endpoint_id", "day"])["energy_full_nominal_avg"].sum()
        .reset_index()
    )
    for ep, g in unit_capacity_daily.groupby("endpoint_id"):
        g = g.sort_values("day").set_index("day")["energy_full_nominal_avg"]
        # a month with only a handful of days (onboarding/deployment) isn't a fair
        # capacity reading - require most of the month present before trusting it.
        counts = g.resample("ME").count()
        monthly = g.resample("ME").mean()
        monthly = monthly[counts >= 15]
        first, last = monthly.iloc[0], monthly.iloc[-1]
        print(f"\n-- {ep} -- first full month={first:.0f} kWh, last full month={last:.0f} kWh, "
              f"change={100*(last-first)/first:+.1f}%")
        print(monthly.round(0).to_string())

    # -------------------------------------------------------- fleet overlap
    print("\n================ FLEET OVERLAP (how many units active on the same day) ================")
    active_by_day = {}
    for ep, g in u.groupby("endpoint_id"):
        daily = daily_throughput(g.sort_values("ts_min"))
        active_by_day[ep] = daily[daily >= ACTIVE_DAY_MIN_KWH].index
    all_days = sorted(set().union(*active_by_day.values()))
    overlap = pd.Series(
        {d: sum(d in active_by_day[ep] for ep in active_by_day) for d in all_days}
    )
    print(overlap.value_counts().sort_index().rename("n_days").to_string())
    print(f"(days with 2+ units active simultaneously: {(overlap >= 2).sum()} of {len(overlap)})")

    # --------------------------------------------------------- usage rhythm
    print("\n================ USAGE RHYTHM ================")
    print("-- avg throughput (kWh) by day of week, fleet-wide --")
    daily_all = []
    for ep, g in u.groupby("endpoint_id"):
        d = daily_throughput(g.sort_values("ts_min")).rename("kwh").reset_index().rename(columns={"ts_min": "day"})
        d["endpoint_id"] = ep
        daily_all.append(d)
    daily_all = pd.concat(daily_all, ignore_index=True)
    daily_all["dow"] = pd.to_datetime(daily_all["day"]).dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    print(daily_all.groupby("dow")["kwh"].mean().reindex(dow_order).round(1).to_string())

    print("\n-- avg throughput (kWh) by month, fleet-wide --")
    daily_all["month"] = pd.to_datetime(daily_all["day"]).dt.to_period("M")
    print(daily_all.groupby("month")["kwh"].mean().round(1).to_string())
