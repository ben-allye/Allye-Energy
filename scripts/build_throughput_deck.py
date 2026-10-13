"""Rebuild notebooks/throughput_summary_deck.html from the raw data.

Recomputes every number and chart in the deck straight from unit_metrics /
pack_metrics (same pipeline as build_throughput_notebook.py), then fills in
scripts/deck_template.html (the DATA_JSON blob the charts draw from, plus the
{{PLACEHOLDER}} tokens used in table rows and prose numbers) and writes the
result to notebooks/throughput_summary_deck.html.

Run this after the underlying data changes, or after editing
deck_template.html. It does NOT touch throughput_summary.ipynb — that's
build_throughput_notebook.py's job.

Usage: uv run python scripts/build_throughput_deck.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import (  # noqa: E402
    load_unit_metrics, load_pack_daily, load_pack_1min, cycle_details,
    LOAD_ON_THRESHOLD_KW, SESSION_GAP_MIN,
)
from summary import daily_throughput  # noqa: E402
from insights import cycle_energy  # noqa: E402

TEMPLATE_PATH = ROOT / "scripts" / "deck_template.html"
OUT_PATH = ROOT / "notebooks" / "throughput_summary_deck.html"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def nice_axis(data_max, n_ticks=4, headroom=1.15):
    """Round y-axis top + evenly spaced ticks that comfortably cover data_max."""
    top_needed = max(data_max, 1e-9) * headroom
    raw_step = top_needed / n_ticks
    mag = 10 ** math.floor(math.log10(raw_step))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw_step <= m * mag), 10 * mag)
    ymax = math.ceil(top_needed / step) * step
    ticks = [round(step * i, 6) for i in range(int(round(ymax / step)) + 1)]
    return ymax, ticks


def main():
    u = load_unit_metrics()
    pack = load_pack_daily()
    pack_1min = load_pack_1min()
    endpoints = sorted(u["endpoint_id"].unique())

    # ---- capacity correction (same as throughput_summary.ipynb) ----
    PACK_CAPACITY_MIN_KWH = 20
    UNIT_ACTIVE_CURRENT_A = 1.0
    PACK_IDLE_CURRENT_A = 0.5
    DAY_EXCLUDE_FRAC = 0.5

    clean_pack = pack[pack["energy_full_nominal_avg"] > PACK_CAPACITY_MIN_KWH].copy()
    clean_pack["day"] = pd.to_datetime(clean_pack["day"])
    piv_1min = pack_1min.pivot_table(
        index=["endpoint_id", "ts_min"], columns="pack_number", values="current_avg", aggfunc="mean"
    ).abs()
    unit_active = piv_1min.median(axis=1, skipna=True) > UNIT_ACTIVE_CURRENT_A
    flag_rows = []
    for pn in piv_1min.columns:
        unavailable_minute = unit_active & (piv_1min[pn].isna() | (piv_1min[pn] < PACK_IDLE_CURRENT_A))
        flag_rows.append(pd.DataFrame({"pack_number": pn, "unit_active": unit_active, "unavailable": unavailable_minute}))
    flags_1min = pd.concat(flag_rows).reset_index()
    flags_1min["day"] = flags_1min["ts_min"].dt.floor("D")
    daily_flags = flags_1min.groupby(["endpoint_id", "pack_number", "day"]).agg(
        active_minutes=("unit_active", "sum"), unavailable_minutes=("unavailable", "sum")
    ).reset_index()
    daily_flags["frac_unavailable"] = daily_flags["unavailable_minutes"] / daily_flags["active_minutes"].replace(0, np.nan)
    excluded = daily_flags[daily_flags["frac_unavailable"] > DAY_EXCLUDE_FRAC][["endpoint_id", "pack_number", "day"]].copy()
    excluded["excluded"] = True
    capacity_pack_days = clean_pack.merge(excluded, on=["endpoint_id", "pack_number", "day"], how="left")
    capacity_pack_days = capacity_pack_days[capacity_pack_days["excluded"] != True]  # noqa: E712
    daily_capacity = capacity_pack_days.groupby(["endpoint_id", "day"])["energy_full_nominal_avg"].sum()

    def capacity_series_for(ep):
        s = daily_capacity.loc[ep]
        full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
        return s.reindex(full_range).ffill().bfill()

    capacity_by_day = {ep: capacity_series_for(ep) for ep in pack["endpoint_id"].unique()}

    # ---- fleet total throughput ----
    daily_all = []
    for ep in endpoints:
        d = daily_throughput(u[u["endpoint_id"] == ep].sort_values("ts_min")).rename("kwh").reset_index().rename(columns={"ts_min": "day"})
        daily_all.append(d)
    daily_all = pd.concat(daily_all, ignore_index=True)
    daily_all["month"] = pd.to_datetime(daily_all["day"]).dt.to_period("M")
    fleet_monthly = daily_all.groupby("month")["kwh"].sum()

    go_live = u.groupby("endpoint_id")["ts_min"].min().dt.floor("D")
    all_days_range = pd.date_range(u["ts_min"].min().floor("D"), u["ts_min"].max().floor("D"), freq="D")
    n_live = pd.Series([sum(go_live[ep] <= d for ep in go_live.index) for d in all_days_range], index=all_days_range)
    n_live_monthly = n_live.groupby(n_live.index.to_period("M")).mean()
    per_live_unit_monthly = fleet_monthly / n_live_monthly

    # ---- usage rhythm (fleet total per day, then averaged by day-of-week) ----
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fleet_per_day = daily_all.groupby("day")["kwh"].sum().reset_index()
    fleet_per_day["dow"] = pd.to_datetime(fleet_per_day["day"]).dt.day_name()
    weekday_kwh = fleet_per_day.groupby("dow")["kwh"].mean().reindex(dow_order)

    # ---- time of use ----
    tou = u.copy()
    tou["dow"] = tou["ts_min"].dt.day_name()
    tou["hour"] = tou["ts_min"].dt.hour
    weekday_tou = tou[~tou["dow"].isin(["Saturday", "Sunday"])]
    n_weekdays = weekday_tou["ts_min"].dt.floor("D").nunique()
    hourly_out = (weekday_tou.groupby("hour")["ac_out_pwr"].apply(lambda s: s.clip(lower=0).sum() / 60.0) / n_weekdays).reindex(range(24), fill_value=0.0)
    hourly_in = (weekday_tou.groupby("hour")["ac_in_pwr"].apply(lambda s: s.clip(lower=0).sum() / 60.0) / n_weekdays).reindex(range(24), fill_value=0.0)

    piv_tou = tou.groupby(["dow", "hour"])["ac_out_pwr"].apply(lambda s: s.clip(lower=0).sum() / 60.0).unstack("hour")
    piv_tou = piv_tou.reindex(index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday", "Sunday"], columns=range(24)).fillna(0.0)

    # ---- cycles: EFC + round-trip efficiency ----
    all_cycles = []
    eff_rows = []
    for ep in endpoints:
        cycles = cycle_details(u, ep)
        cycles = cycle_energy(u, ep, cycles)
        cycles["start_day"] = pd.to_datetime(cycles["start"]).dt.floor("D")
        cap_at_start = cycles["start_day"].map(capacity_by_day[ep])
        cycles["efc"] = cycles["kwh_out"] / cap_at_start
        cycles["endpoint_id"] = ep
        all_cycles.append(cycles)

        valid = cycles[(cycles["kwh_in"] > 5) & (cycles["kwh_out"] > 5)]
        eff = valid["kwh_out"] / valid["kwh_in"]
        eff_rows.append({"endpoint_id": ep, "n_cycles": len(valid), "median_eff": eff.median(), "p10": eff.quantile(.1), "p90": eff.quantile(.9)})
    all_cycles = pd.concat(all_cycles, ignore_index=True)
    eff_df = pd.DataFrame(eff_rows).set_index("endpoint_id")

    fleet_daily_efc = all_cycles.groupby("start_day")["efc"].sum()
    full_range = pd.date_range(fleet_daily_efc.index.min(), fleet_daily_efc.index.max(), freq="D")
    fleet_daily_efc = fleet_daily_efc.reindex(full_range, fill_value=0)

    daily_efc_by_unit = {ep: all_cycles[all_cycles["endpoint_id"] == ep].groupby("start_day")["efc"].sum() for ep in endpoints}

    # ---- charge time ----
    def detect_power_sessions(g, col):
        g = g.dropna(subset=[col]).set_index("ts_min").sort_index()
        active = g[col].abs() > LOAD_ON_THRESHOLD_KW
        active = active.reindex(pd.date_range(g.index.min(), g.index.max(), freq="1min"), fill_value=False)
        run_id = (active != active.shift()).cumsum()
        run_len = active.groupby(run_id).transform("size")
        bridged = active | (~active & (run_len <= SESSION_GAP_MIN))
        session_id = (bridged != bridged.shift(fill_value=False)).cumsum()
        sessions = (
            pd.DataFrame({"active": bridged, "session_id": session_id}).reset_index(names="ts")
            .groupby("session_id").agg(start=("ts", "min"), end=("ts", "max"), active=("active", "first"))
        )
        sessions = sessions[sessions["active"]].copy()
        sessions["duration_h"] = (sessions["end"] - sessions["start"]) / np.timedelta64(1, "h") + (1 / 60)
        return sessions.reset_index(drop=True)

    CHARGE_SOC_MIN = 5.0
    all_charge_sessions = []
    for ep in endpoints:
        g = u[u["endpoint_id"] == ep].sort_values("ts_min").reset_index(drop=True)
        sess = detect_power_sessions(g, "ac_in_pwr")
        soc_series = g.set_index("ts_min")["soc_avg"]
        sess["soc_start"] = soc_series.reindex(sess["start"], method="nearest").to_numpy()
        sess["soc_end"] = soc_series.reindex(sess["end"], method="nearest").to_numpy()
        sess["delta_soc"] = sess["soc_end"] - sess["soc_start"]
        sess["endpoint_id"] = ep
        all_charge_sessions.append(sess)
    all_charge_sessions = pd.concat(all_charge_sessions, ignore_index=True)
    meaningful = all_charge_sessions[all_charge_sessions["delta_soc"] >= CHARGE_SOC_MIN].copy()
    meaningful["rate_pct_per_h"] = meaningful["delta_soc"] / meaningful["duration_h"]
    meaningful["equiv_full_charge_h"] = 100.0 / meaningful["rate_pct_per_h"]
    equiv_full = meaningful.groupby("endpoint_id")["equiv_full_charge_h"].agg(median_h="median", mean_h="mean", n="size")
    fleet_charge_median = meaningful["equiv_full_charge_h"].median()

    # ---- boxplot / histogram stats ----
    box_stats = {}
    hist_counts = {}
    bins = np.linspace(0, 2.0, 21)
    hist_max_count = 0
    for ep in endpoints:
        arr = daily_efc_by_unit[ep].to_numpy()
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inliers = arr[(arr >= lo_fence) & (arr <= hi_fence)]
        outliers = sorted(arr[(arr < lo_fence) | (arr > hi_fence)].tolist())
        box_stats[ep] = {
            "q1": round(q1, 3), "median": round(med, 3), "q3": round(q3, 3),
            "whisker_lo": round(inliers.min(), 3) if len(inliers) else round(q1, 3),
            "whisker_hi": round(inliers.max(), 3) if len(inliers) else round(q3, 3),
            "outliers": [round(v, 3) for v in outliers],
            "min": round(arr.min(), 3), "max": round(arr.max(), 3),
        }
        counts, _ = np.histogram(np.clip(arr, None, 2.0), bins=bins)
        hist_counts[ep] = counts.tolist()
        hist_max_count = max(hist_max_count, int(counts.max()))

    # ================= assemble DATA (drives the JS charts) =================
    fleet_monthly_ymax, fleet_monthly_yticks = nice_axis(fleet_monthly.max())
    per_live_ymax, per_live_yticks = nice_axis(per_live_unit_monthly.max())
    weekday_ymax, weekday_yticks = nice_axis(weekday_kwh.max())
    tou_hourly_ymax, tou_hourly_yticks = nice_axis(max(hourly_out.max(), hourly_in.max()))
    efc_line_ymax, efc_line_yticks = nice_axis(fleet_daily_efc.max())
    box_ymax, box_yticks = nice_axis(max(s["whisker_hi"] for s in box_stats.values()) * 1.3, n_ticks=5)
    charge_ymax, charge_yticks = nice_axis(equiv_full["median_h"].max())

    data = {
        "units": endpoints,
        "months": [str(m) for m in fleet_monthly.index],
        "fleet_monthly": [round(v, 1) for v in fleet_monthly.values],
        "fleet_monthly_ymax": fleet_monthly_ymax, "fleet_monthly_yticks": fleet_monthly_yticks,
        "per_live_unit_monthly": [round(v, 1) for v in per_live_unit_monthly.values],
        "per_live_ymax": per_live_ymax, "per_live_yticks": per_live_yticks,
        "go_live": {ep: str(go_live[ep].date()) for ep in endpoints},
        "weekday_kwh": {k: round(v, 1) for k, v in weekday_kwh.items()},
        "weekday_ymax": weekday_ymax, "weekday_yticks": weekday_yticks,
        "tou_hourly_out": [round(v, 2) for v in hourly_out.values],
        "tou_hourly_in": [round(v, 2) for v in hourly_in.values],
        "tou_hourly_ymax": tou_hourly_ymax, "tou_hourly_yticks": tou_hourly_yticks,
        "tou_heatmap_rows": list(piv_tou.index),
        "tou_heatmap": [[round(v, 1) for v in row] for row in piv_tou.values],
        "tou_heatmap_max": round(float(piv_tou.values.max()), 1),
        "fleet_daily_efc_dates": [d.strftime("%Y-%m-%d") for d in fleet_daily_efc.index],
        "fleet_daily_efc_values": [round(v, 3) for v in fleet_daily_efc.values],
        "efc_line_ymax": efc_line_ymax, "efc_line_yticks": efc_line_yticks,
        "efficiency": {
            ep: {"n": int(eff_df.loc[ep, "n_cycles"]), "median": round(eff_df.loc[ep, "median_eff"], 3),
                 "p10": round(eff_df.loc[ep, "p10"], 3), "p90": round(eff_df.loc[ep, "p90"], 3)}
            for ep in endpoints
        },
        "charge_time": {
            ep: {"median_h": round(equiv_full.loc[ep, "median_h"], 2), "mean_h": round(equiv_full.loc[ep, "mean_h"], 2), "n": int(equiv_full.loc[ep, "n"])}
            for ep in endpoints
        },
        "charge_ymax": charge_ymax, "charge_yticks": charge_yticks,
        "box_stats": box_stats,
        "box_ymax": box_ymax, "box_yticks": box_yticks,
        "hist_bins": bins.tolist(),
        "hist_counts": hist_counts,
        "hist_max_count": hist_max_count,
    }

    # ================= placeholder text (tables + prose numbers) =================
    date_min, date_max = u["ts_min"].min(), u["ts_min"].max()
    n_months = round((date_max - date_min).days / 30.44)

    def golive_chip(ep, i):
        return (
            f'      <div class="chip"><span class="sw" style="background:var(--u{i+1})"></span>'
            f'<span class="id">{ep[-4:]}</span><span class="d">live {go_live[ep].date()}</span></div>'
        )

    weekday_rows = "\n".join(
        f'          <tr><td>{d}</td><td>{weekday_kwh[d]:.1f}</td></tr>' for d in dow_order
    )
    efficiency_rows = "\n".join(
        f'          <tr><td><span class="swatch" style="background:var(--u{i+1})"></span>{ep[-4:]}</td>'
        f'<td>{eff_df.loc[ep, "n_cycles"]}</td><td>{eff_df.loc[ep, "median_eff"]*100:.1f}%</td>'
        f'<td>{eff_df.loc[ep, "p10"]*100:.1f}–{eff_df.loc[ep, "p90"]*100:.1f}%</td></tr>'
        for i, ep in enumerate(endpoints)
    )
    charge_rows = "\n".join(
        f'          <tr><td><span class="swatch" style="background:var(--u{i+1})"></span>{ep[-4:]}</td>'
        f'<td>{equiv_full.loc[ep, "n"]:.0f}</td><td>{equiv_full.loc[ep, "median_h"]:.2f}</td>'
        f'<td>{equiv_full.loc[ep, "mean_h"]:.2f}</td></tr>'
        for i, ep in enumerate(endpoints)
    )
    dist_rows = "\n".join(
        f'      <tr><td><span class="swatch" style="background:var(--u{i+1})"></span>{ep[-4:]}</td>'
        f'<td>{box_stats[ep]["median"]:.2f}</td><td>{box_stats[ep]["q1"]:.2f}–{box_stats[ep]["q3"]:.2f}</td>'
        f'<td>{np.percentile(daily_efc_by_unit[ep], 90):.2f}</td><td>{box_stats[ep]["max"]:.2f}</td></tr>'
        for i, ep in enumerate(endpoints)
    )

    eff_medians = {ep: eff_df.loc[ep, "median_eff"] * 100 for ep in endpoints}
    dist_medians = {ep: box_stats[ep]["median"] for ep in endpoints}
    outlier_unit = max(endpoints, key=lambda ep: box_stats[ep]["max"])
    other_max = max(box_stats[ep]["max"] for ep in endpoints if ep != outlier_unit)
    fastest_unit = equiv_full["median_h"].idxmin()
    slowest_unit = equiv_full["median_h"].idxmax()
    sat_pct = round(weekday_kwh["Saturday"] / weekday_kwh[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean() * 100)
    full_months = fleet_monthly.iloc[1:-1] if len(fleet_monthly) > 2 else fleet_monthly
    peak_month = full_months.idxmax()
    weekday_top = sorted(dow_order[:5], key=lambda day: weekday_kwh[day], reverse=True)[:2]

    placeholders = {
        "DATE_RANGE": f"{date_min:%d %b}".upper() + f" – {date_max:%d %b %Y}".upper(),
        "DATE_RANGE_SUB": f"{date_min:%d %b} – {date_max:%d %b %Y}",
        "N_UNITS": str(len(endpoints)),
        "N_MONTHS": f"~{n_months} months",
        "TOTAL_CYCLES": str(int(eff_df["n_cycles"].sum())),
        "PEAK_MONTH": peak_month.strftime("%B"),
        "PEAK_MONTH_MWH": f"{full_months.max()/1000:.1f}",
        "GOLIVE_CHIPS": "\n".join(golive_chip(ep, i) for i, ep in enumerate(endpoints)),
        "GOLIVE_FIRST": f"{go_live[endpoints[0]]:%b %-d}",
        "GOLIVE_LAST": f"{go_live.max():%b %-d}",
        "WEEKDAY_TABLE_ROWS": weekday_rows,
        "WD_PEAK_AVG": f"{max(weekday_kwh['Tuesday'], weekday_kwh['Wednesday']):.0f}",
        "WD_LOW_LO": f"{min(weekday_kwh['Monday'], weekday_kwh['Friday']):.0f}",
        "WD_LOW_HI": f"{max(weekday_kwh['Monday'], weekday_kwh['Friday']):.0f}",
        "WD_SAT": f"{weekday_kwh['Saturday']:.0f}",
        "WD_SUN": f"{weekday_kwh['Sunday']:.1f}",
        "WD_SAT_PCT": str(sat_pct),
        "WD_TOP_1": weekday_top[0],
        "WD_TOP_2": weekday_top[1],
        "WD_TOP_1_KWH": f"{weekday_kwh[weekday_top[0]]:.0f}",
        "WD_TOP_2_KWH": f"{weekday_kwh[weekday_top[1]]:.0f}",
        "TOU_PEAK_DIS_HOUR": f"{hourly_out.idxmax():02d}:00",
        "TOU_PEAK_DIS_KWH": f"{hourly_out.max():.0f}",
        "TOU_PEAK_CHG_HOUR": f"{hourly_in.idxmax():02d}:00",
        "TOU_PEAK_CHG_KWH": f"{hourly_in.max():.0f}",
        "TOU_HEATMAP_MAX": f"{piv_tou.values.max():.0f}",
        "EFC_MEAN": f"{fleet_daily_efc.mean():.2f}",
        "EFC_MEDIAN": f"{fleet_daily_efc.median():.2f}",
        "EFC_MAX": f"{fleet_daily_efc.max():.2f}",
        "EFC_MAX_RATIO": f"{fleet_daily_efc.max()/len(endpoints):.1f}",
        "EFFICIENCY_TABLE_ROWS": efficiency_rows,
        "EFF_MEDIAN_MIN": f"{min(eff_medians.values()):.1f}",
        "EFF_MEDIAN_MAX": f"{max(eff_medians.values()):.1f}",
        "CHARGE_TABLE_ROWS": charge_rows,
        "CHARGE_FLEET_MEDIAN": f"{fleet_charge_median:.2f}",
        "CHARGE_FASTEST_UNIT": fastest_unit[-4:],
        "CHARGE_FASTEST_H": f"{equiv_full.loc[fastest_unit, 'median_h']:.2f}",
        "CHARGE_SLOWEST_UNIT": slowest_unit[-4:],
        "CHARGE_SLOWEST_H": f"{equiv_full.loc[slowest_unit, 'median_h']:.2f}",
        "DIST_TABLE_ROWS": dist_rows,
        "DIST_MEDIAN_MIN": f"{min(dist_medians.values()):.2f}",
        "DIST_MEDIAN_MAX": f"{max(dist_medians.values()):.2f}",
        "DIST_OUTLIER_UNIT": outlier_unit[-4:],
        "DIST_OUTLIER_MAX": f"{box_stats[outlier_unit]['max']:.2f}",
        "DIST_OTHER_MAX": f"{other_max:.2f}",
    }

    data["summary"] = {
        "start_date": date_min.strftime("%Y-%m-%d"),
        "end_date": date_max.strftime("%Y-%m-%d"),
        "total_cycles": int(eff_df["n_cycles"].sum()),
        "total_throughput_kwh": round(float(fleet_monthly.sum()), 1),
        "fleet_charge_median_h": round(float(fleet_charge_median), 2),
        "saturday_pct_weekday": sat_pct,
    }

    html = TEMPLATE_PATH.read_text()
    html = html.replace("{{DATA_JSON}}", json.dumps(data, separators=(",", ":")))
    for key, value in placeholders.items():
        html = html.replace("{{" + key + "}}", value)

    remaining = [tok for tok in html.split("{{")[1:] if "}}" in tok]
    if remaining:
        missing = sorted({tok.split("}}")[0] for tok in remaining})
        raise RuntimeError(f"Unfilled placeholders in deck template: {missing}")

    OUT_PATH.write_text(html)
    print(f"wrote {OUT_PATH} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
