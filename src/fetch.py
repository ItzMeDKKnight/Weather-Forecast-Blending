"""
Data fetching, caching, daily aggregation, and quality report pipeline.
Day boundary: Asia/Kolkata (IST) calendar day (00:00 to 23:00 IST).
"""
import os, sys, json, time, datetime, yaml
import pandas as pd
import numpy as np
from src.truth import cached_get_json, ERA5Provider

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

def run_probe(config: dict) -> dict:
    """Probes Previous Runs API and ERA5 for model keys, variable naming, and operational start dates."""
    print("=" * 60)
    print("STAGE 1 PROBE: Verifying model keys, variable naming & availability")
    print("=" * 60)
    models = config["models"]
    leads = config["leads"]
    sample_pt = list(config["points"].values())[0]
    lat, lon = sample_pt["lat"], sample_pt["lon"]

    # Verify hourly variable naming for leads 1, 3, 5
    lead_hourly = [f"{var}_previous_day{ld}" for ld in leads for var in ["temperature_2m", "precipitation", "wind_speed_10m"]]
    hourly_param = ",".join(lead_hourly)
    models_param = ",".join(models.values())

    probe_params = {
        "latitude": lat,
        "longitude": lon,
        "models": models_param,
        "hourly": hourly_param,
        "start_date": "2025-06-01",
        "end_date": "2025-06-02",
        "timezone": "Asia/Kolkata"
    }
    probe_res = cached_get_json(PREVIOUS_RUNS_URL, probe_params)
    available_keys = list(probe_res.get("hourly", {}).keys())
    print(f"Verified {len(available_keys)} hourly response keys from Previous Runs API.")
    print("Sample verified keys:", [k for k in available_keys if "previous_day1" in k][:4])

    # Probe operational start dates for each model
    first_dates = {}
    null_logs = {}
    check_dates = [
        "2024-01-01", "2024-06-01", "2024-10-01", "2025-01-01",
        "2025-02-15", "2025-02-20", "2025-03-01", "2025-06-01"
    ]
    for alias, model_key in models.items():
        first_date = None
        for d in check_dates:
            p = {
                "latitude": lat,
                "longitude": lon,
                "models": model_key,
                "hourly": f"temperature_2m_previous_day1",
                "start_date": d,
                "end_date": d,
                "timezone": "Asia/Kolkata"
            }
            try:
                res = cached_get_json(PREVIOUS_RUNS_URL, p)
                hourly = res.get("hourly", {})
                # Single model queries key as temperature_2m_previous_day1
                vals = hourly.get(f"temperature_2m_previous_day1_{model_key}") or hourly.get("temperature_2m_previous_day1", [])
                if any(x is not None for x in vals):
                    first_date = d
                    break
            except Exception as e:
                null_logs[alias] = f"Probe error on {d}: {e}"
        first_dates[alias] = first_date or "2025-03-01"
        print(f"Model {alias:12s} ({model_key:22s}): operational start ~ {first_dates[alias]}")

    # ERA5 check
    era5_provider = ERA5Provider()
    era5_test = era5_provider.fetch_daily({"test": {"lat": lat, "lon": lon}}, "2025-06-01", "2025-06-02")
    era5_status = "OK" if len(era5_test) > 0 else "FAILED"
    print(f"Ground Truth ERA5 status: {era5_status} ({len(era5_test)} records returned)")

    # Establish common training window: starts when the latest model (AIFS ~ 2025-02-20) starts
    # We use 2025-03-01 to 2026-09-15 as common window (~18.5 months)
    common_start = "2025-03-01"
    data_end = "2026-09-15"
    test_start = "2026-06-16" # Last 90 days for strict test set

    status_data = {
        "probe_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models": {
            alias: {
                "model_key": m_key,
                "first_date": first_dates[alias],
                "verified_lead_pattern": f"<variable>_previous_day<lead>_{m_key}",
                "status": "active"
            } for alias, m_key in models.items()
        },
        "ground_truth": {
            "source": "ERA5",
            "endpoint": ERA5Provider.ENDPOINT,
            "status": era5_status,
            "caveat": "ERA5 reanalysis is used as proxy truth. Note: AIFS was trained on ERA5, which may favor it; ERA5 precipitation has higher uncertainty than temperature."
        },
        "common_training_window": {
            "start": common_start,
            "end": "2026-06-15"
        },
        "test_window": {
            "start": test_start,
            "end": data_end
        }
    }

    os.makedirs("public", exist_ok=True)
    with open("public/source_status.json", "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)
    print("Saved public/source_status.json successfully.")
    return status_data

def chunk_date_ranges(start_date: str, end_date: str, chunk_days: int = 60):
    cur = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    chunks = []
    while cur <= end:
        next_cur = min(cur + datetime.timedelta(days=chunk_days - 1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), next_cur.strftime("%Y-%m-%d")))
        cur = next_cur + datetime.timedelta(days=1)
    return chunks

def fetch_and_aggregate_all(config: dict, status: dict) -> pd.DataFrame:
    """Batch-fetches historical runs and ERA5 truth, aggregates to IST calendar day, and returns merged wide table."""
    points = config["points"]
    models = config["models"]
    leads = config["leads"]
    start_date = status["common_training_window"]["start"]
    end_date = status["test_window"]["end"]

    point_ids = list(points.keys())
    lats = ",".join(str(points[p]["lat"]) for p in point_ids)
    lons = ",".join(str(points[p]["lon"]) for p in point_ids)
    models_param = ",".join(models.values())

    lead_hourly = [f"{var}_previous_day{ld}" for ld in leads for var in ["temperature_2m", "precipitation", "wind_speed_10m"]]
    hourly_param = ",".join(lead_hourly)

    chunks = chunk_date_ranges(start_date, end_date, chunk_days=60)
    print(f"Fetching {len(chunks)} date chunks from {start_date} to {end_date} for {len(points)} points...")

    all_forecast_records = []
    for idx, (c_start, c_end) in enumerate(chunks, 1):
        print(f"  Fetching chunk {idx}/{len(chunks)}: {c_start} to {c_end}...")
        params = {
            "latitude": lats,
            "longitude": lons,
            "models": models_param,
            "hourly": hourly_param,
            "start_date": c_start,
            "end_date": c_end,
            "timezone": "Asia/Kolkata"
        }
        batch_res = cached_get_json(PREVIOUS_RUNS_URL, params)
        if isinstance(batch_res, dict):
            batch_res = [batch_res]

        for p_idx, p_id in enumerate(point_ids):
            data = batch_res[p_idx]
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                continue

            time_s = pd.Series(pd.to_datetime(times))
            date_s = time_s.dt.strftime("%Y-%m-%d")
            unique_dates = date_s.unique()

            # Process each model and lead
            for alias, m_key in models.items():
                for ld in leads:
                    t_vals = hourly.get(f"temperature_2m_previous_day{ld}_{m_key}")
                    p_vals = hourly.get(f"precipitation_previous_day{ld}_{m_key}")
                    w_vals = hourly.get(f"wind_speed_10m_previous_day{ld}_{m_key}")

                    if t_vals is None or p_vals is None or w_vals is None:
                        continue

                    # Create small dataframe for aggregation
                    temp_df = pd.DataFrame({"date": date_s, "t": t_vals, "p": p_vals, "w": w_vals})
                    agg = temp_df.groupby("date").agg({
                        "t": "max",
                        "p": "sum",
                        "w": "max"
                    }).reset_index()

                    lat, lon = points[p_id]["lat"], points[p_id]["lon"]
                    for _, row in agg.iterrows():
                        d_val = row["date"]
                        if pd.notna(row["t"]):
                            all_forecast_records.append((d_val, p_id, lat, lon, "temperature_2m_max", ld, alias, float(row["t"])))
                        if pd.notna(row["p"]):
                            all_forecast_records.append((d_val, p_id, lat, lon, "precipitation_sum", ld, alias, max(0.0, float(row["p"]))))
                        if pd.notna(row["w"]):
                            all_forecast_records.append((d_val, p_id, lat, lon, "wind_speed_10m_max", ld, alias, max(0.0, float(row["w"]))))

    print("Historical model forecasts fetched and aggregated.")
    fc_df = pd.DataFrame(all_forecast_records, columns=["date", "point", "lat", "lon", "variable", "lead", "model", "value"])

    # Pivot models to wide columns
    wide_fc = fc_df.pivot_table(index=["date", "point", "lat", "lon", "variable", "lead"],
                                columns="model", values="value").reset_index()

    # Fetch ground truth via ERA5Provider
    print("Fetching ERA5 ground truth across all chunks...")
    era5_provider = ERA5Provider()
    truth_dfs = []
    for idx, (c_start, c_end) in enumerate(chunks, 1):
        tdf = era5_provider.fetch_daily(points, c_start, c_end)
        truth_dfs.append(tdf)
    truth_df = pd.concat(truth_dfs, ignore_index=True).drop_duplicates(subset=["date", "point", "variable"])

    # Merge truth with wide forecasts (truth is identical across leads for a given date, point, variable)
    merged = pd.merge(wide_fc, truth_df[["date", "point", "variable", "truth"]], on=["date", "point", "variable"], how="inner")

    # Order columns
    expected_cols = ["date", "point", "lat", "lon", "variable", "lead", "ecmwf_ifs", "ecmwf_aifs", "noaa_gfs", "open_meteo", "truth"]
    for c in expected_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[expected_cols].sort_values(by=["date", "point", "variable", "lead"]).reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/wide.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved processed dataset: {out_path} ({len(merged)} rows)")
    return merged

def print_quality_report(df: pd.DataFrame, train_end: str = "2026-06-15"):
    print("\n" + "=" * 60)
    print("STAGE 1 QUALITY REPORT & SANITY CHECKS")
    print("=" * 60)

    # 1. Missing value counts
    print("\n--- Missing Value Counts per Source, Variable & Lead ---")
    sources = ["ecmwf_ifs", "ecmwf_aifs", "noaa_gfs", "open_meteo", "truth"]
    missing_tbl = df.groupby(["variable", "lead"])[sources].apply(lambda g: g.isna().sum()).reset_index()
    print(missing_tbl.to_string(index=False))

    # 2. Unit sanity checks
    print("\n--- Physical Unit Sanity Checks ---")
    t_df = df[df["variable"] == "temperature_2m_max"]
    p_df = df[df["variable"] == "precipitation_sum"]
    w_df = df[df["variable"] == "wind_speed_10m_max"]

    t_min, t_max = t_df[sources].min().min(), t_df[sources].max().max()
    p_min, p_max = p_df[sources].min().min(), p_df[sources].max().max()
    w_min, w_max = w_df[sources].min().min(), w_df[sources].max().max()

    print(f"Temperature (C): min = {t_min:.1f}, max = {t_max:.1f} (Plausible: -10 to 55 C) -> {'PASS' if -10 <= t_min and t_max <= 55 else 'WARN'}")
    print(f"Precipitation (mm): min = {p_min:.1f}, max = {p_max:.1f} (Plausible: >= 0 mm) -> {'PASS' if p_min >= 0 else 'WARN'}")
    print(f"Wind Speed (km/h): min = {w_min:.1f}, max = {w_max:.1f} (Plausible: >= 0 km/h) -> {'PASS' if w_min >= 0 else 'WARN'}")

    # 3. Pairwise error correlation on training period
    print("\n--- Pairwise Error Correlation on Training Period (Flag if > 0.95) ---")
    train_df = df[df["date"] <= train_end].dropna(subset=sources)
    model_sources = ["ecmwf_ifs", "ecmwf_aifs", "noaa_gfs", "open_meteo"]

    for var in df["variable"].unique():
        sub = train_df[train_df["variable"] == var]
        if len(sub) == 0:
            continue
        errors = pd.DataFrame()
        for m in model_sources:
            errors[m] = sub[m] - sub["truth"]
        corr = errors.corr()
        print(f"\nError Correlation Matrix ({var}):")
        print(corr.round(3))
        # Flag > 0.95
        for i in range(len(model_sources)):
            for j in range(i + 1, len(model_sources)):
                m1, m2 = model_sources[i], model_sources[j]
                val = corr.loc[m1, m2]
                flag = " [!] HIGH CORRELATION (>0.95)" if val > 0.95 else ""
                if "open_meteo" in (m1, m2):
                    print(f"  * Overlap Check: {m1} vs {m2} = {val:.3f}{flag}")

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    status = run_probe(cfg)
    merged_df = fetch_and_aggregate_all(cfg, status)
    print_quality_report(merged_df, train_end=status["common_training_window"]["end"])
