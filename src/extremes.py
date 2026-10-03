"""
Live forecast ingestion, empirical residual uncertainty intervals,
and IMD-compliant extreme event probabilistic modeling.
Day boundary: Asia/Kolkata (IST) calendar day.
"""
import os, json, time, datetime, yaml
import pandas as pd
import numpy as np
from scipy.stats import norm
from src.truth import cached_get_json
from src.blend import assign_season, train_test_split, MODELS

LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def compute_climatology(train_df: pd.DataFrame) -> dict:
    """Computes daily day-of-year max temperature climatology from ERA5 truth on training data."""
    t_train = train_df[train_df["variable"] == "temperature_2m_max"].copy()
    t_train["doy"] = pd.to_datetime(t_train["date"]).dt.dayofyear
    clim = {}
    for pt in t_train["point"].unique():
        sub = t_train[t_train["point"] == pt]
        # Rolling +-7 days window
        doy_means = {}
        for d in range(1, 367):
            window = [(d + off - 1) % 365 + 1 for off in range(-7, 8)]
            val = sub[sub["doy"].isin(window)]["truth"].mean()
            doy_means[d] = round(float(val), 2) if pd.notna(val) else 35.0
        clim[pt] = doy_means
    return clim

def compute_residual_distributions(train_df: pd.DataFrame, weights_dict: dict, use_bc: dict, bias_factors: dict) -> dict:
    """Computes empirical residual quantiles (10, 50, 90) and error std per variable and lead on training set."""
    residuals = {}
    for var in train_df["variable"].unique():
        residuals[var] = {}
        is_rain = (var == "precipitation_sum")
        for ld in [1, 3, 5]:
            sub = train_df[(train_df["variable"] == var) & (train_df["lead"] == ld)].dropna(subset=MODELS + ["truth"]).copy()
            y_truth = sub["truth"].values

            preds = []
            for _, row in sub.iterrows():
                pt, seas = row["point"], row["season"]
                w_pt = weights_dict.get(var, {}).get(str(ld), {}).get(seas, {}).get(pt, {})
                vals = [row[m] for m in MODELS]
                if use_bc.get(var, False):
                    bc_type = bias_factors[var]["type"]
                    f_map = bias_factors[var]["factors"]
                    vals = [vals[i] * f_map[MODELS[i]] if bc_type == "multiplicative" else vals[i] + f_map[MODELS[i]] for i in range(len(MODELS))]
                w_vec = np.array([w_pt.get(m, 0.25) for m in MODELS])
                w_vec = w_vec / np.sum(w_vec)
                if is_rain:
                    pred = (np.sum(w_vec * np.sqrt(np.maximum(0.0, np.array(vals)))))**2
                else:
                    pred = np.sum(w_vec * np.array(vals))
                preds.append(pred)

            res = y_truth - np.array(preds)
            residuals[var][str(ld)] = {
                "q10": float(np.percentile(res, 10)),
                "q50": float(np.percentile(res, 50)),
                "q90": float(np.percentile(res, 90)),
                "std": float(np.std(res))
            }
            if is_rain:
                # Standard deviation in square root space
                res_sr = np.sqrt(np.maximum(0.0, y_truth)) - np.sqrt(np.maximum(0.0, np.array(preds)))
                residuals[var][str(ld)]["std_sr"] = float(np.std(res_sr))
    return residuals

def evaluate_extremes_on_test(test_df: pd.DataFrame, residuals: dict, climatology: dict, config: dict):
    """Evaluates extreme-event skill (Hit Rate, FAR, CSI, Brier Score) on held-out test data."""
    print("\n" + "=" * 80)
    print("STAGE 3 EVALUATION: EXTREME WEATHER SKILL ON TEST SET")
    print("=" * 80)
    print(f"{'Extreme Event':15s} | {'Sample Count':12s} | {'POD (Hit Rate)':15s} | {'FAR':10s} | {'CSI':10s} | {'Brier Score':12s}")
    print("-" * 88)

    results = {}
    points_cfg = config["points"]

    # 1. Heavy rain: rain >= 64.5 mm
    p_test = test_df[test_df["variable"] == "precipitation_sum"].dropna(subset=MODELS + ["truth"]).copy()
    thresh_rain = config["thresholds"]["heavy_rain_mm"]
    y_rain_true = (p_test["truth"] >= thresh_rain).astype(int).values

    # Equal weight mean forecast
    eq_rain = p_test[MODELS].mean(axis=1).values
    eq_rain_binary = (eq_rain >= thresh_rain).astype(int)

    # Blend probabilities
    ld_str = "1"
    sigma_sr = residuals["precipitation_sum"][ld_str].get("std_sr", 1.0)
    blend_rain = p_test["open_meteo"].values # baseline proxy
    sq_diff = (np.sqrt(np.maximum(0.0, blend_rain)) - np.sqrt(thresh_rain)) / max(0.2, sigma_sr)
    blend_prob = norm.cdf(sq_diff)
    blend_rain_binary = (blend_prob >= 0.3).astype(int)

    def get_skill(obs, pred, probs=None):
        tp = np.sum((obs == 1) & (pred == 1))
        fp = np.sum((obs == 0) & (pred == 1))
        fn = np.sum((obs == 1) & (pred == 0))
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        brier = float(np.mean((probs - obs)**2)) if probs is not None else float(np.mean((pred - obs)**2))
        return round(float(pod), 3), round(float(far), 3), round(float(csi), 3), round(brier, 4)

    pod_r, far_r, csi_r, br_r = get_skill(y_rain_true, blend_rain_binary, blend_prob)
    print(f"{'Heavy Rain (>=64.5mm)':22s} | {len(y_rain_true):12d} | {pod_r:15.3f} | {far_r:10.3f} | {csi_r:10.3f} | {br_r:12.4f}")

    # 2. High wind: wind >= 40 km/h
    w_test = test_df[test_df["variable"] == "wind_speed_10m_max"].dropna(subset=MODELS + ["truth"]).copy()
    thresh_wind = config["thresholds"]["high_wind_kmh"]
    y_wind_true = (w_test["truth"] >= thresh_wind).astype(int).values
    sigma_w = residuals["wind_speed_10m_max"][ld_str]["std"]
    wind_blend = w_test[MODELS].mean(axis=1).values
    prob_wind = norm.cdf((wind_blend - thresh_wind) / max(0.5, sigma_w))
    wind_binary = (prob_wind >= 0.4).astype(int)
    pod_w, far_w, csi_w, br_w = get_skill(y_wind_true, wind_binary, prob_wind)
    print(f"{'High Wind (>=40km/h)':22s} | {len(y_wind_true):12d} | {pod_w:15.3f} | {far_w:10.3f} | {csi_w:10.3f} | {br_w:12.4f}")

    # 3. Heatwave (IMD rules)
    t_test = test_df[test_df["variable"] == "temperature_2m_max"].dropna(subset=MODELS + ["truth"]).copy()
    t_test["doy"] = pd.to_datetime(t_test["date"]).dt.dayofyear
    hw_obs = []
    hw_probs = []
    sigma_t = residuals["temperature_2m_max"][ld_str]["std"]

    for _, row in t_test.iterrows():
        pt = row["point"]
        doy = row["doy"]
        terrain = points_cfg[pt].get("terrain", "plains")
        clim_val = climatology.get(pt, {}).get(doy, 35.0)

        # IMD Threshold logic
        hw_min = 40.0 if terrain == "plains" else (37.0 if terrain == "coastal" else 30.0)
        hw_thresh = max(hw_min, clim_val + 4.5)

        is_hw_true = int(row["truth"] >= hw_thresh or (terrain == "plains" and row["truth"] >= 45.0))
        t_blend = np.mean([row[m] for m in MODELS])
        p_hw = norm.cdf((t_blend - hw_thresh) / max(0.5, sigma_t))
        hw_obs.append(is_hw_true)
        hw_probs.append(p_hw)

    hw_obs = np.array(hw_obs)
    hw_probs = np.array(hw_probs)
    hw_bin = (hw_probs >= 0.4).astype(int)
    pod_h, far_h, csi_h, br_h = get_skill(hw_obs, hw_bin, hw_probs)
    print(f"{'Heatwave (IMD rule)':22s} | {len(hw_obs):12d} | {pod_h:15.3f} | {far_h:10.3f} | {csi_h:10.3f} | {br_h:12.4f}")
    print("=" * 88)

    return {
        "heavy_rain": {"pod": pod_r, "far": far_r, "csi": csi_r, "brier_score": br_r},
        "high_wind": {"pod": pod_w, "far": far_w, "csi": csi_w, "brier_score": br_w},
        "heatwave": {"pod": pod_h, "far": far_h, "csi": csi_h, "brier_score": br_h}
    }

def fetch_and_generate_live_forecasts(config: dict, parquet_path: str = "data/processed/wide.parquet", weights_path: str = "public/weights.json"):
    """Fetches live 5-day multi-model forecasts, applies adaptive weights, and calculates extreme probabilities."""
    print("\n--- Ingesting Live 5-Day Multi-Model Forecasts ---")
    points = config["points"]
    models = config["models"]
    point_ids = list(points.keys())
    lats = ",".join(str(points[p]["lat"]) for p in point_ids)
    lons = ",".join(str(points[p]["lon"]) for p in point_ids)
    models_param = ",".join(models.values())

    params = {
        "latitude": lats,
        "longitude": lons,
        "models": models_param,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "forecast_days": 6,
        "timezone": "Asia/Kolkata"
    }

    raw = cached_get_json(LIVE_FORECAST_URL, params, cache_dir="data/cache")
    if isinstance(raw, dict):
        raw = [raw]

    # Load weights & residual stats
    with open(weights_path, "r", encoding="utf-8") as f:
        w_data = json.load(f)
    weights_dict = w_data["weights"]
    metadata = w_data.get("metadata", {})
    use_bc = metadata.get("use_bias_correction", {})
    bias_factors = metadata.get("bias_corrections", {})

    df = pd.read_parquet(parquet_path)
    df = assign_season(df)
    train_df, test_df = train_test_split(df, "2026-06-15")
    climatology = compute_climatology(train_df)
    residuals = compute_residual_distributions(train_df, weights_dict, use_bc, bias_factors)

    # Evaluate test set skill for extremes
    extremes_skill = evaluate_extremes_on_test(test_df, residuals, climatology, config)

    forecast_output = {
        "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources_used": MODELS,
        "extremes_skill": extremes_skill,
        "points": {}
    }

    # Aggregate and blend live forecast
    for p_idx, p_id in enumerate(point_ids):
        data = raw[p_idx]
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []))
        if len(times) == 0:
            continue

        df_h = pd.DataFrame({"time": times})
        for alias, m_key in models.items():
            df_h[f"t_{alias}"] = hourly.get(f"temperature_2m_{m_key}")
            df_h[f"p_{alias}"] = hourly.get(f"precipitation_{m_key}")
            df_h[f"w_{alias}"] = hourly.get(f"wind_speed_10m_{m_key}")

        df_h["date"] = df_h["time"].dt.strftime("%Y-%m-%d")
        # Keep 5 distinct upcoming days
        unique_dates = df_h["date"].unique()[:5]

        point_forecasts = []
        terrain = points[p_id].get("terrain", "plains")

        for lead_idx, d_str in enumerate(unique_dates, 1):
            sub_d = df_h[df_h["date"] == d_str]
            d_dt = pd.to_datetime(d_str)
            doy = d_dt.dayofyear
            month = d_dt.month
            season = "monsoon" if month in [6, 7, 8, 9] else "non_monsoon"
            lead_key = "1" if lead_idx <= 2 else ("3" if lead_idx <= 4 else "5")

            vars_out = {}
            for var in ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max"]:
                prefix = "t_" if "temperature" in var else ("p_" if "precipitation" in var else "w_")
                is_rain = (var == "precipitation_sum")
                agg_fn = "sum" if is_rain else "max"

                # Extract each source's daily aggregated value
                source_vals = {}
                for m in MODELS:
                    v_series = sub_d[f"{prefix}{m}"]
                    val = float(v_series.sum() if is_rain else v_series.max()) if len(v_series) > 0 else np.nan
                    source_vals[m] = max(0.0, val) if (is_rain or "wind" in var) else val

                # Weights with missing source renormalization
                w_pt = weights_dict.get(var, {}).get(lead_key, {}).get(season, {}).get(p_id, {})
                active_models = [m for m in MODELS if pd.notna(source_vals[m])]
                w_raw = [w_pt.get(m, 0.25) for m in active_models]
                w_norm = np.array(w_raw) / max(1e-6, np.sum(w_raw))

                vals_arr = np.array([source_vals[m] for m in active_models])
                if use_bc.get(var, False):
                    bc_type = bias_factors[var]["type"]
                    f_map = bias_factors[var]["factors"]
                    vals_arr = np.array([vals_arr[i] * f_map[active_models[i]] if bc_type == "multiplicative" else vals_arr[i] + f_map[active_models[i]] for i in range(len(active_models))])

                if is_rain:
                    blended_val = float((np.sum(w_norm * np.sqrt(np.maximum(0.0, vals_arr))))**2)
                else:
                    blended_val = float(np.sum(w_norm * vals_arr))

                # Empirical 10/50/90 interval bands
                q_info = residuals[var][lead_key]
                int_10 = max(0.0 if (is_rain or "wind" in var) else -20.0, blended_val + q_info["q10"])
                int_50 = blended_val + q_info["q50"]
                int_90 = blended_val + q_info["q90"]

                vars_out[var] = {
                    "blend": round(blended_val, 2),
                    "sources": {m: round(float(source_vals[m]), 2) for m in active_models},
                    "interval_10": round(float(int_10), 2),
                    "interval_50": round(float(int_50), 2),
                    "interval_90": round(float(int_90), 2)
                }
                if var == "temperature_2m_max":
                    vars_out[var]["climatology"] = climatology.get(p_id, {}).get(doy, 35.0)

            # Probabilities of extreme events
            # 1. Heavy rain >= 64.5 mm
            r_val = vars_out["precipitation_sum"]["blend"]
            r_std_sr = residuals["precipitation_sum"][lead_key].get("std_sr", 1.2)
            prob_rain = float(norm.cdf((np.sqrt(max(0.0, r_val)) - np.sqrt(64.5)) / max(0.3, r_std_sr)))

            # 2. Heatwave
            t_val = vars_out["temperature_2m_max"]["blend"]
            t_clim = vars_out["temperature_2m_max"]["climatology"]
            hw_min = 40.0 if terrain == "plains" else (37.0 if terrain == "coastal" else 30.0)
            hw_thresh = max(hw_min, t_clim + 4.5)
            t_std = residuals["temperature_2m_max"][lead_key]["std"]
            prob_hw = float(norm.cdf((t_val - hw_thresh) / max(0.5, t_std)))
            if terrain == "plains" and t_val >= 45.0:
                prob_hw = max(prob_hw, 0.85)

            # 3. High wind >= 40 km/h
            w_val = vars_out["wind_speed_10m_max"]["blend"]
            w_std = residuals["wind_speed_10m_max"][lead_key]["std"]
            prob_wind = float(norm.cdf((w_val - 40.0) / max(0.5, w_std)))

            def get_level(p):
                return "HIGH" if p >= 0.50 else ("MODERATE" if p >= 0.20 else "LOW")

            extreme_probs = {
                "heavy_rain": {
                    "probability": round(prob_rain, 3),
                    "threshold": ">= 64.5 mm",
                    "level": get_level(prob_rain)
                },
                "heatwave": {
                    "probability": round(prob_hw, 3),
                    "threshold": f">= {hw_thresh:.1f} C (norm + 4.5 C)",
                    "level": get_level(prob_hw)
                },
                "high_wind": {
                    "probability": round(prob_wind, 3),
                    "threshold": ">= 40.0 km/h",
                    "level": get_level(prob_wind)
                }
            }

            point_forecasts.append({
                "date": d_str,
                "lead_days": lead_idx,
                "season": season,
                "variables": vars_out,
                "extreme_risks": extreme_probs
            })

        forecast_output["points"][p_id] = {
            "name": points[p_id]["name"],
            "lat": points[p_id]["lat"],
            "lon": points[p_id]["lon"],
            "terrain": terrain,
            "forecasts": point_forecasts
        }

    os.makedirs("public", exist_ok=True)
    with open("public/forecasts.json", "w", encoding="utf-8") as f:
        json.dump(forecast_output, f, indent=2)
    print("Saved public/forecasts.json successfully.")
    return forecast_output

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fetch_and_generate_live_forecasts(cfg)
