"""
Multi-model forecast blending engine with constrained adaptive weights.
Fits non-negative weights (summing to 1) per variable, lead, season, and point.
Uses square-root space for rainfall. Evaluates bias correction on training validation tail.
"""
import os, json, yaml
import pandas as pd
import numpy as np
from scipy.optimize import minimize

MODELS = ["ecmwf_ifs", "ecmwf_aifs", "noaa_gfs", "open_meteo"]

def fit_simplex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Finds weights w >= 0 summing to 1 minimizing ||Xw - y||^2 using SLSQP."""
    m = X.shape[1]
    w0 = np.ones(m) / m
    def obj(w):
        res = X @ w - y
        return 0.5 * np.dot(res, res)
    def grad(w):
        return X.T @ (X @ w - y)

    bounds = [(0.0, 1.0) for _ in range(m)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(m)}
    res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success and np.all(res.x >= -1e-5):
        w = np.clip(res.x, 0.0, 1.0)
        return w / np.sum(w)
    return w0

def train_test_split(df: pd.DataFrame, train_end: str = "2026-06-15"):
    train = df[df["date"] <= train_end].copy()
    test = df[df["date"] > train_end].copy()
    # Leakage assertion: Zero shared dates
    overlap = set(train["date"]).intersection(set(test["date"]))
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: shared dates {overlap}"
    return train, test

def assign_season(df: pd.DataFrame) -> pd.DataFrame:
    months = pd.to_datetime(df["date"]).dt.month
    df["season"] = np.where(months.isin([6, 7, 8, 9]), "monsoon", "non_monsoon")
    return df

def fit_blending_weights(config: dict, parquet_path: str = "data/processed/wide.parquet") -> dict:
    df = pd.read_parquet(parquet_path)
    df = assign_season(df)
    train_end = "2026-06-15"
    train_df, test_df = train_test_split(df, train_end)

    points_cfg = config["points"]
    point_to_region = {p: points_cfg[p]["region"] for p in points_cfg}
    train_df["region"] = train_df["point"].map(point_to_region)

    # Validation tail inside training (last 30 days of train: 2026-05-16 to 2026-06-15)
    val_split_date = "2026-05-15"
    fit_train = train_df[train_df["date"] <= val_split_date]
    val_train = train_df[train_df["date"] > val_split_date]

    # Evaluate bias correction on validation tail
    print("\n--- Evaluating Bias Correction on Training Validation Tail ---")
    bias_corrections = {}
    use_bias_correction = {}

    for var in config["variables"].keys():
        sub_fit = fit_train[fit_train["variable"] == var].dropna(subset=MODELS + ["truth"])
        sub_val = val_train[val_train["variable"] == var].dropna(subset=MODELS + ["truth"])

        if var == "precipitation_sum":
            # Multiplicative bias
            ratios = {}
            for m in MODELS:
                truth_mean = sub_fit["truth"].mean() + 1e-3
                m_mean = sub_fit[m].mean() + 1e-3
                ratios[m] = float(np.clip(truth_mean / m_mean, 0.5, 2.0))
            bias_corrections[var] = {"type": "multiplicative", "factors": ratios}

            # Check raw vs corrected RMSE on validation tail
            raw_err = [np.mean((sub_val[m] - sub_val["truth"])**2) for m in MODELS]
            corr_err = [np.mean((sub_val[m] * ratios[m] - sub_val["truth"])**2) for m in MODELS]
            use_bc = np.mean(corr_err) < np.mean(raw_err)
        else:
            # Additive bias
            offsets = {}
            for m in MODELS:
                offsets[m] = float(np.clip((sub_fit["truth"] - sub_fit[m]).mean(), -5.0, 5.0))
            bias_corrections[var] = {"type": "additive", "factors": offsets}

            raw_err = [np.mean((sub_val[m] - sub_val["truth"])**2) for m in MODELS]
            corr_err = [np.mean((sub_val[m] + offsets[m] - sub_val["truth"])**2) for m in MODELS]
            use_bc = np.mean(corr_err) < np.mean(raw_err)

        use_bias_correction[var] = bool(use_bc)
        status_str = "ENABLED" if use_bc else "DISABLED (raw performed better)"
        print(f"  {var:22s}: Bias correction {status_str}")

    # Now fit final weights on entire training set
    print("\n--- Fitting Regional & Point Adaptive Weights ---")
    weights_dict = {}
    fallbacks_count = 0

    for var in config["variables"].keys():
        weights_dict[var] = {}
        is_rain = (var == "precipitation_sum")
        bc = bias_corrections[var] if use_bias_correction[var] else None

        for ld in config["leads"]:
            weights_dict[var][str(ld)] = {}
            for season in ["monsoon", "non_monsoon"]:
                weights_dict[var][str(ld)][season] = {}

                # Global pooled data for fallback
                glob_sub = train_df[(train_df["variable"] == var) & (train_df["lead"] == ld) & (train_df["season"] == season)].dropna(subset=MODELS + ["truth"])
                if len(glob_sub) > 0:
                    X_glob = glob_sub[MODELS].values.copy()
                    y_glob = glob_sub["truth"].values.copy()
                    if bc:
                        for idx, m in enumerate(MODELS):
                            X_glob[:, idx] = X_glob[:, idx] * bc["factors"][m] if bc["type"] == "multiplicative" else X_glob[:, idx] + bc["factors"][m]
                    if is_rain:
                        X_glob = np.sqrt(np.maximum(0.0, X_glob))
                        y_glob = np.sqrt(np.maximum(0.0, y_glob))
                    glob_w = fit_simplex_weights(X_glob, y_glob)
                else:
                    glob_w = np.ones(len(MODELS)) / len(MODELS)

                for p_id in points_cfg.keys():
                    sub = train_df[(train_df["variable"] == var) & (train_df["lead"] == ld) &
                                   (train_df["season"] == season) & (train_df["point"] == p_id)].dropna(subset=MODELS + ["truth"])

                    if len(sub) >= 20:
                        X = sub[MODELS].values.copy()
                        y = sub["truth"].values.copy()
                        if bc:
                            for idx, m in enumerate(MODELS):
                                X[:, idx] = X[:, idx] * bc["factors"][m] if bc["type"] == "multiplicative" else X[:, idx] + bc["factors"][m]
                        if is_rain:
                            X = np.sqrt(np.maximum(0.0, X))
                            y = np.sqrt(np.maximum(0.0, y))
                        w = fit_simplex_weights(X, y)
                    else:
                        # Fallback to regional or global
                        reg = point_to_region[p_id]
                        reg_sub = train_df[(train_df["variable"] == var) & (train_df["lead"] == ld) &
                                           (train_df["season"] == season) & (train_df["region"] == reg)].dropna(subset=MODELS + ["truth"])
                        if len(reg_sub) >= 20:
                            X_reg = reg_sub[MODELS].values.copy()
                            y_reg = reg_sub["truth"].values.copy()
                            if bc:
                                for idx, m in enumerate(MODELS):
                                    X_reg[:, idx] = X_reg[:, idx] * bc["factors"][m] if bc["type"] == "multiplicative" else X_reg[:, idx] + bc["factors"][m]
                            if is_rain:
                                X_reg = np.sqrt(np.maximum(0.0, X_reg))
                                y_reg = np.sqrt(np.maximum(0.0, y_reg))
                            w = fit_simplex_weights(X_reg, y_reg)
                        else:
                            w = glob_w
                        fallbacks_count += 1

                    weights_dict[var][str(ld)][season][p_id] = {
                        MODELS[i]: round(float(w[i]), 4) for i in range(len(MODELS))
                    }

    print(f"Adaptive weights fitted. (Fallback hierarchy triggered {fallbacks_count} times).")

    output_payload = {
        "metadata": {
            "train_end": train_end,
            "models": MODELS,
            "bias_corrections": bias_corrections,
            "use_bias_correction": use_bias_correction
        },
        "weights": weights_dict
    }

    os.makedirs("public", exist_ok=True)
    with open("public/weights.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print("Saved public/weights.json successfully.")
    return output_payload

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fit_blending_weights(cfg)
