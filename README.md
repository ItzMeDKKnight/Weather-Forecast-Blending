# SIH26081: Hybrid AI-NWP Multi-Model Forecast Blending System

An operational, serverless forecasting engine and real-time dashboard designed for **Smart India Hackathon (SIH 2024 / Problem SIH26081)**. The system dynamically blends weather predictions from physics-based NWP models and state-of-the-art AI models into one optimized forecast with spatially, seasonally, and lead-time adaptive weights across 12 distinct Indian climatic zones.

---

## Architecture Overview

```
                      +-------------------------------------------------------+
                      |               Open-Meteo Free APIs                    |
                      | (Previous Runs API, Live Forecast API, ERA5 Archive)  |
                      +---------------------------+---------------------------+
                                                  |
                                                  v
                                       +---------------------+
                                       |     src/fetch.py    |
                                       |  (Disk Cache & IST  |
                                       |  Daily Aggregation) |
                                       +----------+----------+
                                                  |
                                                  v
                                       +---------------------+
                                       |     src/truth.py    |
                                       | (GroundTruthProvider|
                                       |     ERA5 / IMD)     |
                                       +----------+----------+
                                                  |
                                                  v
                                       +---------------------+
                                       | data/processed/     |
                                       | wide.parquet (60k+) |
                                       +----------+----------+
                                                  |
                     +----------------------------+---------------------------+
                     |                                                        |
                     v                                                        v
          +---------------------+                                  +---------------------+
          |     src/blend.py    |                                  |   src/evaluate.py   |
          | (Simplex SLSQP Fit, |                                  | (Held-Out Test RMSE,|
          |  Sqrt Rain, Bias)   |                                  |  MAE, Bias, FrqBias)|
          +----------+----------+                                  +----------+----------+
                     |                                                        |
                     +----------------------------+---------------------------+
                                                  |
                                                  v
                                       +---------------------+
                                       |   src/extremes.py   |
                                       | (Live 5-Day Blend,  |
                                       | 10/50/90 Intervals, |
                                       | IMD Extreme Probs)  |
                                       +----------+----------+
                                                  |
                                                  v
                                       +---------------------+
                                       |    public/*.json    |
                                       |  (weights, metrics, |
                                       |  forecasts, status) |
                                       +----------+----------+
                                                  |
                                                  v
                                       +---------------------+
                                       |  public/index.html  |
                                       | (Zero-Build Leaflet |
                                       |  & Chart.js UI)     |
                                       +---------------------+
```

---

## Integrated Weather Sources

All meteorological data is sourced through Open-Meteo free API endpoints (zero API keys required, rate-limited and cached locally):

| Source Name | Model Identifier | Model Category | Description |
|---|---|---|---|
| **ECMWF IFS** | `ecmwf_ifs025` | Physics NWP | High-resolution global atmospheric model ($0.25^\circ$) |
| **ECMWF AIFS** | `ecmwf_aifs025_single` | AI / Deep Learning | Data-driven AI global forecast model ($0.25^\circ$) |
| **NOAA GFS** | `gfs_global` | Physics NWP | US National Weather Service Global Forecast System |
| **Open-Meteo** | `best_match` | Statistical Blend | Open-Meteo's internal operational multi-model blend |
| **Ground Truth** | `ERA5` (via archive API) | Reanalysis / Truth | ECMWF 5th Generation Reanalysis (modularized interface) |

> **Operational Window Note**: `ecmwf_ifs025`, `gfs_global`, and `best_match` have continuous history from 2024. `ecmwf_aifs025_single` operational history begins **February 20, 2025**. The automated probe in `src/fetch.py` detects this and aligns the common multi-model training window to **2025-03-01 through 2026-06-15**.

---

## 12 Indian Climatic Observation Points

The network represents the full geographical and meteorological diversity of India:

1. **Dehradun**: Himalayan foothills (Hill station terrain)
2. **Delhi**: Indo-Gangetic plain north (Plains terrain)
3. **Patna**: Indo-Gangetic plain east (Plains terrain)
4. **Bhopal**: Central India plateau (Plains terrain)
5. **Mahabaleshwar**: Western Ghats highland (Hill station terrain)
6. **Chennai**: East coast south (Coastal terrain)
7. **Bhubaneswar**: East coast central (Coastal terrain)
8. **Guwahati**: Northeast Brahmaputra valley (Plains terrain)
9. **Jodhpur**: Arid Thar desert west (Plains terrain)
10. **Hyderabad**: Deccan interior (Plains/Plateau terrain)
11. **Mumbai**: West coast Arabian Sea metro (Coastal terrain)
12. **Bengaluru**: South interior plateau ($\approx 920$m elevation)

All daily statistics use the **Asia/Kolkata (IST)** calendar day boundary (00:00 to 23:00 IST).

---

## Verification & Skill Benchmark Table

Evaluated on the strict held-out test set (last 3 months: **June 16, 2026 to September 15, 2026**, 92 calendar days, 3,312 point-days), strictly isolated from training data:

| Weather Variable | Lead Time | Best Single Model | RMSE Best Single | RMSE Equal Weight | RMSE Adaptive Blend | Improvement vs Best Single | Improvement vs Equal Weight |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Max Temperature** | 1d | ECMWF IFS | 1.09 °C | 1.07 °C | **0.89 °C** | **+18.4%** | **+16.5%** |
| **Max Temperature** | 3d | ECMWF IFS | 1.40 °C | 1.29 °C | **1.09 °C** | **+22.5%** | **+15.7%** |
| **Max Temperature** | 5d | ECMWF IFS | 1.64 °C | 1.46 °C | **1.25 °C** | **+23.6%** | **+14.4%** |
| **Total Rainfall** | 1d | ECMWF AIFS | 12.17 mm | 13.19 mm | **12.56 mm** | -3.3% | **+4.7%** |
| **Total Rainfall** | 3d | ECMWF AIFS | 13.51 mm | 14.78 mm | **14.21 mm** | -5.2% | **+3.9%** |
| **Total Rainfall** | 5d | NOAA GFS | 21.50 mm | 16.79 mm | **15.98 mm** | **+25.7%** | **+4.8%** |
| **Max Wind Speed** | 1d | ECMWF IFS | 2.89 km/h | 2.92 km/h | **2.47 km/h** | **+14.4%** | **+15.2%** |
| **Max Wind Speed** | 3d | ECMWF IFS | 3.42 km/h | 3.29 km/h | **2.86 km/h** | **+16.4%** | **+13.1%** |
| **Max Wind Speed** | 5d | ECMWF IFS | 3.66 km/h | 3.46 km/h | **3.12 km/h** | **+14.8%** | **+9.8%** |

### Extreme Weather Event Skill (Held-Out Test Set)

| Hazard Event | Test Samples | Probability of Detection (POD) | False Alarm Ratio (FAR) | Critical Success Index (CSI) | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Heavy Rain** ($\ge 64.5$ mm) | 3,312 | 0.493 | 0.443 | 0.354 | 0.0157 |
| **High Wind** ($\ge 40$ km/h) | 3,312 | 0.000 (No Gale Events in Test) | 0.000 | 0.000 | 0.0000 |
| **Heatwave** (IMD Criteria) | 3,312 | 1.000 | 0.700 | 0.300 | 0.0061 |

---

## Data Leakage Prevention Guarantee

Data leakage is formally prevented through:
1. **Strict Temporal Partitioning**: Training ends on `2026-06-15`; testing strictly begins on `2026-06-16`.
2. **No Shared Parameters**: Climatologies, weights, bias correction factors, and residual quantiles are fitted purely on training indices.
3. **Automated Unit Audit**: `tests/test_leakage.py` asserts zero set intersection between training and test dates and confirms parameter timestamps.

Run the test:
```bash
python -m tests.test_leakage
```
Output:
```
LEAKAGE AUDIT: PASSED
  Training period: 2025-03-01 to 2026-06-15 (472 distinct dates)
  Test period:     2026-06-16 to 2026-09-15 (92 distinct dates)
  Shared dates:    0 (Strict temporal separation confirmed)
```

---

## Step-by-Step Reproduction Guide

### 1. Prerequisites & Setup
Clone the repository and install pinned dependencies:
```bash
git clone https://github.com/<your-username>/sih26081.git
cd sih26081
pip install -r requirements.txt
```

### 2. Stage-by-Stage Execution
- **Stage 1 (Data Probe & Historical Download)**:
  ```bash
  python -m src.fetch
  ```
  Generates `public/source_status.json` and `data/processed/wide.parquet`.

- **Stage 2 (Adaptive Blending & Test Set Evaluation)**:
  ```bash
  python -m src.blend
  python -m src.evaluate
  ```
  Generates `public/weights.json`, `public/metrics.json`, and runs the leakage audit.

- **Stage 3 (Live 5-Day Forecast & Extreme Hazard Probabilities)**:
  ```bash
  python -m src.extremes
  ```
  Generates `public/forecasts.json`.

### 3. Master End-to-End Run
To run the entire pipeline with a single command:
```bash
python -m src.run_all
```

### 4. Stage 4: Run the Interactive Dashboard Locally
Browsers block local `fetch()` calls when opening raw HTML files directly from disk. Serve the `public/` directory with Python's built-in static server:
```bash
python -m http.server -d public 8000
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## Deployment to Vercel (Static Hosting)

Because this system requires zero server-side runtime, the `public/` folder deploys seamlessly as a fast, global static website on Vercel.

### Method A: Deploy via Vercel CLI
```bash
# Install Vercel CLI if not present
npm i -g vercel

# Deploy directly from repository root (reads vercel.json outputDirectory: "public")
vercel --prod
```

### Method B: Git Integration via Vercel Dashboard
1. Push repository to GitHub.
2. Go to [vercel.com/new](https://vercel.com/new) and import the repository.
3. In Build and Output Settings:
   - **Framework Preset**: Other
   - **Output Directory**: `public`
4. Click **Deploy**.

The preconfigured [`vercel.json`](vercel.json) automatically routes requests to `public/`.

---

## Automated Daily Workflow (GitHub Actions)

The scheduled pipeline [`.github/workflows/daily_update.yml`](.github/workflows/daily_update.yml) runs automatically every day at **06:00 UTC (11:30 AM IST)**:
1. Checks out the repository and installs pinned dependencies.
2. Executes `python -m src.run_all` to ingest the latest live forecasts, re-evaluate weights, and compute updated hazard probabilities.
3. Strictly commits only `public/*.json` artifacts back to the repository (never committing cached files or Parquet datasets).
4. Pushing to `main` automatically triggers Vercel static redeployment.

---

## Known Scientific Limitations & Transparency

1. **ERA5 Reanalysis as Proxy Truth**:
   - ERA5 reanalysis was used as the ground truth benchmark because gridded IMD observations require restricted credentials.
   - **AI Training Favoritism**: ECMWF AIFS was explicitly trained on ERA5 historical fields; consequently, evaluating on ERA5 may slightly favor AIFS relative to operational weather stations.
2. **Precipitation Uncertainty**:
   - Reanalysis precipitation represents numerical model physics rather than direct rain-gauge observations. Local convective thunderstorm cells may be underestimated in reanalysis.
3. **Convective Peak Smoothing**:
   - Multi-model ensemble averaging mathematically dampens variance. While this substantially cuts false alarms for heavy rainfall, it slightly compresses extreme convective rain peaks.
4. **Model Overlap**:
   - Open-Meteo's `best_match` model incorporates ECMWF and GFS data. Pairwise error correlation was continuously tracked during Stage 1 (measured at $0.18$ to $0.57$, well below the $0.95$ threshold), confirming distinct residual behavior.

---

## How to Plug in IMD Gridded Observations

The system abstracts ground truth behind the `GroundTruthProvider` interface in [`src/truth.py`](src/truth.py):

```python
from src.truth import GroundTruthProvider
import pandas as pd

class IMDGriddedProvider(GroundTruthProvider):
    def __init__(self, imd_netcdf_dir: str):
        self.data_dir = imd_netcdf_dir

    def fetch_daily(self, points: dict, start_date: str, end_date: str, cache_dir: str = "data/cache") -> pd.DataFrame:
        # 1. Read IMD 0.25x0.25 degree daily NetCDF files (rainfall & temperature)
        # 2. Extract nearest grid-cell values for points in points dict
        # 3. Return DataFrame with: [date, point, lat, lon, variable, truth]
        ...
```
Replacing `ERA5Provider()` with `IMDGriddedProvider()` requires changing exactly one line in `src/fetch.py`.
