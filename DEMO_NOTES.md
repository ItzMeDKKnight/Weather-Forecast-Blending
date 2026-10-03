# SIH26081: 2-Minute Demo Script & Judge Defense Cheat Sheet

This document contains a structured **2-minute live presentation script** and **prepared defense answers** for hackathon judges and evaluation panels.

---

## Part 1: The 2-Minute Live Pitch Script

### [0:00 - 0:25] The Problem & The Solution
> *"Good morning, respected judges. Weather forecasting in India faces a critical dilemma: global physics models like ECMWF IFS and NOAA GFS excel in synoptic patterns but struggle with local topography, while modern AI models like ECMWF AIFS deliver lightning-fast predictions but can suffer from regional bias. Single-source forecasts frequently fail during monsoon extremes.*
>
> *Our solution is **SIH26081**: a **Hybrid AI-NWP Multi-Model Forecast Blending System**. We dynamically fuse predictions from 4 distinct models—ECMWF IFS (Physics), ECMWF AIFS (AI), NOAA GFS (Physics), and Open-Meteo's blend—using constrained least-squares optimization with adaptive weights fitted across 12 distinct Indian climatic zones, 3 lead times, and monsoon vs. non-monsoon seasons."*

### [0:25 - 0:55] Live Map & Adaptive Weights Walkthrough
> *(Point to the Interactive Map on screen)*
> *"Here is our live dashboard. Across our 12 stations—from Himalayan foothills in Dehradun to the Western Ghats in Mahabaleshwar and the coasts of Mumbai and Chennai—you immediately notice that **no single model dominates everywhere**.
>
> When we toggle between Max Temperature, Rainfall, and Wind, the system dynamically reallocates weights. For instance, in the Western Ghats during the monsoon, physical orographic modeling carries higher weight, whereas for plains temperature, the AI model and IFS take the lead. Each point's weights strictly sum to 100% with non-negativity constraints, avoiding erratic extrapolation."*

### [0:55 - 1:25] 5-Day Forecast, Uncertainty Envelopes & IMD Extremes
> *(Switch to the Forecast & Extremes panel)*
> *"For any station, our system delivers a 5-day blended forecast accompanied by a **10th-to-90th percentile empirical confidence interval band**, calculated from real historical error distributions. Decision-makers see not just a number, but the forecast uncertainty.
>
> Below, our **Extreme Weather Hazard Panel** applies official **IMD standards**:
> 1. Heavy rainfall alert triggers at $\ge 64.5$ mm/day using square-root transformed Gaussian error estimation.
> 2. Heatwave alerts dynamically adjust to geography—triggering at $40^\circ\text{C}$ for plains, $37^\circ\text{C}$ for coastal stations, and $30^\circ\text{C}$ for hill stations, factoring in a $+4.5^\circ\text{C}$ climatology anomaly computed strictly from training truth."*

### [1:25 - 1:55] Rigorous Verification & Leakage Guarantee
> *(Scroll to the Verification Table)*
> *"Our system is scientifically honest. On our held-out 3-month test set (over 3,300 point-days), our adaptive blend achieves an **18% to 24% RMSE improvement** in maximum temperature and **14% to 16% in wind speed** over the best individual model. In 5-day rainfall, we outperform NOAA GFS by **+25.7%**.
>
> We strictly enforced temporal separation: training data ends June 15, 2026, and testing begins June 16, 2026. Our automated unit test proves **zero date leakage**."*

### [1:55 - 2:00] Deployment & Scalability
> *"The entire architecture is serverless: a GitHub Actions workflow executes `run_all.py` daily, refreshes JSON artifacts, and triggers a static deployment to Vercel. Thank you, and we welcome your questions!"*

---

## Part 2: Judge Defense & Deep-Dive Q&A

### Q1: How did you guarantee that NO data leakage occurred between training and testing?
**Answer:**
> *"We enforced strict temporal splitting rather than random cross-validation. The entire dataset was partitioned on June 15, 2026. The training set (March 1, 2025 to June 15, 2026; 472 calendar dates) was used to compute:
> 1. Day-of-year temperature climatology normals,
> 2. Model bias correction factors,
> 3. SLSQP constrained weights, and
> 4. Residual quantile distributions ($q_{10}, q_{50}, q_{90}$).
>
> The test set (June 16, 2026 to September 15, 2026; 92 calendar dates) was strictly held out until final evaluation. Our automated test script `tests/test_leakage.py` executes a mathematical set intersection check on date timestamps and asserts that `len(train_dates ∩ test_dates) == 0`. It runs automatically in our master pipeline and CI/CD."*

---

### Q2: Why did you use ERA5 as ground truth, and what is the path to official IMD data?
**Answer:**
> *"We utilized ERA5 reanalysis via Open-Meteo's open archive API to build a reproducible, globally accessible baseline without credential blockers during rapid prototyping.
>
> **Known Caveat**: We explicitly document that ECMWF AIFS was trained on ERA5 reanalysis fields, so evaluating against ERA5 gives AIFS an intrinsic advantage. Furthermore, reanalysis rainfall is derived from numerical models rather than direct surface rain gauges.
>
> **IMD Upgrade Path**: In `src/truth.py`, we designed an abstract `GroundTruthProvider` interface. Switching from ERA5 to IMD's $0.25^\circ \times 0.25^\circ$ daily gridded rainfall and temperature NetCDF datasets requires implementing a single class `IMDGriddedProvider` that inherits from `GroundTruthProvider` and reading IMD gridded files. The rest of the pipeline—weight fitting, live inference, and dashboard—requires zero changes."*

---

### Q3: Multi-model averaging is known to smooth out extremes. How does your system overcome this?
**Answer:**
> *"Standard linear ensemble averaging tends to dampen peaks, which reduces false alarms but can underestimate peak convective rainfall. We address this with three specific architectural decisions:
> 1. **Square-Root Transformation for Rainfall**: We transform rainfall into square-root space ($\sqrt{\text{rain}}$) before optimizing weights and back-transform with squaring. This stabilizes variance, handles precipitation skewness, and preserves rainfall intensity better than arithmetic averaging.
> 2. **Ensemble Spread & Predictive Uncertainty**: Instead of only outputting a point estimate, we output the 10th and 90th percentile empirical prediction intervals, reflecting the true spread and tail risk.
> 3. **Probabilistic Hazard Classification**: For extreme alerts (heavy rain $\ge 64.5$ mm, heatwaves), we evaluate exceedance probabilities using the blend value and the empirical residual error distribution. Even if the mean blend is slightly smoothed, a high ensemble spread elevates the hazard probability, warning authorities early."*

---

### Q4: What happens if a live model fails or is delayed in the daily automated run?
**Answer:**
> *"Our live ingestion in `src/extremes.py` includes a dynamic missing-source fallback and renormalization protocol:
> 1. When live forecasts are ingested, the system inspects each source for null or missing values.
> 2. If a model (e.g. GFS or AIFS) drops out or is missing from an API run, the pre-fitted weights for that point and lead time are dynamically filtered to active models only and renormalized so that $\sum w_{\text{active}} = 1.0$.
> 3. The `sources_used` metadata in `public/forecasts.json` explicitly documents the active sources utilized for that run. The dashboard never crashes and displays the calibrated blend seamlessly."*

---

### Q5: Why is Open-Meteo `best_match` included as a separate member, and how did you measure model overlap?
**Answer:**
> *"Open-Meteo's `best_match` is a widely deployed operational multi-model blend. Including it serves as an essential, transparent benchmark to prove whether our adaptive regional and seasonal weighting adds measurable skill beyond an out-of-the-box blend.
>
> Because `best_match` internally integrates ECMWF and GFS data, we explicitly tested for collinearity and error overlap during Stage 1. On our 472-day training set across all 12 points, we computed the pairwise error correlation matrix between `best_match` and each model:
> - Rainfall error correlation with IFS was **0.390**, and with GFS was **0.301**.
> - Temperature error correlation with IFS was **0.573**, and with GFS was **0.285**.
> - Wind error correlation with IFS was **0.417**, and with GFS was **0.180**.
>
> None of the correlations exceeded the $0.95$ collinearity threshold, demonstrating that `best_match` provides independent residual information that our constrained optimizer successfully utilizes."*
