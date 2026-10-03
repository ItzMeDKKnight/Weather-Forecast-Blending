"""
Master pipeline runner for SIH26081 Hybrid AI-NWP Multi-Model Forecast Blending System.
Executes end-to-end workflow: fetch -> blend -> evaluate -> leakage audit -> extremes.
Exits with code 0 on complete success, or non-zero with diagnostic error on failure.
"""
import sys, time, yaml
from src.fetch import run_probe, fetch_and_aggregate_all, print_quality_report
from src.blend import fit_blending_weights
from src.evaluate import evaluate_all
from src.extremes import fetch_and_generate_live_forecasts
from tests.test_leakage import run_leakage_audit

def main():
    print("=" * 80)
    print("STARTING END-TO-END SIH26081 FORECAST BLENDING PIPELINE")
    print("=" * 80)
    t0 = time.time()

    try:
        # Load configuration
        print("\n[Step 1/5] Loading config.yaml...")
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Stage 1: Probe, Fetch & Quality Report
        print("\n[Step 2/5] Running API Probe & Historical Data Pipeline...")
        status = run_probe(cfg)
        merged_df = fetch_and_aggregate_all(cfg, status)
        print_quality_report(merged_df, train_end=status["common_training_window"]["end"])

        # Stage 2: Fit Weights & Evaluate
        print("\n[Step 3/5] Fitting Adaptive Simplex Weights & Baselines...")
        fit_blending_weights(cfg)
        evaluate_all(cfg)

        # Leakage Audit Assertion
        print("\n[Step 4/5] Executing Data Leakage Audit...")
        leak_ok = run_leakage_audit()
        if not leak_ok:
            raise RuntimeError("Data leakage audit failed.")

        # Stage 3: Live Forecast & Extremes
        print("\n[Step 5/5] Fetching Live Forecasts & Computing Extreme Risks...")
        fetch_and_generate_live_forecasts(cfg)

        elapsed = time.time() - t0
        print("\n" + "=" * 80)
        print(f"PIPELINE RUN COMPLETED SUCCESSFULLY in {elapsed:.1f}s")
        print("Artifacts generated in public/:")
        print("  - public/source_status.json")
        print("  - public/weights.json")
        print("  - public/metrics.json")
        print("  - public/forecasts.json")
        print("=" * 80)
        sys.exit(0)

    except Exception as e:
        print(f"\n[!] PIPELINE FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
