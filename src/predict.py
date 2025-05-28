"""Entry point — runs the full Cup-GPT prediction pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "results.csv"
MODEL_PATH = ROOT / "models" / "xgb_model.json"
REPORT_PATH = ROOT / "output" / "report.html"


def main():
    from fetch_data import fetch
    from train import train
    from simulate import run as run_simulation
    from report import generate_report

    if not DATA_CSV.exists():
        print("=== Fetching data ===")
        fetch()
    else:
        print(f"Data found at {DATA_CSV}")

    if not MODEL_PATH.exists():
        print("\n=== Training model ===")
        train()
    else:
        print(f"Model found at {MODEL_PATH}")

    print("\n=== Running simulation ===")
    run_simulation()

    print("\n=== Generating report ===")
    generate_report()

    print(f"\nDone! Open {REPORT_PATH}")


if __name__ == "__main__":
    main()
