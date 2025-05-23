"""Dataset acquisition and validation for World Cup 2026 prediction."""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)
RESULTS_FILE = DATA_DIR / "results.csv"


def download_with_retry(url: str, dest: Path, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return False


def validate_csv(path: Path) -> bool:
    if not path.exists():
        print(f"Error: {path} does not exist")
        return False
    if path.stat().st_size == 0:
        print(f"Error: {path} is empty")
        return False
    try:
        df = pd.read_csv(path, nrows=5)
        if df.empty:
            print(f"Error: {path} contains no data rows")
            return False
        required = {"date", "home_team", "away_team", "home_score", "away_score"}
        missing = required - set(df.columns)
        if missing:
            print(f"Error: {path} missing columns: {missing}")
            return False
        return True
    except pd.errors.ParserError as e:
        print(f"Error: {path} is not a valid CSV — {e}")
        return False


def fetch() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_FILE.exists() and validate_csv(RESULTS_FILE):
        print(f"Dataset already exists at {RESULTS_FILE}")
        return RESULTS_FILE

    print(f"Downloading dataset from {RESULTS_URL} ...")
    if not download_with_retry(RESULTS_URL, RESULTS_FILE):
        print("Error: failed to download dataset after all retries")
        sys.exit(1)

    if not validate_csv(RESULTS_FILE):
        print("Error: downloaded file failed validation")
        sys.exit(1)

    size_kb = RESULTS_FILE.stat().st_size / 1024
    print(f"Dataset saved to {RESULTS_FILE} ({size_kb:.0f} KB)")
    return RESULTS_FILE


if __name__ == "__main__":
    fetch()
