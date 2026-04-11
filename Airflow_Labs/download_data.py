"""
download_data.py
Run this once to download the diabetes dataset before starting Airflow.

Usage:
    python download_data.py
"""

import urllib.request
import os

URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
SAVE_PATH = os.path.join(os.path.dirname(__file__), "working_data", "diabetes.csv")

def download():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    if os.path.exists(SAVE_PATH):
        print(f"[INFO] Dataset already exists at: {SAVE_PATH}")
        return

    print(f"[INFO] Downloading diabetes dataset...")
    urllib.request.urlretrieve(URL, SAVE_PATH)
    print(f"[SUCCESS] Saved to: {SAVE_PATH}")

if __name__ == "__main__":
    download()