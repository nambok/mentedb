#!/usr/bin/env python3
"""Download the LongMemEval dataset from HuggingFace."""

import os
import sys
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

FILES = {
    "longmemeval_s_cleaned.json": "LongMemEval_S (~115k tokens, ~40 sessions per question)",
    "longmemeval_m_cleaned.json": "LongMemEval_M (~500 sessions per question)",
    "longmemeval_oracle.json": "LongMemEval Oracle (evidence sessions only)",
}


def download_file(filename, description):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"  Already exists: {filename}")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {filename} ({description})...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    print(f"\r  [{pct:3d}%] {downloaded:,} / {total:,} bytes", end="")

        print(f"\r  Done: {filename} ({downloaded:,} bytes)")
        return True

    except requests.RequestException as e:
        print(f"\n  Failed to download {filename}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading LongMemEval dataset...\n")

    # Default to just the _s dataset, download all if --all flag
    if "--all" in sys.argv:
        targets = FILES
    else:
        targets = {"longmemeval_s_cleaned.json": FILES["longmemeval_s_cleaned.json"]}
        print("  (Pass --all to download all dataset variants)\n")

    success = True
    for filename, description in targets.items():
        if not download_file(filename, description):
            success = False

    if success:
        print("\nDataset ready.")
    else:
        print("\nSome downloads failed. Check your connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
