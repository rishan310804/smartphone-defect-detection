import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm.auto import tqdm

# Configuration
DATASET_URL = "https://github.com/VincentHancoder/SSGD/archive/refs/heads/master.zip"
# 'smartphone defect detection' root
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"
ZIP_PATH = DATA_DIR / "ssgd.zip"
EXTRACT_DIR = DATA_DIR / "images"


def download_ssgd():
    """
    Downloads the SSGD dataset ZIP if not already present,
    extracts its contents into data/raw/images/,
    and cleans up the ZIP file.
    """
    # 1. Prepare directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Skip download if already extracted
    if any(EXTRACT_DIR.iterdir()):
        print(f"Dataset already extracted at {EXTRACT_DIR}")
        return

    # 3. Download with progress bar
    try:
        print(f"⬇Downloading SSGD dataset from {DATASET_URL} ...")
        resp = requests.get(DATASET_URL, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(ZIP_PATH, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, unit_divisor=1024,
            desc="Downloading ZIP"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return

    # 4. Extract and clean up
    try:
        print(f"Extracting to {EXTRACT_DIR} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("Invalid ZIP file.", file=sys.stderr)
    finally:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
            print(f"Removed archive {ZIP_PATH}")


if __name__ == "__main__":
    download_ssgd()
