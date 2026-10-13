"""
Download a subset of HAM10000 due to limited disk space.

Downloads only HAM10000_images_part_1.zip and extracts a subset of images.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ensure_dirs, load_config

# Harvard Dataverse file IDs
METADATA_URL = "https://dataverse.harvard.edu/api/access/datafile/4338392"
IMAGES_PART1_URL = "https://dataverse.harvard.edu/api/access/datafile/3172585"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download a file with progress."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\rDownloading {dest.name}: {pct:.1f}%", end="", flush=True)
    print()


def extract_subset(zip_path: Path, extract_dir: Path, max_images: int = 3000) -> None:
    """Extract only a subset of images from the zip."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg"))]
        for name in names[:max_images]:
            # Extract to flat directory
            z.extract(name, extract_dir)
            count += 1
            if count % 500 == 0:
                print(f"Extracted {count} images...")
    print(f"Extracted {count} images to {extract_dir}")


def prepare_metadata(raw_csv: Path, processed_csv: Path, allowed_image_ids: set) -> None:
    """Create metadata CSV for the subset."""
    import pandas as pd

    df = pd.read_csv(raw_csv)
    df = df[df["image_id"].isin(allowed_image_ids)].copy()
    df = df[["image_id", "dx"]].rename(columns={"dx": "label"})
    df.to_csv(processed_csv, index=False)
    print(f"Prepared metadata with {len(df)} entries at {processed_csv}")


def main() -> None:
    config = load_config("config.yaml")
    ensure_dirs(config)

    raw_dir = Path(config["dataset"]["raw_dir"])
    processed_dir = Path(config["dataset"]["processed_dir"])
    images_dir = raw_dir / "images"

    max_images = 3000

    print(f"Will download HAM10000 part 1 and extract {max_images} images.")
    print("This requires about 2 GB free space temporarily.")

    # Download metadata
    metadata_path = raw_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        print("Downloading metadata...")
        download_file(METADATA_URL, metadata_path)
    else:
        print("Metadata already exists.")

    # Download images part 1
    zip_path = raw_dir / "HAM10000_images_part_1.zip"
    if not zip_path.exists():
        print("Downloading images part 1 (1.37 GB)...")
        download_file(IMAGES_PART1_URL, zip_path)
    else:
        print("Images zip already exists.")

    # Extract subset
    print(f"Extracting {max_images} images...")
    extract_subset(zip_path, images_dir, max_images=max_images)

    # Get image IDs we actually have
    allowed_ids = {p.stem for p in images_dir.glob("*.jpg")}
    print(f"Available image IDs: {len(allowed_ids)}")

    # Prepare metadata
    processed_csv = processed_dir / "HAM10000_metadata.csv"
    prepare_metadata(metadata_path, processed_csv, allowed_ids)

    # Optionally delete zip to save space
    print("Removing zip file to save space...")
    zip_path.unlink()

    print("Done. You can now run:")
    print("  python scripts/train_centralized.py")


if __name__ == "__main__":
    main()
