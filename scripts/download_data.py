"""
HAM10000 data preparation helper.

The HAM10000 dataset is publicly available at:
  - Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T
  - ISIC Archive: https://challenge.isic-archive.com/data/

This script expects the following files in data/raw/:
  - HAM10000_metadata.csv
  - images/ folder containing all dermoscopic .jpg files

If the files are not present, the script prints manual download instructions.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ensure_dirs, load_config


def print_download_instructions(raw_dir: Path) -> None:
    print(
        f"""
HAM10000 data not found in {raw_dir}.

Please download the dataset manually:

1. Visit https://doi.org/10.7910/DVN/DBW86T
2. Download:
   - HAM10000_metadata.csv
   - HAM10000_images_part_1.zip
   - HAM10000_images_part_2.zip
3. Extract both zip files and move all .jpg images into:
   {raw_dir / "images"}
4. Move HAM10000_metadata.csv into:
   {raw_dir}

Expected structure:
{raw_dir}/
  HAM10000_metadata.csv
  images/
    ISIC_0024306.jpg
    ISIC_0024307.jpg
    ...
"""
    )


def prepare_metadata(raw_dir: Path, processed_dir: Path) -> None:
    """Preserve lesion IDs so related images stay together during splitting."""
    import pandas as pd

    raw_csv = raw_dir / "HAM10000_metadata.csv"
    out_csv = processed_dir / "HAM10000_metadata.csv"

    df = pd.read_csv(raw_csv)
    # HAM10000 uses 'dx' as the diagnostic label and 'image_id' as the image name.
    df = df[["image_id", "dx", "lesion_id"]].rename(columns={"dx": "label"})
    df.to_csv(out_csv, index=False)
    print(f"Prepared metadata: {out_csv}")


def main() -> None:
    config = load_config("config.yaml")
    ensure_dirs(config)

    raw_dir = Path(config["dataset"]["raw_dir"])
    processed_dir = Path(config["dataset"]["processed_dir"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    images_dir = raw_dir / "images"
    metadata_csv = raw_dir / "HAM10000_metadata.csv"

    if not metadata_csv.exists() or not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        print_download_instructions(raw_dir)
        return

    # Optionally unpack any zip files in raw_dir
    for zip_path in raw_dir.glob("*.zip"):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
        # Move extracted images into images/
        for extracted_dir in raw_dir.iterdir():
            if extracted_dir.is_dir() and extracted_dir.name != "images":
                for img in extracted_dir.glob("*.jpg"):
                    shutil.move(str(img), str(images_dir / img.name))
                extracted_dir.rmdir() if not any(extracted_dir.iterdir()) else None

    prepare_metadata(raw_dir, processed_dir)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
