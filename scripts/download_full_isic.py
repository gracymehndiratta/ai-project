"""Download all ISIC 2018 Task 3 images and validate their metadata."""
import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.full_isic import prepare_release

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.workers <= 96:
        parser.error("--workers must be between 1 and 96")
    os.chdir(ROOT)
    print(prepare_release(workers=args.workers), flush=True)
