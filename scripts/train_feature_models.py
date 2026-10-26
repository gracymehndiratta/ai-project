"""Complete full-data pipeline: download, cache features, train all three models."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the entire official release first")
    parser.add_argument("--wait-for-data", action="store_true", help="Wait for a separately running full downloader")
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="Image loading worker processes")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone", choices=["efficientnet_b0", "resnet18", "mobilenet_v2"], default="efficientnet_b0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--xgb-estimators", type=int, default=1200)
    parser.add_argument("--resampling", choices=["none", "smote"], default="none")
    parser.add_argument("--smote-target", type=int, default=500, help="Minimum training examples per minority class after SMOTE")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.xgb_estimators < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Epochs, estimator count and batch size must be positive; workers must be nonnegative")
    if not 1 <= args.download_workers <= 96:
        parser.error("--download-workers must be between 1 and 96")
    if args.smote_target < 1:
        parser.error("--smote-target must be positive")
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import pandas as pd
    import torch
    from src.data.full_isic import prepare_release
    from src.data.feature_cache import assign_splits, cache_features
    from src.utils.feature_training import train_all
    torch.set_num_threads(4)
    manifest_path = Path("data/processed/isic2018_full_manifest.csv")
    if args.download:
        manifest_path = prepare_release(workers=args.download_workers)
    if args.wait_for_data:
        completion = Path("data/raw/isic2018_archives/download_manifest.json")
        while not (manifest_path.exists() and completion.exists()):
            print("Waiting for the complete verified dataset manifest; no subset or synthetic fallback.", flush=True)
            time.sleep(30)
    if not manifest_path.exists():
        parser.error("Full dataset not prepared. Use --download or run scripts/download_full_isic.py first.")
    frame = pd.read_csv(manifest_path)
    expected = {"training": 10015, "validation": 193, "test": 1512}
    if frame.source_partition.value_counts().to_dict() != expected:
        raise ValueError("Full official release is required: 10,015 training, 193 validation, 1,512 test images")
    frame = assign_splits(frame, args.seed)
    features, cache_info = cache_features(frame, "data/features", args.backbone, args.batch_size, args.workers)
    suffix = "_smote" if args.resampling == "smote" else ""
    output = args.output_dir or f"outputs/feature_models/{args.backbone}{suffix}"
    print(train_all(features, frame, cache_info, output, args.seed, args.max_epochs, args.xgb_estimators,
                    resampling=args.resampling, smote_target=args.smote_target).to_string(index=False), flush=True)
    print(f"All three models completed. Results: {output}/REPORT.md", flush=True)


if __name__ == "__main__":
    main()
