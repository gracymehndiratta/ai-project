"""Train a bounded, validation-selected frozen EfficientNet multi-view search."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/feature_models/efficientnet_b0_improved")
    args = parser.parse_args()
    if args.workers < 0 or args.epochs < 1 or args.trees < 1:
        parser.error("Workers must be nonnegative; epochs and tree count must be positive")
    os.chdir(ROOT)
    import pandas as pd
    import torch
    from src.data.feature_cache import assign_splits, cache_features
    from src.utils.tuned_training import train_tuned
    torch.set_num_threads(4)
    frame = assign_splits(pd.read_csv("data/processed/isic2018_full_manifest.csv"))
    expected = {"training": 10015, "validation": 193, "test": 1512}
    if frame.source_partition.value_counts().to_dict() != expected:
        raise ValueError("The complete official dataset is required")
    features, info = cache_features(frame, "data/features", batch_size=16, workers=args.workers,
                                    feature_views="full_d4_center")
    train_tuned(features, frame, info, args.output_dir, args.epochs, args.trees)


if __name__ == "__main__":
    main()
