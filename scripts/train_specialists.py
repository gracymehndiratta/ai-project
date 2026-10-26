"""Train seven calibrated lesion specialists using the complete existing data."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--output-dir", default="outputs/feature_models/efficientnet_b0_specialists")
    args = parser.parse_args()
    if args.epochs < 1 or args.trees < 1:
        parser.error("Epoch and tree counts must be positive")
    os.chdir(ROOT)
    import json
    import numpy as np
    import pandas as pd
    import torch
    from src.utils.specialist_training import train_specialists
    torch.set_num_threads(4)
    baseline = Path("outputs/feature_models/efficientnet_b0_improved")
    frame = pd.read_csv(baseline / "split_manifest.csv")
    info = json.loads((baseline / "experiment.json").read_text())
    cache = Path(f"data/features/efficientnet_b0_{info['fingerprint'][:16]}.npy")
    features = np.load(cache, mmap_mode="r")
    if features.shape != (11720, 2560) or not np.isfinite(features).all():
        raise ValueError("The complete verified multi-view feature cache is required")
    train_specialists(features, frame, info, args.output_dir, args.epochs, args.trees)


if __name__ == "__main__":
    main()
