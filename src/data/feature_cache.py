"""Persist original EfficientNet features and lesion-disjoint split assignments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision

from src.models.feature_models import build_extractor, MultiViewTransform, extract_multiview


def assign_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratify unique training lesions; keep official external sets separate."""
    frame = frame.copy().reset_index(drop=True)
    train = frame[frame.source_partition == "training"]
    if train.image_id.duplicated().any() or train.lesion_id.isna().any():
        raise ValueError("Training IDs/groupings are missing or duplicated")
    if (train.groupby("lesion_id").label.nunique() > 1).any():
        raise ValueError("A lesion has inconsistent diagnostic labels")
    groups = train[["lesion_id", "label"]].drop_duplicates("lesion_id")
    fit, holdout = train_test_split(groups, test_size=0.30, stratify=groups.label, random_state=seed)
    val, test = train_test_split(holdout, test_size=0.50, stratify=holdout.label, random_state=seed)
    lookup = {g: split for split, rows in [("train", fit), ("validation", val), ("test", test)] for g in rows.lesion_id}
    frame["split"] = frame.lesion_id.map(lookup)
    frame.loc[frame.source_partition == "validation", "split"] = "official_validation"
    frame.loc[frame.source_partition == "test", "split"] = "official_test"
    if frame.split.isna().any() or frame.groupby("lesion_id").split.nunique().max() != 1:
        raise ValueError("Split assignment is incomplete or leaks lesions")
    return frame


class ManifestImages(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame.image_path.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor


def cache_features(frame, cache_dir, backbone="efficientnet_b0", batch_size=64, workers=4, feature_views="canonical"):
    """Resume extraction without retaining the extractor during head training."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(frame.to_csv(index=False).encode())
    for name in frame.image_path:
        stat = Path(name).stat()
        identity.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    identity.update(f"{backbone}:{torchvision.__version__}:original-imagenet-weights-transforms-v1".encode())
    if feature_views not in ("canonical", "full_d4_center"):
        raise ValueError("Unknown image views")
    if feature_views != "canonical":
        identity.update(feature_views.encode())
    fingerprint = identity.hexdigest()
    array_path = cache_dir / f"{backbone}_{fingerprint[:16]}.npy"
    status_path = array_path.with_suffix(".json")
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["fingerprint"] != fingerprint:
            raise ValueError("Feature cache identity mismatch")
        if status["complete"]:
            result = np.load(array_path, mmap_mode="r")
            if result.shape != (len(frame), status["dimension"]) or not np.isfinite(result).all():
                raise ValueError("Invalid completed feature cache")
            print(f"Using complete cache: {array_path}", flush=True)
            return result, status
    else:
        status = {"fingerprint": fingerprint, "complete": False, "rows_completed": 0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, dimension, weight_name = build_extractor(backbone)
    if feature_views == "full_d4_center":
        transform = MultiViewTransform(transform)
        dimension *= 2
    model.to(device).eval()
    mode = "r+" if array_path.exists() and status_path.exists() else "w+"
    features = np.lib.format.open_memmap(array_path, mode=mode, dtype=np.float32, shape=(len(frame), dimension))
    start = status["rows_completed"]
    status.update(backbone=backbone, weights=weight_name, dimension=dimension, device=str(device),
                  torchvision_version=torchvision.__version__, transform=repr(transform), rows=len(frame))
    status["feature_views"] = feature_views
    dataset = Subset(ManifestImages(frame, transform), range(start, len(frame)))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    print(f"Extracting {len(frame)} full-dataset images on {device}; resume at {start}", flush=True)
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            values = (extract_multiview(model, batch) if feature_views != "canonical" else model(batch)).float().cpu().numpy()
            if values.shape[1] != dimension or not np.isfinite(values).all():
                raise ValueError("Extractor produced invalid features")
            features[start:start + len(values)] = values
            start += len(values)
            features.flush()
            status["rows_completed"] = start
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(status_path)
            if start % (batch_size * 10) == 0 or start == len(frame):
                print(f"Features: {start}/{len(frame)}", flush=True)
    status["complete"] = True
    status_path.write_text(json.dumps(status, indent=2))
    del loader, model, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frame.to_csv(cache_dir / f"{backbone}_{fingerprint[:16]}_manifest.csv", index=False)
    return np.load(array_path, mmap_mode="r"), status
