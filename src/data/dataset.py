"""
Dataset utilities for HAM10000 skin lesion images.
Supports real HAM10000 data as well as a tiny synthetic dataset for quick tests.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for dermoscopic images and diagnostic labels."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        image_col: str = "image_id",
        label_col: str = "label",
        extension: str = ".jpg",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.extension = extension

        self.df = pd.read_csv(csv_path)
        if self.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in CSV.")

        self.classes = sorted(self.df[self.label_col].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        image_name = row[self.image_col]
        label = self.class_to_idx[row[self.label_col]]

        image_path = self.image_dir / f"{image_name}{self.extension}"
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_label_distribution(self) -> dict:
        counts = Counter(self.df[self.label_col])
        return dict(counts)


def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split a dataset into train/val/test subsets."""
    n = len(dataset)
    indices = np.arange(n)

    labels = None
    if stratify and hasattr(dataset, "df") and dataset.label_col in dataset.df.columns:
        labels = dataset.df[dataset.label_col].map(dataset.class_to_idx).values

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=labels[indices] if labels is not None else None,
    )

    if val_ratio == 0:
        return Subset(dataset, train_idx), Subset(dataset, temp_idx), Subset(dataset, [])

    test_size = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size,
        random_state=seed,
        stratify=labels[temp_idx] if labels is not None else None,
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def partition_data_iid(dataset: Dataset, num_clients: int, seed: int = 42) -> list:
    """Partition dataset indices evenly and randomly among clients."""
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_sizes = np.array_split(indices, num_clients)
    return [Subset(dataset, idx.tolist()) for idx in split_sizes]


def partition_data_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list:
    """
    Partition dataset in a non-IID fashion using a Dirichlet distribution.
    Each class's samples are distributed across clients proportionally to a
    Dirichlet draw with concentration parameter `alpha`.
    """
    rng = np.random.default_rng(seed)

    # Determine the indices we are allowed to partition.
    if isinstance(dataset, Subset):
        available_indices = np.array(dataset.indices)
        base_dataset = dataset.dataset
    else:
        available_indices = np.arange(len(dataset))
        base_dataset = dataset

    if hasattr(base_dataset, "df") and hasattr(base_dataset, "label_col") and hasattr(base_dataset, "class_to_idx"):
        all_labels = base_dataset.df[base_dataset.label_col].map(base_dataset.class_to_idx).values
        labels = all_labels[available_indices]
    else:
        labels = np.array([dataset[int(i)][1] for i in available_indices])

    n_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(n_classes):
        cls_positions = np.where(labels == cls)[0]
        cls_indices = available_indices[cls_positions]
        rng.shuffle(cls_indices)

        if len(cls_indices) == 0:
            continue

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        split_points = np.cumsum((proportions[:-1] * len(cls_indices)).astype(int))
        splits = np.split(cls_indices, split_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [Subset(base_dataset, idx) for idx in client_indices]


def create_synthetic_dataset(
    output_dir: str,
    num_samples: int = 1000,
    image_size: int = 128,
    num_classes: int = 7,
    seed: int = 42,
) -> str:
    """
    Generate a tiny synthetic HAM10000-like dataset for quick integration tests.
    Returns the path to the generated CSV file.
    """
    set_seed(seed)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    class_names = [f"class_{i}" for i in range(num_classes)]

    rows = []
    for i in range(num_samples):
        label = rng.integers(0, num_classes)
        image_name = f"img_{i:05d}"

        # Generate a random colored blob
        img_array = rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(image_dir / f"{image_name}.jpg")

        rows.append({"image_id": image_name, "label": class_names[label]})

    df = pd.DataFrame(rows)
    csv_path = output_dir / "synthetic_metadata.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def create_metadata_csv(
    image_dir: str,
    output_csv: str,
    label_map: Optional[dict] = None,
) -> str:
    """
    Create a metadata CSV from a directory of images.
    If label_map is not provided, the parent folder name is used as the label.
    """
    image_dir = Path(image_dir)
    rows = []
    for path in image_dir.rglob("*.jpg"):
        image_id = path.stem
        label = label_map.get(image_id, path.parent.name) if label_map else path.parent.name
        rows.append({"image_id": image_id, "label": label})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return output_csv
