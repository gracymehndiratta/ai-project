"""Run a paired regularization/SMOTE experiment without modifying earlier runs."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    import numpy as np
    import pandas as pd
    import torch
    from src.utils.regularized_specialists import train_variant
    torch.set_num_threads(4)
    parent = Path('outputs/feature_models')
    audit = Path('outputs/diagnostics/regularized_specialists')
    audit.mkdir(parents=True, exist_ok=True)
    receipt = audit / 'previous_artifact_hashes.json'
    if not receipt.exists():
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*')
                  if p.is_file() and 'specialists_regularized' not in str(p) and 'specialists_smote' not in str(p)}
        receipt.write_text(json.dumps(hashes, indent=2))
    baseline = parent / 'efficientnet_b0_improved'
    manifest = pd.read_csv(baseline / 'split_manifest.csv')
    assert manifest.split.value_counts().to_dict() == {'train': 7014, 'validation': 1509, 'test': 1492,
                                                        'official_validation': 193, 'official_test': 1512}
    assert manifest[manifest.source_partition == 'training'].groupby('lesion_id').split.nunique().eq(1).all()
    info = json.loads((baseline / 'experiment.json').read_text())
    features = np.load(f"data/features/efficientnet_b0_{info['fingerprint'][:16]}.npy", mmap_mode='r')
    assert features.shape == (11720, 2560) and np.isfinite(features).all()
    for smote, name in [(False, 'efficientnet_b0_specialists_regularized'),
                         (True, 'efficientnet_b0_specialists_smote')]:
        train_variant(features, manifest, info, parent / name, smote)
    hashes = json.loads(receipt.read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    print(f'Preserved {len(hashes)} previous artifacts unchanged.', flush=True)


if __name__ == '__main__':
    main()
