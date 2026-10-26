"""Extract all existing images with frozen PanDerm Base; no classifier training."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='artifacts/panderm_base.pth')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Batch size must be positive and workers nonnegative')
    os.chdir(ROOT)
    import numpy as np
    import pandas as pd
    import timm
    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset
    from PIL import Image
    from src.data.feature_cache import ManifestImages
    from src.models.panderm import (build_panderm, CHECKPOINT_ID, get_source,
        sha256_file, SOURCE_SHA256, UPSTREAM_COMMIT)
    torch.set_num_threads(4)
    # Make patch-convolution/matmul precision consistent across batch shapes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        import gdown
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = checkpoint.with_suffix('.download')
        gdown.download(id=CHECKPOINT_ID, output=str(partial), quiet=False)
        partial.replace(checkpoint)
    get_source()
    checkpoint_sha = sha256_file(checkpoint)
    manifest_path = Path('outputs/feature_models/efficientnet_b0_improved/split_manifest.csv')
    frame = pd.read_csv(manifest_path)
    assert len(frame) == 11720 and frame.image_id.is_unique
    assert frame.split.value_counts().to_dict() == {'train': 7014, 'validation': 1509,
        'test': 1492, 'official_validation': 193, 'official_test': 1512}
    assert frame[frame.source_partition == 'training'].groupby('lesion_id').split.nunique().eq(1).all()
    identity = hashlib.sha256(frame.to_csv(index=False).encode())
    for name in frame.image_path:
        stat = Path(name).stat()
        identity.update(f'{stat.st_size}:{stat.st_mtime_ns}'.encode())
    profile = {'backbone': 'panderm_base', 'checkpoint_sha256': checkpoint_sha,
        'source_sha256': SOURCE_SHA256, 'upstream_commit': UPSTREAM_COMMIT,
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': timm.__version__,
        'preprocessing': 'RGB;Resize(shorter=256,bilinear,antialias=True);CenterCrop224;ToTensor;Normalize(mean=.485,.456,.406;std=.228,.224,.225)',
        'pooling': 'official forward_features(is_train=False), final-normalized CLS token',
        'dimension': 768, 'dtype': 'float32', 'cuda_matmul_tf32': False, 'cudnn_tf32': False,
        'views': 'one canonical view; no augmentation',
        'training': 'Frozen eval-mode encoder, no gradients, no adaptation on any split'}
    identity.update(json.dumps(profile, sort_keys=True).encode())
    fingerprint = identity.hexdigest()
    folder = Path('data/features')
    folder.mkdir(parents=True, exist_ok=True)
    array_path = folder / f'panderm_base_{fingerprint[:16]}.npy'
    status_path = array_path.with_suffix('.json')
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        **profile, 'fingerprint': fingerprint, 'complete': False, 'rows_completed': 0,
        'rows': len(frame), 'split_counts': frame.split.value_counts().to_dict(),
        'checkpoint_path': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
        'manifest_sha256': sha256_file(manifest_path),
        'created_utc': datetime.now(timezone.utc).isoformat()}
    assert status['fingerprint'] == fingerprint
    if status['complete']:
        values = np.load(array_path, mmap_mode='r')
        assert values.shape == (11720, 768) and np.isfinite(values).all()
        assert sha256_file(array_path) == status['embeddings_sha256']
        print(f'Complete verified cache already available: {array_path}', flush=True)
        return
    frame.to_csv(array_path.with_name(array_path.stem + '_manifest.csv'), index=False)
    model, transform = build_panderm(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    status.update({'encoder_parameters': sum(p.numel() for p in model.parameters()),
        'encoder_tensor_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': str(device), 'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None})
    assert status['trainable_parameters'] == 0
    model.to(device).eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if status['rows_completed'] and not array_path.exists():
        raise ValueError('Resume metadata exists but the embedding array is missing')
    mode = 'r+' if array_path.exists() and status_path.exists() else 'w+'
    values = np.lib.format.open_memmap(array_path, mode=mode, dtype=np.float32, shape=(len(frame), 768))
    start = status['rows_completed']
    if start:
        assert np.isfinite(values[:start]).all()
    loader = DataLoader(Subset(ManifestImages(frame, transform), range(start, len(frame))),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
        pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0)
    before = time.monotonic()
    print(f'PanDerm Base: {status["encoder_parameters"]:,} parameters; checkpoint {checkpoint.stat().st_size:,} bytes', flush=True)
    print(f'Extracting {len(frame)} images on {device}; resume at {start}', flush=True)
    with torch.inference_mode():
        for batch in loader:
            embeddings = model.forward_features(batch.to(device, non_blocking=True), is_train=False).float().cpu().numpy()
            assert embeddings.shape == (len(batch), 768) and np.isfinite(embeddings).all()
            values[start:start + len(batch)] = embeddings
            start += len(batch)
            values.flush()
            status['rows_completed'] = start
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(status_path)
            if start % (args.batch_size * 20) == 0 or start == len(frame):
                print(f'Embeddings: {start}/{len(frame)}; elapsed {time.monotonic()-before:.1f}s', flush=True)
    assert np.isfinite(values).all() and np.all(np.linalg.norm(values, axis=1) > 0)
    # A real image from each class independently verifies cache row alignment
    # and single-image consistency against the batched extraction.
    checks = []
    for _, row in frame[frame.split == 'test'].groupby('label', sort=True).head(1).iterrows():
        index = frame.index[frame.image_id == row.image_id][0]
        with Image.open(row.image_path) as image:
            tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.inference_mode():
            single = model.forward_features(tensor, is_train=False).float().cpu().numpy()[0]
        np.testing.assert_allclose(single, values[index], rtol=1e-4, atol=1e-4)
        checks.append({'image_id': row.image_id, 'max_abs_delta': float(np.abs(single-values[index]).max())})
    del values, loader
    status.update({'complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'extraction_seconds_this_session': time.monotonic() - before,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0,
        'array_bytes': array_path.stat().st_size, 'embeddings_sha256': sha256_file(array_path),
        'single_image_checks': checks})
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(status_path)
    receipt = Path('outputs/diagnostics/panderm_embeddings')
    receipt.mkdir(parents=True, exist_ok=True)
    (receipt / 'extraction_receipt.json').write_text(json.dumps({**status, 'embeddings_path': str(array_path),
        'manifest_path': str(array_path.with_name(array_path.stem + '_manifest.csv'))}, indent=2))
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
