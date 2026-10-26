"""Original Large encoder + unchanged author probe on the released HAM split.

This is a released-code reference attempt, not a claim of bitwise reproduction:
the downloadable split differs from the manuscript's stated 60/20/20 split.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Subset
    from src.data.full_isic import CLASS_NAMES
    from src.data.feature_cache import ManifestImages
    from src.models.panderm import build_panderm, sha256_file, SOURCE_SHA256, UPSTREAM_COMMIT
    from src.utils.feature_training import score_predictions
    from src.utils.tuned_training import TunedClassifier
    from sklearn.preprocessing import FunctionTransformer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path('outputs/diagnostics/panderm_paper_reference')
    out.mkdir(parents=True, exist_ok=True)
    old_files = out / 'previous_artifact_hashes.json'
    if not old_files.exists():
        old_files.write_text(json.dumps({str(p): sha256_file(p) for p in Path('outputs/feature_models').rglob('*') if p.is_file()}, indent=2))
    checkpoint = Path('artifacts/panderm_paper/panderm_large.pth')
    checkpoint_sha = sha256_file(checkpoint)
    paper_csv = Path('artifacts/panderm_paper/metadata/HAM10000_clean/ISIC2018_splits/HAM_clean.csv')
    paper = pd.read_csv(paper_csv)
    frame = pd.read_csv('outputs/feature_models/efficientnet_b0_improved/split_manifest.csv')
    by_id = frame.set_index('image_id')
    assert set(paper.image_id) <= set(by_id.index) and paper.image_id.is_unique
    assert paper.label.tolist() == paper.dx.map(dict(zip(CLASS_NAMES, range(7)))).tolist()
    assert paper.dx.tolist() == by_id.loc[paper.image_id, 'label'].tolist()
    entries = json.loads(Path('artifacts/panderm_paper/archive_entries.json').read_text())
    archived = {Path(entry['name']).stem: entry for entry in entries if entry['name'].lower().endswith('.jpg')}
    # Prove existing images are byte-identical to the released bundle, without
    # downloading another 3.25 GB copy. ZIP CRC and uncompressed byte size match.
    for image_id in paper.image_id:
        entry = archived[image_id]
        payload = Path(by_id.loc[image_id].image_path).read_bytes()
        assert len(payload) == entry['bytes'] and zlib.crc32(payload) == entry['crc'], image_id
    used = paper[paper.split.isin(['train', 'val', 'test'])]
    assert used.groupby('lesion_id').split.nunique().eq(1).all()
    paper.assign(project_split=by_id.loc[paper.image_id, 'split'].to_numpy()).to_csv(out / 'released_split_manifest.csv', index=False)
    plan = {'model': 'PanDerm_Large_LP', 'upstream_commit': UPSTREAM_COMMIT,
        'checkpoint_sha256': checkpoint_sha, 'source_sha256': SOURCE_SHA256,
        'released_csv_sha256': sha256_file(paper_csv), 'released_split_counts': paper.split.value_counts().to_dict(),
        'unrecognized_rows': paper[~paper.split.isin(['train', 'val', 'test'])].image_id.tolist(),
        'unknown_split_policy': 'Exactly upstream Derm_Dataset: only train, val and test are selected; do not relabel trainisic.',
        'train_order': 'Upstream Derm_Dataset: sample(frac=1, random_state=42).reset_index(drop=True)',
        'probe': 'Exact upstream logistic_regression.py class, no edits; C=M*classes/100; max_iter1000; seed0; no class weighting or scaling',
        'preprocessing': 'Exact upstream builder.py Resize256/CenterCrop224, ImageNet mean, std .228/.224/.225',
        'architecture': 'Unmodified published Large encoder, 24 blocks, 1024-dimensional CLS features; frozen',
        'hardware_adaptations': 'Float32, TF32 disabled, extraction batch8 for 8GB GPU; current installed torch/timm versions',
        'image_integrity': 'All 10015 released CSV images matched existing local images by ZIP uncompressed size and CRC32',
        'manuscript_discrepancy': 'Methods states60/20/20; released CSV actually train8207,val575,test1232,trainisic1. This is a released-code reference, not guaranteed exact paper replication.',
        'published_targets': {'HAM_weighted_f1': .926, 'HAM_balanced_accuracy': .807},
        'test_policy': 'Fixed published recipe. No validation/test parameter search. Paper test is distinct from our original internal test.'}
    (out / 'protocol.json').write_text(json.dumps(plan, indent=2))
    print('Verified all released images against archive CRC; published split:', plan['released_split_counts'], flush=True)
    identity = hashlib.sha256((checkpoint_sha+SOURCE_SHA256+frame.to_csv(index=False)+'canonical-fp32-no-tf32').encode()).hexdigest()
    cache = Path(f'data/features/panderm_large_{identity[:16]}.npy')
    status_path = cache.with_suffix('.json')
    model, transform = build_panderm(checkpoint, variant='large')
    assert all(not p.requires_grad for p in model.parameters())
    print('Loaded Large:', sum(p.numel() for p in model.parameters()), 'encoder parameters;',
          len(model.pretraining_only_keys), 'unused pretraining-only tensors.', flush=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        'fingerprint': identity, 'complete': False, 'rows_completed': 0,
        'backbone': 'panderm_large', 'dimension': 1024, 'checkpoint_sha256': checkpoint_sha,
        'upstream_commit': UPSTREAM_COMMIT, 'parameters': sum(p.numel() for p in model.parameters()),
        'unused_pretraining_keys': model.pretraining_only_keys}
    assert status['fingerprint'] == identity
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    if not status['complete']:
        values = np.lib.format.open_memmap(cache, mode='r+' if status_path.exists() else 'w+',
            dtype=np.float32, shape=(len(frame), 1024))
        start = status['rows_completed']
        loader = DataLoader(Subset(ManifestImages(frame, transform), range(start, len(frame))),
            batch_size=8, num_workers=4, shuffle=False, pin_memory=device=='cuda', persistent_workers=True)
        begin = time.monotonic()
        with torch.inference_mode():
            for batch in loader:
                embeddings = model.forward_features(batch.to(device), is_train=False).cpu().numpy()
                assert embeddings.shape == (len(batch),1024) and np.isfinite(embeddings).all()
                values[start:start+len(batch)] = embeddings
                start += len(batch)
                values.flush()
                status['rows_completed'] = start
                temporary = status_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(status, indent=2)); temporary.replace(status_path)
                if start % 800 == 0 or start == len(frame):
                    print(f'Large embeddings: {start}/{len(frame)}; {time.monotonic()-begin:.1f}s', flush=True)
        del loader, values
        status['complete'] = True
        status['embeddings_sha256'] = sha256_file(cache)
        status_path.write_text(json.dumps(status, indent=2))
        frame.to_csv(cache.with_name(cache.stem+'_manifest.csv'), index=False)
    features = np.load(cache, mmap_mode='r')
    assert features.shape == (11720,1024) and np.isfinite(features).all()
    checks = []
    for index, row in frame[frame.split=='test'].groupby('label', sort=True).head(1).iterrows():
        with Image.open(row.image_path) as image:
            batch = transform(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.inference_mode(): single = model.forward_features(batch,is_train=False).cpu().numpy()[0]
        np.testing.assert_allclose(single, features[index], atol=1e-4, rtol=1e-4)
        checks.append({'image_id': row.image_id, 'max_delta': float(np.abs(single-features[index]).max())})
    del model
    torch.cuda.empty_cache()
    probe_source = Path('artifacts/PanDerm/classification/panderm_model/downstream/eval_features/logistic_regression.py')
    probe_sha = sha256_file(probe_source)
    spec = importlib.util.spec_from_file_location('_unchanged_author_panderm_probe', probe_source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Same order and arguments as the released Dataset and linear_eval.py.
    training = paper[paper.split=='train'].sample(frac=1, random_state=42).reset_index(drop=True)
    lookup = {image_id:index for index,image_id in enumerate(frame.image_id)}
    train_indices = np.array([lookup[i] for i in training.image_id])
    x = torch.tensor(np.asarray(features[train_indices]))
    y = torch.tensor(training.label.to_numpy(),dtype=torch.long)
    cost = features.shape[1]*len(CLASS_NAMES)/100
    classifier = module.LogisticRegression(C=cost,max_iter=1000,verbose=True,random_state=0)
    classifier.fit(x,y)
    wrapped = TunedClassifier([classifier.logreg.cpu().eval()], FunctionTransformer(validate=False))
    joblib.dump(wrapped,out/'model.joblib')
    torch.save(classifier.logreg.state_dict(),out/'author_probe_state.pt')
    (out/'selection_complete.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'model_sha256':sha256_file(out/'model.joblib'),'probe_source_sha256':probe_sha,
        'cost':cost,'max_iter':1000,'seed':0,'no_validation_or_test_selection':True},indent=2))
    loaded = joblib.load(out/'model.joblib')
    metrics, comparisons = {},[]
    for split in ['train','val','test']:
        subset = paper[paper.split==split]
        indices = np.array([lookup[i] for i in subset.image_id])
        probability=loaded.predict_proba(np.asarray(features[indices]))
        measured=score_predictions(subset.label.to_numpy(),probability)
        metrics['released_'+split]=measured
        records=pd.DataFrame(probability,columns=CLASS_NAMES)
        records.insert(0,'true_label',subset.dx.to_numpy());records.insert(0,'image_id',subset.image_id.to_numpy())
        records.to_csv(out/f'released_{split}_predictions.csv',index=False)
        comparisons.append({'split':'released_'+split,**{k:measured[k] for k in ['accuracy','macro_f1','macro_f05','weighted_f1','balanced_accuracy']}})
    # Official external images are never part of the released CSV training set.
    for split in ['official_validation','official_test']:
        indices=np.flatnonzero(frame.split==split)
        probability=loaded.predict_proba(np.asarray(features[indices]))
        truth=frame.iloc[indices].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
        measured=score_predictions(truth,probability);metrics[split]=measured
        records=pd.DataFrame(probability,columns=CLASS_NAMES)
        records.insert(0,'true_label',frame.iloc[indices].label.to_numpy());records.insert(0,'image_id',frame.iloc[indices].image_id.to_numpy())
        records.to_csv(out/f'{split}_predictions.csv',index=False)
        comparisons.append({'split':split,**{k:measured[k] for k in ['accuracy','macro_f1','macro_f05','weighted_f1','balanced_accuracy']}})
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(comparisons).to_csv(out/'comparison.csv',index=False)
    previous=json.loads(old_files.read_text());assert all(sha256_file(p)==digest for p,digest in previous.items())
    (out/'verification.json').write_text(json.dumps({'passed':True,'previous_artifacts_unchanged':len(previous),
        'real_image_checks':checks,'checkpoint_sha256':checkpoint_sha,'probe_source_sha256':probe_sha,
        'published_split_preserved':True,'matching_local_images':len(paper),'model_cache':str(cache)},indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=='__main__': main()
