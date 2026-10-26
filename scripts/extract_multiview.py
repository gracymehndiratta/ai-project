import pandas as pd
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.feature_cache import assign_splits, cache_features
if __name__ == '__main__':
    torch.set_num_threads(4)
    frame = assign_splits(pd.read_csv('data/processed/isic2018_full_manifest.csv'))
    features, info = cache_features(frame, 'data/features', batch_size=16, workers=4, feature_views='full_d4_center')
    print(info, flush=True)
