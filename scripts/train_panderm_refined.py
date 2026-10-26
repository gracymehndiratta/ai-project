from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    os.chdir(ROOT)
    from src.utils.panderm_refinement import train
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker-kind',choices=['mlp','xgboost','lda','pca256'])
    train(worker_kind=parser.parse_args().worker_kind)
