"""Compare the author probe and three custom heads on the released HAM split."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    os.chdir(ROOT)
    from src.utils.panderm_training import train
    train('outputs/feature_models/panderm_large_comparison', large=True)
