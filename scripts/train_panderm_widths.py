from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    os.chdir(ROOT)
    from src.utils.panderm_widths import train
    train()
