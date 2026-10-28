"""Streamlit entry point for the local research dashboard."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
from src.ui.dashboard import main
if __name__=="__main__": main()
