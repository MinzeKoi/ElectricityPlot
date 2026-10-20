"""Launch with: python -m streamlit run scripts/app.py"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_submeter.app.dashboard import main

main(ROOT)
