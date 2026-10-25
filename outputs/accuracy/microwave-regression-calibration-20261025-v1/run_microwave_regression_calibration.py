"""Run against the saved source when invoked from a sealed run directory."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here/'source' if (here/'source').is_dir() else here.parent/'src'))
from virtual_submeter.accuracy.microwave_regression_calibration import main

if __name__ == '__main__':
    main()
