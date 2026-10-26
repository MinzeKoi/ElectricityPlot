"""Ordinary opt-in center-TCN inference; incomplete C31 centers remain missing."""
import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE/"registration.json").exists() else HERE.parent
sys.path.insert(0, str(ROOT/("source" if (HERE/"registration.json").exists() else "src")))
from virtual_submeter.models.center_mask_tcn import load_state_model, predict_state_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--aggregate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; refusing to overwrite")
    meta, model = load_state_model(a.model)
    with a.aggregate.open(encoding="utf-8", newline="") as stream:
        rows = predict_state_rows(meta, model, list(csv.DictReader(stream)))
    with a.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "probability_on", "state"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
