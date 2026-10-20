"""Budget/strategy/seed comparisons; simulated confirmations are explicitly marked."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_submeter.adaptation.runner import run_comparison
from virtual_submeter.data.common import DataError


def main():
    parser = argparse.ArgumentParser(description="ElectricityPlot M3 weak-label comparison")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/refit_m1_protocol/protocol.json"))
    parser.add_argument("--settings", type=Path, default=Path("configs/m3_adaptation.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--strategies", nargs="+", choices=("random", "uncertainty", "proposed"), default=["random", "uncertainty", "proposed"])
    parser.add_argument("--no-test-evaluation", action="store_true", help="Freeze all adapted models without opening test labels")
    args = parser.parse_args()
    try:
        result = run_comparison(args.model, args.manifest, args.settings, args.output, args.budgets, args.seeds, args.strategies, not args.no_test_evaluation)
    except (DataError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"{result['status']}: {len(result['cases'])} cases in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
