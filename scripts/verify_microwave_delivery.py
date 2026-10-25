"""Read-only byte verification of the complete microwave diagnosis delivery."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "outputs/accuracy/microwave-diagnosis-20261025-v2")
    args = parser.parse_args()
    manifest = json.loads((args.run / "delivery_manifest.json").read_text(encoding="utf-8"))
    count = 0
    for key, base in [("run_artifacts", args.run), ("external_delivery_artifacts", root)]:
        for relative, expected in manifest[key].items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError("Path escapes delivery")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Delivery changed: {relative}")
            count += 1
    print(json.dumps({"status": "delivery_hashes_passed", "files": count}))


if __name__ == "__main__":
    main()
