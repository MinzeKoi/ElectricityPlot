"""Offline verification of prepared real artifacts and their pilot protocol."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_submeter.data.common import DataError, read_json, safe_child, sha256
from virtual_submeter.data.protocol import read_partition


def main():
    parser = argparse.ArgumentParser(description="ElectricityPlot M1 offline artifact verification")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit_path = args.prepared / "manifest.json"
        audit = read_json(audit_path)
        manifest_path = args.protocol / "protocol.json"
        manifest = read_json(manifest_path)
        if sha256(audit_path) != manifest["audit_sha256"]:
            raise DataError("Protocol refers to a different audit")
        if sha256(args.protocol / "plan.json") != manifest["plan_sha256"]:
            raise DataError("Frozen plan changed")
        for stats in audit["households"].values():
            for artifact in stats["artifacts"].values():
                if sha256(safe_child(args.prepared, artifact["path"])) != artifact["sha256"]:
                    raise DataError("Prepared artifact hash mismatch")
        for partition in manifest["partitions"]:
            role = partition["role"]
            consumer = "training" if role.startswith("source_") else "selection" if role == "target_adaptation" else "inference"
            result = read_partition(manifest_path, partition["partition_id"], consumer)
            if set(result) != {"aggregate"}:
                raise DataError("Unexpected labels in aggregate-only read")
            for artifact in partition["artifacts"].values():
                if sha256(safe_child(args.protocol, artifact["path"])) != artifact["sha256"]:
                    raise DataError("Partition artifact hash mismatch")
            if role in ("target_adaptation", "frozen_test"):
                try:
                    read_partition(manifest_path, partition["partition_id"], consumer, include_labels=True)
                except DataError:
                    pass
                else:
                    raise DataError("Target submeter answers leaked to a model consumer")
    except (DataError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "verification_failed", "error": str(error)}))
        return 2
    print(json.dumps({"status": "m1_artifacts_verified", "purpose": manifest["purpose"],
                      "audited_households": len(audit["households"]), "partitions": len(manifest["partitions"]),
                      "training": "not_checked_by_m1_verifier", "metrics": "not_checked_by_m1_verifier"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
