"""Report coverage-only candidate support without reading event/prediction metrics."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_submeter.data.common import read_json, safe_child, sha256, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; choose a new audit version")
    manifest = read_json(args.manifest)
    rows = []
    for house, info in manifest["households"].items():
        start = utc(info["start"])
        end = utc(info["end"])
        cut = start + ((end - start) // 16) * 8
        windows = {"adaptation": (start, cut), "future": (cut + 3600, end)}
        support = {}
        for channel, artifact in info["artifacts"].items():
            path = safe_child(args.manifest.parent, artifact["path"])
            if sha256(path) != artifact["sha256"]:
                raise ValueError(f"Artifact hash mismatch: {path}")
            counts = {name: {"bins": 0, "valid_bins": 0} for name in windows}
            power_key = "aggregate_power_w" if channel == "aggregate" else "power_w"
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    stamp = utc(row["timestamp"])
                    for name, (first, last) in windows.items():
                        if first <= stamp and stamp + 8 <= last:
                            counts[name]["bins"] += 1
                            counts[name]["valid_bins"] += int(bool(row[power_key]))
            for name, item in counts.items():
                first, last = windows[name]
                expected = max(0, (last - first) // 8)
                item["expected_bins"] = expected
                item["coverage"] = item["valid_bins"] / expected if expected else 0
                item["valid_days"] = item["valid_bins"] * 8 / 86400
                item["passes_coverage_and_days"] = (
                    item["bins"] == expected and item["coverage"] >= 0.9 and item["valid_days"] >= 7
                )
            support[channel] = counts
        complete = (info.get("provenance") or {}).get("kind") == "real_complete"
        required = {"aggregate", "kettle", "microwave", "washing_machine"}
        coverage_pass = complete and required.issubset(support) and all(
            support[channel][window]["passes_coverage_and_days"]
            for channel in required for window in windows
        )
        rows.append({
            "household_id": house, "start": info["start"], "end": info["end"],
            "role": "existing_source" if house in {"2", "5"} else "unassigned_candidate",
            "source_kind": (info.get("provenance") or {}).get("kind"),
            "coverage_and_duration_candidate_pass": coverage_pass,
            "pv_storage_qualification": "No explicit PV note for these households in retrieved README; absence of a note is not confirmation of no generation/storage.",
            "final_admission": "not_frozen", "channels": support,
        })
    report = {
        "schema_version": 1, "prepared_manifest_sha256": sha256(args.manifest),
        "prepared_manifest": str(args.manifest), "scope": "coverage_only_no_events_or_model_scores",
        "split_gap_seconds": 3600, "minimum_coverage": 0.9, "minimum_valid_days_per_partition": 7,
        "known_pv_notes_in_original_refit_readme": ["11", "21"],
        "other_literature_pv_conflicts_pending_review": ["3", "13"],
        "final_households": [], "households": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["household_id"], row["role"], "coverage_pass=", row["coverage_and_duration_candidate_pass"])


if __name__ == "__main__":
    main()
