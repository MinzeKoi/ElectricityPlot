"""Verify R0 evidence and prepared support without computing activity metrics."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_submeter.data.common import DataError, read_json, safe_child, sha256


def verify(root, validation):
    evidence = read_json(validation)
    for relative, expected in evidence["files"].items():
        if sha256(safe_child(root, relative)) != expected:
            raise DataError(f"Evidence hash mismatch: {relative}")
    roles = read_json(root / "configs/r0_roles_20261022_v2.json")
    final = set(roles["final_reserved"])
    development = set(roles["new_development_reserved"])
    sources = set(roles["source_train_existing"] + roles["source_validation_existing"] + roles["new_source_reserved"])
    if len(final) < 3 or final & (development | sources) or development & sources:
        raise DataError("R0 household roles overlap or final reserve is insufficient")
    results = read_json(root / "docs/R0_RESULTS.json")
    houses = {item["household_id"]: item for item in results["households"]}
    coverage = read_json(root / "outputs/r0-20261022-final-date-coverage.json")
    checked = {item["household_id"]: item for item in coverage["households"]}
    prepared = root / "data/processed/refit_r0_final_dates_20261022"
    manifest = read_json(prepared / "manifest.json")
    selection_path = root / "configs/r0_dates_final_20261022.json"
    selection = read_json(selection_path)
    if manifest.get("date_selection_sha256") != sha256(selection_path) or manifest.get("date_selection") != selection:
        raise DataError("Preparation lost frozen date selection identity")
    if set(manifest["households"]) != final or set(checked) != final:
        raise DataError("Prepared final reserve differs from roles")
    for house in final:
        chosen = houses[house]["all_aligned_28_day_windows"]["first_pass"]
        current = checked[house]
        info = manifest["households"][house]
        if (not current["coverage_and_duration_candidate_pass"] or info["source_sha256"] != houses[house]["source_sha256"]
                or current["start"] != chosen["start"] or current["end"] != chosen["end"]):
            raise DataError("Prepared dates/source/support differ from audit")
        for channel, partitions in current["channels"].items():
            for partition, values in partitions.items():
                expected = chosen["channels"][channel][partition]
                if values["coverage"] != expected["coverage"] or values["valid_days"] != expected["valid_days"]:
                    raise DataError("Production resampling support differs from full-record audit")
        for stats in info["channels"].values():
            if stats["nonzero_bins"] is not None or stats["observed_energy_kwh"] is not None:
                raise DataError("Final preparation computed forbidden activity/energy summaries")
        for artifact in info["artifacts"].values():
            if sha256(safe_child(prepared, artifact["path"])) != artifact["sha256"]:
                raise DataError("Prepared artifact hash mismatch")
    print(f"R0 verified: {len(houses)} audited households; {len(final)} date-selected reserves match production support; formal admission remains pending.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("docs/R0_VALIDATION.json"))
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], args.validation)
