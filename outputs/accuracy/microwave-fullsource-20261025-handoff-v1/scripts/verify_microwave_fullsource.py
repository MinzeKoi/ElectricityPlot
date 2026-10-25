"""Independent raw-context/pool and augmentation arithmetic checks; no target CSV truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run/"source"))
    from virtual_submeter.data.common import DataError, read_json, sha256, utc, write_json
    from virtual_submeter.models.arrays import numeric
    from virtual_submeter.accuracy.microwave_diagnosis import load_npz
    from virtual_submeter.accuracy import microwave_fullsource_execution as e
    output = run/"independent_verification.json"
    if output.exists():
        raise DataError("Independent verification receipt already exists")
    result = e.verify(run)
    data, pools = load_npz(run/"train.npz"), load_npz(run/"pools.npz")
    support, protocol = read_json(run/"support.json"), read_json(run/"upstream_protocol.json")
    raw_power = np.full_like(data["x"], np.nan)
    allowed_context = np.zeros(len(data["y"]), bool)
    for number, house in enumerate(("refit_2", "refit_19")):
        original = next(p for p in protocol["partitions"] if p["household_id"] == house and p["role"] == "source_train")
        low, high = utc(original["start"]), utc(original["end"])
        mappings = {}
        for channel in ("aggregate", "microwave"):
            item = support[house]["inputs"][channel]
            if sha256(Path(item["path"])) != item["sha256"]:
                raise DataError("Raw source changed")
            with Path(item["path"]).open(encoding="utf-8", newline="") as stream:
                mappings[channel] = {utc(r["timestamp"]): numeric(r["aggregate_power_w"] if channel == "aggregate" else r["power_w"]) for r in csv.DictReader(stream)}
        indices = np.flatnonzero(data["house"] == number)
        for i in indices:
            center = int(data["t"][i])
            stamps = [center+8*j for j in range(-15, 16)]
            aggregate = np.array([mappings["aggregate"].get(s, np.nan) for s in stamps], np.float32)
            np.testing.assert_array_equal(data["x"][i], aggregate)
            raw_power[i] = [mappings["microwave"].get(s, np.nan) for s in stamps]
            allowed_context[i] = center-120 >= low and center+128 <= high
    np.testing.assert_array_equal(raw_power, pools["power"])
    np.testing.assert_array_equal(raw_power[:, 15], data["y"])
    residual = data["x"]-raw_power
    np.testing.assert_array_equal(residual, pools["residual"])
    valid = allowed_context & np.isfinite(raw_power).all(axis=1) & np.isfinite(data["x"]).all(axis=1)
    valid &= (raw_power >= 0).all(axis=1) & (data["x"] >= 0).all(axis=1) & (residual >= 0).all(axis=1)
    np.testing.assert_array_equal(pools["template_ok"], valid & (data["y"] >= 200))
    np.testing.assert_array_equal(pools["background_ok"], valid & (raw_power < 200).all(axis=1))
    expected_hashes = [[support[h]["inputs"][c]["sha256"] for c in ("aggregate", "microwave")] for h in ("refit_2", "refit_19")]
    if pools["source_hashes"].tolist() != expected_hashes:
        raise DataError("Augmentation source identities do not match raw CSVs")
    arithmetic = []
    for arm, seed, dest in e.model_dirs(run):
        if arm != "event_background":
            continue
        attempts, synthesized = 0, 0
        with (dest/"augmentation.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                attempts += 1
                i = r["template_index"]
                if r["source_data_hashes"] != expected_hashes:
                    raise DataError("Trace missing raw source identities")
                if r["status"] == "synthesized":
                    synthesized += 1
                    j = r["background_index"]
                    if not pools["template_ok"][i] or not pools["background_ok"][j]:
                        raise DataError("Ineligible template/background used")
                    x = raw_power[i]+residual[j]
                    if r["template_power_sha256"] != digest(raw_power[i]) or r["background_residual_sha256"] != digest(residual[j]):
                        raise DataError("Trace raw arithmetic inputs mismatch")
                else:
                    x = data["x"][i]
                if r["output_sha256"] != digest(x):
                    raise DataError("Trace combined power/fallback arithmetic mismatch")
        arithmetic.append({"seed": seed, "attempts_checked": attempts, "synthesized_checked": synthesized})
    result.update(status="independently_verified", source_windows_checked=len(data["y"]),
                  template_centers_checked=int(pools["template_ok"].sum()), background_centers_checked=int(pools["background_ok"].sum()),
                  raw_arithmetic_checks=arithmetic, registration_sha256=sha256(run/"registration.json"),
                  freeze_sha256=sha256(run/"freeze.json"), script_sha256=sha256(Path(__file__)))
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
