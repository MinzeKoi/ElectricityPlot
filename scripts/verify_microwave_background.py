"""Independent physical/readback checks after the three-fit candidate is frozen."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / "source"))
    import numpy as np
    from virtual_submeter.accuracy import microwave_background as candidate
    from virtual_submeter.data.common import DataError, inventory, read_json, sha256, utc, write_json
    dest = run / "independent_verification.json"
    if dest.exists():
        raise DataError("Independent verification exists; refuse overwrite")
    candidate.registered(run)
    candidate.check_training(run)
    candidate.check_freeze(run)
    delivery = read_json(run / "delivery_manifest.json")
    candidate.check_artifacts(run, delivery["artifacts"])
    control = candidate.control_path(run)
    spec = importlib.util.spec_from_file_location("virtual_submeter.accuracy._control_sampling", control / "source/virtual_submeter/accuracy/microwave_sampling.py")
    sampling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampling)
    p = read_json(run / "plan.json")
    data = candidate.pooled(run)
    pools = candidate.load_npz(run / "pools.npz")
    audited = read_json(run / "audit_manifest.json")
    bounds = {part["household_id"]: (utc(part["start"]), utc(part["start"])+10*86400-p["boundary_half_gap_seconds"])
              for part in audited["sources"]}
    # Independently reconstruct every eligible power template/background from raw
    # timestamps, stopping before validation; no missing-label-to-zero conversion.
    raw = {}
    for house in candidate.HOUSES:
        low, high = bounds[house]
        raw[house] = {}
        for channel, key in (("aggregate", "aggregate_power_w"), ("microwave", "power_w")):
            values = {}
            with (control / f"inputs/{house}/{channel}.csv").open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    stamp = utc(row["timestamp"])
                    if stamp >= high:
                        break
                    if stamp >= low:
                        values[stamp] = float(row[key]) if row[key] else np.nan
            raw[house][channel] = values
    checked_templates = checked_backgrounds = 0
    template_segment_support = {}
    source_support = read_json(run / "support.json")
    for h, house in enumerate(candidate.HOUSES):
        eligible = (data["house"] == h) & pools["template_ok"]
        ids = set(data["event"][eligible].tolist())
        complete = set(source_support[house]["blocks"]["train"]["complete_event_ids"])
        template_segment_support[house] = {"template_threshold_segments": len(ids),
                                           "complete_observed_template_segments": len(ids & complete),
                                           "source_complete_observed_segments": len(complete)}
    all_eligible = pools["template_ok"] | pools["background_ok"]
    for i in np.flatnonzero(all_eligible):
        house = candidate.HOUSES[int(data["house"][i])]
        low, high = bounds[house]
        t = int(data["t"][i])
        if not (low <= t-120 and t+128 <= high):
            raise DataError("Context outside strict source training")
        stamps = range(t-120, t+121, 8)
        target = np.array([raw[house]["microwave"].get(stamp, np.nan) for stamp in stamps], np.float32)
        total = np.array([raw[house]["aggregate"].get(stamp, np.nan) for stamp in stamps], np.float32)
        if not np.isfinite(target).all() or not np.isfinite(total).all() or (target < 0).any() or (total < target).any():
            raise DataError("Invalid eligible raw context")
        np.testing.assert_array_equal(total, data["x"][i])
        np.testing.assert_array_equal(target, pools["power"][i])
        np.testing.assert_array_equal(total-target, pools["residual"][i])
        if pools["template_ok"][i]:
            if target[15] != data["y"][i] or target[15] < 200:
                raise DataError("Template identity/label mismatch")
            checked_templates += 1
        if pools["background_ok"][i]:
            if not (target < 200).all():
                raise DataError("Background retains on target")
            checked_backgrounds += 1
    seeds = {}
    for seed in candidate.SEEDS:
        stream = sampling.BatchStream(data, "event", seed, p)
        exposure = np.zeros(len(data["y"]), np.int64)
        attempts = synthesized = cross_house = 0
        events = set()
        path = run / f"models/event_background/seed_{seed}"
        with (path / "augmentation.jsonl").open(encoding="utf-8") as f:
            for update in range(1, p["checkpoints"]*p["steps_per_checkpoint"]+1):
                idx = stream.next()
                np.add.at(exposure, idx, 1)
                result = data["x"][idx].copy()
                used_slots = set()
                for _ in range(8):
                    record = json.loads(next(f))
                    i, slot = record["template_index"], record["slot"]
                    if slot in used_slots or i != idx[slot] or data["y"][i] < 200 or record["update"] != update:
                        raise DataError("Template center/batch slot identity mismatch")
                    used_slots.add(slot)
                    if record["batch_indices_sha256"] != candidate.array_hash(idx):
                        raise DataError("Original task2 batch stream changed")
                    if record["template_t"] != int(data["t"][i]) or record["template_event"] != int(data["event"][i]):
                        raise DataError("Template provenance changed")
                    if record["source_data_hashes"] != pools["source_hashes"].tolist():
                        raise DataError("Source digest provenance changed")
                    if record["scale"] != 1 or record["time_shift"] != 0:
                        raise DataError("Unregistered transform")
                    attempts += 1
                    if record["status"] == "synthesized":
                        j = record["background_index"]
                        if not pools["template_ok"][i] or not pools["background_ok"][j]:
                            raise DataError("Invalid template/background identity")
                        if record["background_t"] != int(data["t"][j]):
                            raise DataError("Background timestamp changed")
                        result[slot] = pools["residual"][j]+pools["power"][i]
                        events.add((int(data["house"][i]), int(data["event"][i])))
                        synthesized += 1
                        cross_house += int(data["house"][i] != data["house"][j])
                    if record["output_sha256"] != candidate.array_hash(result[slot]):
                        raise DataError("Independent physical addition mismatch")
                if record["batch_output_sha256"] != candidate.array_hash(result):
                    raise DataError("Full batch output differs")
            if f.read():
                raise DataError("Extra augmentation log rows")
        with (path / "exposure.csv").open(encoding="utf-8", newline="") as f:
            saved_exposure = np.array([int(row["count"]) for row in csv.DictReader(f)])
        with (control / f"models/event/seed_{seed}/exposure.csv").open(encoding="utf-8", newline="") as f:
            control_exposure = np.array([int(row["count"]) for row in csv.DictReader(f)])
        np.testing.assert_array_equal(exposure, saved_exposure)
        np.testing.assert_array_equal(exposure, control_exposure)
        seeds[str(seed)] = {"attempts": attempts, "synthesized": synthesized,
                            "cross_house_synthesized": cross_house, "threshold_segments": len(events),
                            "raw_event_exposure_matches_control": True}
    result = {"status": "independently_verified", "script_sha256": sha256(Path(__file__)),
              "registration_sha256": sha256(run / "registration.json"),
              "freeze_sha256": sha256(run / "freeze.json"), "templates_raw_checked": checked_templates,
              "backgrounds_raw_checked": checked_backgrounds, "seeds": seeds,
              "template_segment_support": template_segment_support,
              "ordinary_inference": delivery["verification"],
              "ordinary_inference_receipt_sha256": sha256(run / "delivery_manifest.json")}
    write_json(dest, result)
    write_json(run / "independent_delivery_manifest.json", {"artifacts": inventory(run)})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
