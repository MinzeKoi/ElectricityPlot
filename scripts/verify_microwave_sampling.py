"""Independent arithmetic and exposure readback for the sealed sampling run."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / "source"))
    from virtual_submeter.accuracy import microwave_sampling as s
    from virtual_submeter.data.common import read_json, sha256
    result = s.verify(run)
    p = read_json(run / "plan.json")
    rows = read_json(run / "evaluation/results.json")["rows"]
    source = s.pooled(run)
    label = source["y"] >= 200
    exposure_checks = []
    for arm, seed, model_path in s.model_dirs(run):
        with (model_path / "exposure.csv").open(encoding="utf-8", newline="") as f:
            exposure_rows = list(csv.DictReader(f))
        np.testing.assert_array_equal([int(r["t"]) for r in exposure_rows], source["t"])
        actual = np.array([int(r["count"]) for r in exposure_rows])
        sampler = s.BatchStream(source, arm, seed, p)
        expected = np.zeros(len(label), np.int64)
        for _ in range(p["checkpoints"]*p["steps_per_checkpoint"]):
            idx = sampler.next()
            if arm == "event":
                assert int(label[idx].sum()) == 16
            np.add.at(expected, idx, 1)
        np.testing.assert_array_equal(actual, expected)
        with (model_path / "batches.csv").open(encoding="utf-8", newline="") as f:
            batches = list(csv.DictReader(f))
        assert sum(int(r["positive"]) for r in batches) == int(actual[label].sum())
        assert sum(int(r["negative"]) for r in batches) == int(actual[~label].sum())
        assert len(batches) == 3528
        for row in batches:
            a, b = float(row["positive_loss_sum"]), float(row["negative_loss_sum"])
            assert abs(float(row["positive_effective_loss_fraction"])-a/(a+b)) < 1e-12
        exposure_checks.append({"arm": arm, "seed": seed, "updates": len(batches),
                                "positive_exposure": int(actual[label].sum()),
                                "negative_exposure": int(actual[~label].sum()),
                                "exact_sampler_replay": True})
    # Arithmetic below deliberately does not call experiment.metrics or pr_curve.
    checked = 0
    for row in rows:
        arm, seed, partition = row["arm"], row["seed"], row["partition"]
        if not partition.endswith("/outer"):
            continue
        house = partition.split("/")[0]
        with np.load(run / f"outer_labels/{house}.npz") as a:
            y = a["y"] >= 200
        with np.load(run / f"predictions/{arm}_{seed}_{house}.npz") as a:
            scores = a["scores"]
        pred = scores >= row["threshold"]
        counts = [int((pred & y).sum()), int((pred & ~y).sum()),
                  int((~pred & y).sum()), int((~pred & ~y).sum())]
        assert counts == [row[k] for k in ("tp", "fp", "fn", "tn")]
        tp, fp, fn, tn = counts
        assert row["precision"] == (tp/(tp+fp) if tp+fp else 0)
        assert row["recall"] == tp/(tp+fn)
        assert row["off_false_positive_rate"] == fp/(fp+tn)
        order = np.argsort(-scores, kind="stable")
        sorted_s, sorted_y = scores[order], y[order]
        ends = np.r_[np.where(np.diff(sorted_s) != 0)[0], len(y)-1]
        cumulative = np.cumsum(sorted_y)[ends]
        ap = float(np.sum(np.diff(np.r_[0., cumulative/y.sum()])*cumulative/(ends+1)))
        assert abs(row["ap"]-ap) < 1e-12
        checked += 1
    # Independent context/event isolation using actual manifest center identities.
    with (run / "window_manifest.csv").open(encoding="utf-8", newline="") as f:
        windows = list(csv.DictReader(f))
    from datetime import datetime
    stats = read_json(run / "support.json")
    events = read_json(run / "events.json")["events"]
    for house in s.HOUSES:
        role_centers = {}
        for role, block in stats[house]["blocks"].items():
            low = int(datetime.fromisoformat(block["start"].replace("Z", "+00:00")).timestamp())
            high = int(datetime.fromisoformat(block["end"].replace("Z", "+00:00")).timestamp())
            centers = np.array([int(datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp())
                                for r in windows if r["house"] == house and r["role"] == role])
            assert (centers-120 >= low).all() and (centers+128 <= high).all()
            role_centers[role] = centers
            for e in events:
                if e["house"] == house and (e["start"] < low or e["end"] > high):
                    assert not ((centers >= e["start"]) & (centers < e["end"])).any()
        roles = list(role_centers)
        for i, a in enumerate(roles):
            for b in roles[i+1:]:
                assert not len(np.intersect1d(role_centers[a], role_centers[b]))
    output = {**result, "status": "independently_verified", "metric_checks": checked,
              "exposure_checks": exposure_checks, "context_event_isolation": True,
              "registration_sha256": sha256(run / "registration.json"),
              "freeze_sha256": sha256(run / "freeze.json"),
              "verifier_sha256": sha256(Path(__file__)),
              "scope": "local automated independent arithmetic, not a second investigator or machine"}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
