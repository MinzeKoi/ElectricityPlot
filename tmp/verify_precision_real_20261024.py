"""Additional real-data checks; development input is aggregate-only throughout."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.accuracy import precision
from virtual_submeter.data.common import read_json, sha256, write_json
from virtual_submeter.data.protocol import read_partition
from virtual_submeter.models.artifacts import load_model
from virtual_submeter.models.prediction import predict_rows
from virtual_submeter.models.training import dataset, negative_sample_factors

run = ROOT / "outputs/accuracy/precision-20261024-v1"
manifest = run / "protocol/protocol.json"
protocol = read_json(manifest)
frozen = precision.validate_frozen(run)
report = {"freeze_sha256": sha256(run / "freeze.json"), "development_labels_read": False,
          "final_households_accessed": False, "models": [], "negative_pools": [], "full_inference": []}
for path in sorted((run / "models").glob("*/seed_*/model.json")):
    meta = read_json(path)
    reference = read_json(run / f"reference/models/{path.parent.name}/model.json")
    for key in ("input_log_mean", "input_log_std", "power_scales_w", "source_train_windows", "source_validation_windows"):
        assert meta[key] == reference[key], (path, key)
    assert meta["optimizer_steps"] == 7044
    assert meta["settings"]["window_samples"] == 31 and meta["settings"]["on_thresholds_w"]["microwave"] == 200
    assert meta["state_threshold_view"] == "raw_power_w" and not meta["target_labels_read"]
    report["models"].append({"candidate": path.parent.parent.name, "seed": meta["settings"]["seed"],
                             "selected_checkpoint": meta["selected_epoch"], "optimizer_steps": meta["optimizer_steps"]})
assert len(report["models"]) == 12

metadata = read_json(run / "models/W50/seed_42/model.json")
parts = [p for p in protocol["partitions"] if p["role"] == "source_train"]
x, y, _, keys = dataset(manifest, parts, ["microwave"], metadata["settings"], 8, return_keys=True)
for seed in precision.SEEDS:
    pool = read_json(run / f"negative_windows/seed_{seed}.json")
    meta, model = load_model(run / f"models/W50/seed_{seed}")
    assert pool["source_model_sha256"] == sha256(run / f"models/W50/seed_{seed}/model.json")
    scores = precision._predict_values(meta, model, x)[:, 0]
    expected = np.isfinite(y[:, 0]) & (y[:, 0] < 200) & (scores >= 200)
    weights = negative_sample_factors(keys, y, ["microwave"], [200], pool)
    np.testing.assert_array_equal(weights[:, 0] == 4, expected)
    report["negative_pools"].append({"seed": seed, "windows": len(pool["windows"]),
                                    "full_source_mask_reproduced": True, "counts": pool["counts_by_household"]})
del x, y

rows = read_partition(manifest, precision.DEVELOPMENT, "inference")["aggregate"]
for group in ("W50", "W10"):
    meta, model = load_model(run / f"models/{group}/seed_42")
    ordinary = predict_rows(meta, model, rows)
    actual = np.asarray([r["appliances"]["microwave"]["raw_power_w"] if r["appliances"]["microwave"]["raw_power_w"] is not None else np.nan for r in ordinary])
    states = [r["appliances"]["microwave"]["state"] for r in ordinary]
    case = next(c for c in frozen["cases"] if c["candidate"] == group and c["seed"] == 42 and c["partition"] == precision.DEVELOPMENT)
    with np.load(run / case["path"], allow_pickle=False) as a:
        expected = a["raw"][:, 1]
        np.testing.assert_array_equal(actual, expected)
        assert all(s is None if not np.isfinite(p) else s == bool(p >= 200) for s, p in zip(states, expected))
    report["full_inference"].append({"candidate": group, "seed": 42, "rows": len(rows),
                                     "maximum_raw_power_difference_w": 0, "all_states_and_missingness_equal": True})
    del ordinary
write_json(ROOT / "tmp/precision-real-readback-20261024.json", report)
print(json.dumps({"models": len(report["models"]), "negative_pools": report["negative_pools"],
                  "full_inference": report["full_inference"], "development_labels_read": False}, indent=2))
