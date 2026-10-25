"""Independent source-only preparation audit; explicit named arrays only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run/"source"))
    import numpy as np
    import torch
    from virtual_submeter.accuracy import microwave_tcn as f
    from virtual_submeter.accuracy import microwave_tcn_execution as e
    from virtual_submeter.accuracy.microwave_sampling import BatchStream
    from virtual_submeter.data.common import read_json, sha256
    f.verify_preparation(run)
    plan = read_json(run/"plan.json")
    upstream = Path(read_json(run/"upstream_identity.json")["run"])
    assert sha256(upstream/"partial_mask.npz") == sha256(run/"TCN31.npz")
    # Every opened NPZ below is a named source artifact; no development data is opened.
    with np.load(run/"TCN31.npz",allow_pickle=False) as z:
        d = {k:z[k] for k in ("x","mask","y","t","event","house","raw_index")}
    with np.load(run/"TCN127.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(d["x"],z["x"][:,48:79])
        for k in ("y","t","event","house","raw_index"):
            np.testing.assert_array_equal(d[k],z[k])
        assert int(((z["outer_observed"] > 0) & (d["y"] >= 200)).sum()) == 849
    init = read_json(run/"initializations.json")
    for seed in plan["seeds"]:
        model,_ = e.configure(seed,plan)
        digest = hashlib.sha256()
        for name,v in model.state_dict().items():
            digest.update(name.encode()); digest.update(v.numpy().tobytes())
        assert digest.hexdigest() == init[str(seed)]
        stream = BatchStream(d,"event",seed,plan)
        with np.load(run/f"sampling_{seed}.npz",allow_pickle=False) as z:
            indices = z["indices"]
        for idx in indices:
            np.testing.assert_array_equal(idx,stream.next())
            assert int((d["y"][idx] >= 200).sum()) == 16
        replay = read_json(run/"dry_replay.json")["matrix"]
        assert replay[f"TCN31/{seed}"] == replay[f"TCN127/{seed}"]
    nvalidation = 0
    for i in range(6):
        assert sha256(upstream/f"validation/block_{i}.npz") == sha256(run/f"validation/TCN31/block_{i}.npz")
        with np.load(run/f"validation/TCN31/block_{i}.npz",allow_pickle=False) as z:
            nvalidation += len(z["t"])
    access = read_json(run/"preparation_access.json")
    assert access["guard_active_during_prepare"] and not access["forbidden_reads"]
    assert not (run/"training_started.json").exists() and not (run/"models").exists()
    print(json.dumps({"status":"independently_verified_source_only","actual_fits":0,
          "registration_sha256":sha256(run/"registration.json"),"validation_centers":nvalidation,
          "paired_centers_labels_masks_initializations_and_streams":True,
          "house17_truth_opened":False,"final_house_data_accessed":False},indent=2))


if __name__ == "__main__": main()
