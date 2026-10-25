"""Independent confusion/support/budget audit, guarded before any truth open."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path)
    run = p.parse_args().run.resolve()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(run/"source"))
    from virtual_submeter.accuracy import microwave_tcn as f
    f.evaluation_guard(run)  # Includes complete model/prediction/selection hash bindings.
    receipt = read(run/"evaluation_complete.json")
    for name,h in receipt["artifacts"].items():
        assert digest(run/"evaluation"/name)==h
    with np.load(run/"evaluation/truth_once.npz",allow_pickle=False) as z:
        truth,t = z["y"],z["t"]
    baseline = Path(read(run/"baseline_identity.json")["reference_run"])
    metrics = read(run/"evaluation/metrics.json")["rows"]
    for seed in (42,1019,2026):
        name="frozen_test_refit_17_0.npz"
        with np.load(baseline/f"predictions/W50/seed_{seed}"/name,allow_pickle=False) as z:
            wraw=z["raw"]; np.testing.assert_array_equal(t,z["stamps"])
        with np.load(baseline/f"reference/predictions/seed_{seed}"/name,allow_pickle=False) as z:
            raw=z["raw"]; np.testing.assert_array_equal(t,z["stamps"])
        valid=np.isfinite(truth)&np.isfinite(wraw[:,1]); assert int(valid.sum())==137523
        predicted={"A":raw[:,1]>=200,"W50":wraw[:,1]>=200}
        for arm in ("partial_mask","TCN31","TCN127"):
            with np.load(run/f"predictions/{arm}_{seed}.npz",allow_pickle=False) as z:
                np.testing.assert_array_equal(z["t"],t)
                np.testing.assert_array_equal(z["raw"],raw)
                np.testing.assert_array_equal(np.isfinite(z["scores"]),np.isfinite(wraw[:,1]))
                predicted[arm]=z["state"]==1
        actual=truth[valid]>=200
        for arm, values in predicted.items():
            state=values[valid]
            tp,fp=int((state&actual).sum()),int((state&~actual).sum())
            fn,tn=int((~state&actual).sum()),int((~state&~actual).sum())
            found=next(r for r in metrics if r["seed"]==seed and r["arm"]==arm)
            assert [found[k] for k in ("tp","fp","fn","tn","n")]==[tp,fp,fn,tn,137523]
            assert found["precision"]==tp/max(tp+fp,1)
            assert found["recall"]==tp/max(tp+fn,1)
            assert found["f1"]==2*tp/max(2*tp+fp+fn,1)
    update_count=0
    for arm in ("TCN31","TCN127"):
        for seed in (42,1019,2026):
            meta=read(run/f"models/{arm}/seed_{seed}/model.json")
            assert meta["parameter_count"]==5329 and meta["updates"]==3528
            update_count+=meta["updates"]
    assert update_count==21168
    assert read(run/"execution_authorization.json")["cumulative_maximum_fit_starts"]==7
    result={"status":"independently_verified_completed_delivery","successful_fits":6,
            "failed_source_fit_starts":1,"total_source_fit_starts":7,"total_source_updates":update_count+294,
            "confusions_precision_recall_f1_exact":True,"common_support_per_seed":137523,
            "raw_A_all_devices_exact":True,"labels_read_only_after_frozen_guard":True,
            "freeze_sha256":digest(run/"freeze.json"),"evaluation_complete_sha256":digest(run/"evaluation_complete.json")}
    (run/"independent_delivery_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
