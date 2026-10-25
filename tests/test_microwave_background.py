"""Physical synthesis, source isolation, replay and a synthetic gated lifecycle."""
import csv
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_submeter.accuracy import microwave_background as b
from virtual_submeter.accuracy import microwave_sampling as s
from virtual_submeter.data.common import DataError, iso, read_json, sha256, utc, write_json

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return read_json(ROOT / "configs/microwave_background_20261025.json")


def fixture():
    t = np.r_[np.arange(16)*8+240, np.arange(16)*8+86400+240]
    y = np.r_[np.full(8, 400), np.zeros(8), np.full(8, 600), np.zeros(8)].astype(np.float32)
    power = np.repeat(y[:, None], 31, axis=1)
    residual = np.repeat((np.arange(32)+100)[:, None], 31, axis=1).astype(np.float32)
    d = {"t": t, "house": np.repeat([0, 1], 16), "y": y, "x": power+residual,
         "event": np.where(y >= 200, np.repeat([0, 1], 16), -1)}
    pools = {"power": power, "residual": residual, "template_ok": y >= 200,
             "background_ok": y < 200, "reason": np.full(32, "eligible")}
    return d, pools


class BackgroundTests(unittest.TestCase):
    def test_physical_sum_label_identity_and_cross_house(self):
        d, pools = fixture()
        sampler = b.BatchStream(d, "event", 42, plan())
        idx = sampler.next()
        x, rows = b.BackgroundStream(d, pools, 42, plan()).apply(idx, 1)
        self.assertEqual(len(rows), 8)
        changed = {r["slot"] for r in rows}
        for r in rows:
            i, j, slot = r["template_index"], r["background_index"], r["slot"]
            self.assertEqual(i, idx[slot])
            self.assertNotEqual(d["house"][i], d["house"][j])
            np.testing.assert_array_equal(x[slot], pools["residual"][j]+pools["power"][i])
            self.assertTrue(pools["power"][i, 15] >= 200)
            self.assertEqual(r["template_event"], d["event"][i])
        for slot in set(range(256))-changed:
            np.testing.assert_array_equal(x[slot], d["x"][idx[slot]])

    def test_original_sampling_replays_and_trace_is_deterministic(self):
        d, pools = fixture()
        a, c = s.BatchStream(d, "event", 1019, plan()), b.BatchStream(d, "event", 1019, plan())
        u, v = b.BackgroundStream(d, pools, 1019, plan()), b.BackgroundStream(d, pools, 1019, plan())
        for step in range(1, 40):
            ia, ic = a.next(), c.next()
            np.testing.assert_array_equal(ia, ic)
            xa, ra = u.apply(ia, step)
            xc, rc = v.apply(ic, step)
            np.testing.assert_array_equal(xa, xc)
            self.assertEqual(ra, rc)

    def test_incomplete_template_falls_back_without_event_replacement(self):
        d, pools = fixture()
        pools["template_ok"][:] = False
        pools["reason"][:] = "missing_or_nonfinite_context"
        idx = b.BatchStream(d, "event", 2026, plan()).next()
        stream = b.BackgroundStream(d, pools, 2026, plan())
        x, rows = stream.apply(idx, 1)
        np.testing.assert_array_equal(x, d["x"][idx])
        self.assertTrue(all(r["status"] == "fallback_original" for r in rows))
        self.assertEqual(stream.summary()["synthesized"], 0)

    def test_pool_missing_negative_and_boundary_rejection(self):
        t = np.arange(8)*256+256
        d = {"t": t, "house": np.zeros(8, int), "event": np.arange(8),
             "x": np.full((8, 31), 1000, np.float32), "y": np.full(8, 400, np.float32)}
        mapping = {int(c+offset*8): 400 for c in t for offset in range(-15, 16)}
        mapping[int(t[1]-8)] = np.nan
        d["x"][2, 0] = 399
        d["x"][3, 0] = -1
        d["x"][4, 0] = np.nan
        bounds = {b.HOUSES[0]: (int(t[0]), 10000), b.HOUSES[1]: (0, 10000)}
        pools, _ = b.make_pools(d, {b.HOUSES[0]: mapping}, bounds)
        self.assertEqual(pools["reason"][:5].tolist(), ["outside_train_context", "missing_or_nonfinite_context",
                         "negative_background_residual", "negative_source_power", "missing_or_nonfinite_context"])
        self.assertEqual(int(pools["template_ok"].sum()), 3)
        d["y"][5] = 401
        with self.assertRaises(DataError):
            b.make_pools(d, {b.HOUSES[0]: mapping}, bounds)

    def test_closed_background_subtracts_subthreshold_target_and_rejects_on_context(self):
        d = {"t": np.array([256, 768]), "house": np.zeros(2, int), "event": np.full(2, -1),
             "x": np.full((2, 31), 1000, np.float32), "y": np.full(2, 10, np.float32)}
        mapping = {int(c+offset*8): 10 for c in d["t"] for offset in range(-15, 16)}
        mapping[760] = 300
        pools, _ = b.make_pools(d, {b.HOUSES[0]: mapping}, {b.HOUSES[0]: (0, 2000), b.HOUSES[1]: (0, 2000)})
        self.assertEqual(pools["background_ok"].tolist(), [True, False])
        np.testing.assert_array_equal(pools["residual"][0], np.full(31, 990))

    def test_no_cross_house_support_is_explicit_fallback_to_available_house(self):
        d, pools = fixture()
        d["house"][:] = 0
        idx = b.BatchStream(d, "event", 42, plan()).next()
        _, rows = b.BackgroundStream(d, pools, 42, plan()).apply(idx, 1)
        self.assertTrue(all(r["template_house"] == r["background_house"] for r in rows))

    def test_label_filter_does_not_parse_heldout_power(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "labels.csv"
            path.write_text("timestamp,household_id,power_w\n1970-01-01T00:00:00Z,refit_2,400\n1970-01-01T00:00:08Z,refit_2,DO_NOT_PARSE\n", encoding="utf-8")
            self.assertEqual(b.training_labels(path, "refit_2", 0, 8), {0: 400.})

    def test_frozen_selection_matches_control(self):
        d, _ = fixture()
        scores = np.where(d["y"] >= 200, .8, .1)
        blocks, sc = {"a": d, "b": d}, {"a": scores, "b": scores}
        self.assertEqual(b.selection(sc, blocks, plan()), s.selection(sc, blocks, plan()))

    def test_forbidden_house_and_outer_training_access(self):
        with patch.object(b, "load_npz", side_effect=AssertionError("no read")):
            for h, role in (("refit_17", "train"), ("refit_8", "train"), ("refit_9", "train"),
                            ("refit_20", "train"), ("refit_2", "outer")):
                with self.assertRaises(DataError):
                    b.source_array(Path("unused"), h, role)

    def test_zero_support_never_opens_controls_or_starts_fit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_json(root / "registration.json", {"preparation_passed": False})
            with patch.object(b, "control_path", side_effect=AssertionError("must stop before dependency")):
                with self.assertRaises(DataError):
                    b.require_controls_complete(root, {})
            self.assertFalse((root / "training_started.json").exists())

    def test_synthetic_lifecycle_and_gates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp = read_json(ROOT / "configs/microwave_sampling_20261025.json")
            cp.update(checkpoints=2, steps_per_checkpoint=1)
            pp = root / "control_plan.json"
            write_json(pp, cp)
            audit = root / "audit.json"
            write_json(audit, {"status": "synthetic_only"})
            m = {"schema": "microwave_sampling_audit_v1", "status": "ready",
                 "audit": {"path": audit.name, "sha256": sha256(audit)}, "sources": []}
            start = utc("2014-10-01T00:00:00Z")
            for h in s.HOUSES:
                t = np.concatenate([start+day*86400+offset+np.arange(160)*8 for day, offset in ((0, 0), (10, 3600), (11, 3600), (12, 3600))])
                y = np.zeros(len(t))
                for j in range(4):
                    y[j*160+30:j*160+37] = 400
                    y[j*160+90:j*160+97] = 400
                artifacts = {}
                for channel in ("aggregate", "microwave"):
                    path = root / f"{h}_{channel}.csv"
                    with path.open("w", encoding="utf-8", newline="") as f:
                        key = "aggregate_power_w" if channel == "aggregate" else "power_w"
                        w = csv.DictWriter(f, fieldnames=["timestamp", "household_id", key])
                        w.writeheader()
                        w.writerows({"timestamp": iso(int(stamp)), "household_id": h, key: float(y[i]+1000 if channel == "aggregate" else y[i])} for i, stamp in enumerate(t))
                    artifacts[channel] = {"path": path.name, "sha256": sha256(path)}
                m["sources"].append({"household_id": h, "start": iso(start), "end": iso(start+14*86400), "artifacts": artifacts})
            mp, control, run = root / "manifest.json", root / "control", root / "candidate"
            write_json(mp, m)
            with patch.object(s, "validate_plan"):
                s.prepare(control, pp, mp)
            bp = plan()
            bp.update(checkpoints=2, steps_per_checkpoint=1)
            bp_path = root / "background.json"
            write_json(bp_path, bp)
            with patch.object(b, "validate_plan"):
                result = b.prepare(run, bp_path, control)
            self.assertEqual(result["status"], "prepared_not_trained")
            self.assertFalse((run / "outer_labels").exists())
            with self.assertRaises(DataError):
                b.evaluate(run)
            self.assertFalse((run / "evaluation").exists())
            auth = root / "execute.json"
            payload = {"execute": True, "maximum_fits": 3, "registration_sha256": sha256(run / "registration.json"),
                       "audit_sha256": sha256(audit), "coordinator_execution_message": "synthetic tests only"}
            write_json(auth, payload)
            with self.assertRaises(DataError):
                b.train(run, auth)
            self.assertFalse((run / "training_started.json").exists())
            ca = root / "control_execute.json"
            write_json(ca, {**payload, "maximum_fits": 6, "registration_sha256": sha256(control / "registration.json")})
            s.train(control, ca)
            s.freeze(control)
            s.evaluate(control)
            for name in ("training_complete", "freeze", "evaluation_complete"):
                payload["control_"+name+"_sha256"] = sha256(control / (name+".json"))
            write_json(auth, payload)
            b.train(run, auth)
            b.freeze(run)
            b.evaluate(run)
            self.assertEqual(b.verify(run)["prediction_readbacks"], 18)
            self.assertEqual(b.report(run)["status"], "reported")
            independent = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/verify_microwave_background.py"),
                                          "--run", str(run)], capture_output=True, text=True)
            self.assertEqual(independent.returncode, 0, independent.stdout+independent.stderr)
            self.assertEqual(read_json(run / "independent_verification.json")["status"], "independently_verified")
            for action in (lambda: b.prepare(run, bp_path, control), lambda: b.train(run, auth), lambda: b.freeze(run), lambda: b.evaluate(run), lambda: b.report(run)):
                with self.assertRaises(DataError):
                    action()
            prediction = run / "predictions/event_background_42_refit_2.npz"
            prediction.write_bytes(b"tampered")
            with self.assertRaises(DataError):
                b.check_freeze(run)


if __name__ == "__main__":
    unittest.main()
