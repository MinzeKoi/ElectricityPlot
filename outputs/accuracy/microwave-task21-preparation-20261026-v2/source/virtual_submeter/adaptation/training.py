"""Export self-contained frozen-power models with a query-supervised state head."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from ..data.common import DataError, read_json, safe_child, sha256, source_fingerprint, write_json
from ..data.protocol import read_partition
from ..environment import environment_report
from ..models.artifacts import load_model
from .session import read_session
from .state import fit_head


def adapt(session_dir: Path, output: Path):
    started = time.perf_counter()
    if output.exists():
        raise DataError("Adaptation destination exists; choose a new model version")
    session = read_session(session_dir)
    if session["pending"] is not None:
        raise DataError("Answer or skip the pending question before freezing the adapted model")
    manifest = Path(session["manifest_path"])
    # Hash-check the actual aggregate via the adaptation-only consumer.
    read_partition(manifest, session["partition_id"], "adaptation")
    base_dir = Path(session["model_path"])
    metadata, _ = load_model(base_dir)
    rows = read_json(session_dir / "prediction/predictions.json")["rows"]
    adapter, history, counts = fit_head(rows, session["records"], metadata["appliances"],
                                      metadata["settings"]["on_thresholds_w"], session["settings"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".adapt-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        if "weights" in metadata:
            shutil.copyfile(safe_child(base_dir, metadata["weights"]["path"]), staging / "weights.pt")
            metadata["weights"] = {"path": "weights.pt", "sha256": sha256(staging / "weights.pt")}
        # With no usable answers, preserve the base decision rule exactly (B1).
        if any(counts.values()):
            metadata["state_adapter"] = adapter
            metadata["state_definition"] = "independent MIL state score >= 0.5; power output frozen"
        metadata["base_model_id"] = session["model_id"]
        metadata["base_model_manifest_sha256"] = session["model_manifest_sha256"]
        metadata["adaptation"] = "weak_window_state_only"
        metadata["adaptation_household_id"] = session["household_id"]
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        metadata["code_sha256"] = source_fingerprint()
        metadata["adaptation_settings"] = session["settings"]
        metadata["query_summary"] = {"budget_limit": session["budget"], "budget_used": len(session["records"]),
                                     "responses": dict(Counter(r["response"] for r in session["records"])),
                                     "label_sources": dict(Counter(r["label_source"] for r in session["records"])),
                                     "strategy": session["strategy"], "seed": session["seed"], "usable_windows": counts}
        metadata["target_pointwise_labels_read"] = False
        metadata["target_weak_labels_read"] = any(counts.values())
        metadata["power_output_frozen"] = True
        write_json(staging / "queries.json", {"records": session["records"], **metadata["query_summary"]})
        metadata["queries_sha256"] = sha256(staging / "queries.json")
        identity = {"base": session["model_manifest_sha256"], "queries": metadata["queries_sha256"],
                    "settings": session["settings"], "adapter": metadata.get("state_adapter"), "code": metadata["code_sha256"]}
        metadata["model_id"] = "weak-state-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        metadata["adaptation_seconds"] = time.perf_counter() - started
        write_json(staging / "model.json", metadata)
        write_json(staging / "adaptation_history.json", {"loss": history, "supervision": "window maximum; never expanded to per-point labels"})
        write_json(staging / "settings.json", session["settings"])
        environment = environment_report()
        environment.update(numpy_version=np.__version__, adaptation_execution="CPU / NumPy MIL state head")
        write_json(staging / "environment.json", environment)
        (staging / "protocol.json").write_bytes(manifest.read_bytes())
        from ..data.common import publish_directory
        publish_directory(staging, output)
    return {"status": "adapted", "model_id": metadata["model_id"], "budget_used": len(session["records"]),
            "power_output_frozen": True, "model_dir": str(output)}
