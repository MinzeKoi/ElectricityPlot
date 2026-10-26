"""Portable R4 replay evidence; never calls a target submeter/evaluator."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys

from ..data.common import DataError, read_json, sha256, write_json
from ..environment import environment_report
from ..models.prediction import predict
from . import live
from .dynamic_session import digest


def runtime():
    return {**environment_report(), "python_prefix": sys.prefix,
            "packages": {name: version(name) for name in ("numpy", "torch", "streamlit", "pandas", "altair")}}


def signatures(payload, directory):
    return {"rows_sha256": digest(payload["rows"]), "model_id": payload["model_id"],
            "report_sha256": sha256(directory / "report.html"),
            "csv_sha256": sha256(directory / "predictions.csv")}


def build(session, output, repository, historical_run=None):
    if output.exists():
        raise DataError("Release destination exists; use a fresh directory")
    current = live.status(session)
    if current["budget_used"] != current["fitted_budget"]:
        raise DataError("Release needs a completed five-answer update")
    output.mkdir(parents=True)
    shutil.copytree(session / "inputs", output / "inputs")
    shutil.copytree(repository / "src", output / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output / "scripts").mkdir()
    for name in ("replay_release.py", "app.py"):
        shutil.copyfile(repository / "scripts" / name, output / "scripts" / name)
    for name in ("requirements-m4-cpu.lock", "requirements-m2-cpu.lock", "pyproject.toml"):
        shutil.copyfile(repository / name, output / name)
    shutil.copytree(repository / "configs", output / "configs")
    (output / "docs").mkdir()
    for name in ("REFERENCES.md", "DATA_CONTRACT.md", "R4_R5_HANDOFF.md", "R2_STABILITY_HANDOFF.md", "R2_STABILITY_RESULTS.json"):
        path = repository / "docs" / name
        if path.exists():
            shutil.copyfile(path, output / "docs" / name)
    diff = subprocess.run(["git", "diff", "--binary"], cwd=repository, capture_output=True, check=False)
    (output / "working-tree.patch").write_bytes(diff.stdout)
    write_json(output / "provenance.json", {"git_diff_available": diff.returncode == 0,
               "source_snapshot": "src", "notice": "All shipped source/config bytes, including untracked files, are bound by manifest.json. Historical R2 retains its own frozen source and negative results."})
    write_json(output / "records.json", {"records": current["records"]})
    if historical_run is not None:
        history = output / "historical_r2"
        history.mkdir()
        for name in ("run.json", "freeze.json", "evidence.json", "summary.json", "comparison.csv", "report.html"):
            path = historical_run / name
            if path.exists():
                shutil.copyfile(path, history / name)
        shutil.copytree(historical_run / "frozen/source", history / "source")
        (history / "README.md").write_text(
            "Historical R2 evidence excerpt, retaining its execution source. Full-run inventory files reference the original complete archive; "
            "this excerpt does not claim full historical verification without that archive. Use the shipped current code only for the live replay.\n", encoding="utf-8")
    expected = {}
    for label, model in (("before", session / "session/models/budget_0"), ("after", Path(current["model"]))):
        destination = output / "evidence" / label
        predict(model, destination, session / "inputs/protocol/protocol.json", current["partition_id"])
        payload = read_json(destination / "predictions.json")
        expected[label] = signatures(payload, destination)
    metadata = read_json(Path(current["model"]) / "model.json")
    write_json(output / "expected.json", {"predictions": expected, "adapter": metadata["accuracy_state_head"],
               "model_id": metadata["model_id"], "service": {k: current[k] for k in
               ("partition_id", "strategy", "seed", "budget", "learner", "fitted_budget")}})
    (output / "README.md").write_text(
        "# ElectricityPlot R4/R5 portable replay\n\n"
        "Development demonstration only. R3 final evaluation was skipped. No final-household files or target submeter values are included.\n\n"
        "Python 3.11+; create an environment and install requirements-m4-cpu.lock. From this directory run:\n\n"
        "```powershell\npython scripts/replay_release.py --output ../replay-result\n"
        "python -m streamlit run scripts/app.py\n```\n\n"
        "The UI defaults to bundled source/protocol. Choose its adaptation partition and load the newly replayed session at ../replay-result/session to inspect the same records.\n\n"
        "Replay verifies all file hashes, regenerates every query and five-answer fit, compares source/adapted predictions and rebuilds HTML/CSV. "
        "The supplied before/after reports and predictions are under evidence/. See docs/REFERENCES.md for REFIT licensing and attribution; confirm its redistribution terms before sharing. "
        "Simulated records remain simulated; a successful automated replay does not establish another member or machine has reproduced it.\n",
        encoding="utf-8")
    write_json(output / "manifest.json", {"schema_version": 1, "usage": "development_demo_only",
               "r3": "skipped_by_user", "independent_member_reproduction": "pending",
               "created_at": datetime.now(timezone.utc).isoformat(), "environment": runtime(),
               "files_sha256": live.inventory(output)})
    return {"status": "release_built", "path": str(output), "files": len(read_json(output / "manifest.json")["files_sha256"])}


def verify(bundle):
    record = read_json(bundle / "manifest.json")
    actual = live.inventory(bundle)
    actual.pop("manifest.json", None)
    # Importing the shipped code may create bytecode before verification runs.
    actual = {k: v for k, v in actual.items() if "__pycache__" not in Path(k).parts and not k.endswith(".pyc")}
    if record.get("schema_version") != 1 or actual != record["files_sha256"]:
        raise DataError("Release file inventory differs; do not replay changed inputs")
    return record


def replay(bundle, output):
    # Keep writable replay results outside the immutable bundle.
    if output.resolve().is_relative_to(bundle.resolve()) or output.exists():
        raise DataError("Choose a new replay output directory outside the sealed bundle")
    verify(bundle)
    expected = read_json(bundle / "expected.json")
    config = expected["service"]
    inputs = bundle / "inputs"
    directory = output / "session"
    live.create(directory, inputs / "model", inputs / "protocol/protocol.json", config["partition_id"],
                inputs / "learner.json", inputs / "query.json", inputs / "legacy.json",
                config["strategy"], config["seed"], config["budget"],
                inputs / "source_head" if (inputs / "source_head").exists() else None)
    for record in read_json(bundle / "records.json")["records"]:
        pending = live.status(directory)["pending"]
        if pending is None or any(pending[k] != record[k] for k in pending):
            raise DataError("Replay selected a different question/model version")
        actual = live.answer(directory, pending["query_id"], record["response"], record["label_source"])
        if actual != record:
            raise DataError("Replay changed a confirmation record")
    current = live.status(directory)
    metadata = read_json(Path(current["model"]) / "model.json")
    if metadata["accuracy_state_head"] != expected["adapter"] or metadata["model_id"] != expected["model_id"]:
        raise DataError("Replay adapter differs")
    results = {}
    for label, model in (("before", directory / "session/models/budget_0"), ("after", Path(current["model"]))):
        destination = output / label
        predict(model, destination, directory / "inputs/protocol/protocol.json", current["partition_id"])
        payload = read_json(destination / "predictions.json")
        results[label] = signatures(payload, destination)
        if results[label] != expected["predictions"][label]:
            raise DataError(f"Replay prediction/report differs: {label}")
    receipt = {"status": "portable_replay_verified", "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
               "confirmed_answers": current["budget_used"], "fitted_budget": current["fitted_budget"],
               "predictions": results, "environment": runtime(), "python_executable": sys.executable,
               "verification_kind": "automated_relocated_replay", "independent_member_reproduction": False,
               "second_machine": False, "full_training_reproduction": False,
               "target_pointwise_labels_read": False, "formal_evaluation": False}
    write_json(output / "receipt.json", receipt)
    return receipt
