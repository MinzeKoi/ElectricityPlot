"""Preserve original report/receipt and record a formatting-only correction."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_submeter.data.common import read_json, sha256, write_json

report = ROOT / "docs/MICROWAVE_SAMPLING_REPORT_20261025.md"
old_delivery = ROOT / "docs/MICROWAVE_SAMPLING_DELIVERY_20261025.json"
archive = ROOT / "docs/MICROWAVE_SAMPLING_REPORT_20261025.initial.md"
new_delivery = ROOT / "docs/MICROWAVE_SAMPLING_DELIVERY_20261025_v2.json"
assert not archive.exists() and not new_delivery.exists()
receipt = read_json(old_delivery)
assert sha256(report) == receipt["artifacts"]["docs/MICROWAVE_SAMPLING_REPORT_20261025.md"]
original = report.read_bytes()
archive.write_bytes(original)
text = original.decode("utf-8")
text = text.replace("|refit_19|train|14866|", "|家庭|分区|有效窗|正中心|部分支持阈值段|完整可观测段|有效小时|排除网格中心|\n|---|---|---|---|---|---|---|---|\n|refit_19|train|14866|")
text = text.replace("平均F1变化+5.23%；", "平均F1变化+5.23个百分点；").replace("平均F1变化+1.25%；", "平均F1变化+1.25个百分点；")
report.write_text(text, encoding="utf-8")
write_json(new_delivery, {"reason": "Formatting only: repeat table header after household note; label absolute F1 differences as percentage points. No metric, protocol, model, prediction or run artifact changed.",
    "original_delivery_sha256": sha256(old_delivery),
    "original_report_preserved": str(archive.relative_to(ROOT)).replace(chr(92), "/"),
    "artifacts": {str(p.relative_to(ROOT)).replace(chr(92), "/"): sha256(p) for p in
        (archive, report, old_delivery, ROOT / "scripts/summarize_microwave_sampling.py", Path(__file__))}})
print(report)
