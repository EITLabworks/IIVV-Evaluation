"""Diagnostic checks with saved fixtures only; no model imports or training."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "diagnose_revision", Path(__file__).resolve().parents[1] / "scripts/diagnose_revision.py"
)
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


class DiagnoseRevisionTests(unittest.TestCase):
    def test_progress_uses_last_observation_and_detects_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.log"
            path.write_text(
                "MC/nested_cv_repeat0_fold2: 2000 candidates completed\n"
                "MC/nested_cv_repeat0_fold2: 250 candidates completed\n"
                "MC: nested_cv_repeat0_fold3 completed (5036 candidates)\n"
            )
            rows = diagnose.log_progress(path)
            self.assertEqual(rows[("MC", "nested_cv_repeat0_fold2")],
                             {"count": 250, "reset": True})
            self.assertEqual(rows[("MC", "nested_cv_repeat0_fold3")]["count"], 5036)

    def test_reused_outer_fits_are_counted_once(self):
        row = {"candidate": {"model": "lr", "params_json": "{}"}}
        self.assertEqual(len(diagnose.saved_fits([row, dict(row)])), 1)

    def test_read_only_report_and_tampered_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            done = run / "fold-checkpoints/MC/nested_cv_repeat0_fold0"
            done.mkdir(parents=True)
            (run / "run_manifest.json").write_text(json.dumps({
                "started_utc": "2026-01-01T00:00:00+00:00", "status": "running", "config": {}
            }))
            (run / "parallel_execution.json").write_text(json.dumps({"fold_order": [
                {"task": "MC", "key": [0, i, "nested_cv"]} for i in [0, 2]
            ]}))
            inventory = {}
            for name in diagnose.REQUIRED:
                path = done / name
                path.write_text("[]" if name.endswith(".json") else "x\n")
                inventory[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (done / "complete.json").write_text(json.dumps({
                "task": "MC", "fold_key": [0, 0, "nested_cv"],
                "elapsed_seconds": 7200, "artifacts": inventory,
            }))
            log = root / "run.log"
            log.write_text("MC/nested_cv_repeat0_fold2: 2000 candidates completed\n")
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            args = ["--results-dir", str(run), "--log", str(log), "--folds", "MC:0:2"]
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(diagnose.main(args), 0)
            self.assertIn("2.00/2.00/2.00 h", output.getvalue())
            self.assertIn("Restlaufzeit: nicht belastbar", output.getvalue())
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            (done / "selections.json").write_text("changed")
            self.assertEqual(diagnose.main(args), 2)


if __name__ == "__main__":
    unittest.main()
