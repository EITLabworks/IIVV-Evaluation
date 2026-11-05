"""Launcher/source integrity checks; no clinical or synthetic model fitting."""

import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from run_revision import main, runtime_manifest


def test_missing_quality_is_rejected_before_output_creation(tmp_path):
    output = tmp_path / "do-not-create"
    with pytest.raises(ValueError, match="requires --quality-results"):
        main(["--output", str(output)])
    assert not output.exists()


def test_runtime_inventory_contains_only_existing_current_sources():
    config = json.loads(Path("configs/revision.json").read_text())
    manifest = runtime_manifest(SimpleNamespace(synthetic=False), config)
    assert "run_all.sh" in manifest["source_hashes"]
    assert "README.md" in manifest["source_hashes"]
    assert "PROTOCOL.md" not in manifest["source_hashes"]
    assert "run_final.sh" not in manifest["source_hashes"]
    assert set(manifest["task_definitions"]) == set(config["tasks"])
    assert all(Path(name).is_file() for name in manifest["source_hashes"])


def test_launcher_help_and_empty_paths_do_not_start_computation():
    help_result = subprocess.run(
        ["bash", "run_all.sh", "--help"], capture_output=True, text=True
    )
    assert help_result.returncode == 0
    assert "--data-quality-only" in help_result.stdout
    assert "224-255" in help_result.stdout
    for flag in ("--output=", "--quality-results="):
        result = subprocess.run(
            ["bash", "run_all.sh", flag], capture_output=True, text=True
        )
        assert result.returncode == 2 and "requires a path" in result.stderr


def test_notebook_code_is_parseable():
    for path in Path(".").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:cell{index}")
