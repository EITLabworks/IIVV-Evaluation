.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.PHONY: help

PYTHON ?= python
REVISION_OUTPUT ?= results/revision-three-tasks
QUALITY_OUTPUT ?= results/revision-quality

help:
	@printf '%s\n' 'Run revision-quality, inspect Data-Quality.ipynb, then revision-evaluation and revision-check.' 'See README.md. The launcher runs regression checks before tuning.'

.PHONY: revision-quality revision-evaluation revision-resume revision-check revision-tests
revision-quality:
	./run_all.sh --data-quality-only --output "$(QUALITY_OUTPUT)"

revision-evaluation:
	./run_all.sh --quality-results "$(QUALITY_OUTPUT)" --output "$(REVISION_OUTPUT)"

revision-resume:
	./run_all.sh --quality-results "$(QUALITY_OUTPUT)" --resume --output "$(REVISION_OUTPUT)"

revision-check:
	$(PYTHON) -m revision.report_cli check --results-dir "$(REVISION_OUTPUT)" --verify-inputs

revision-tests:
	$(PYTHON) -m pytest -q tests

# Optional local document targets; evaluation runs do not require this file.
-include manuscript.mk
