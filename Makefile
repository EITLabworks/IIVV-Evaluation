.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.PHONY: help

PYTHON ?= python
REVISION_CONFIG ?= configs/revision.json
REVISION_OUTPUT ?= results/revision-expanded
PREFLIGHT_OUTPUT ?= results/revision-expanded-preflight

help:
	@printf '%s\n' 'Run sequentially: revision-tests, revision-preflight, revision-evaluation, revision-check.' 'Use revision-resume for an interrupted run. See SERVER_RUN.md.'

.PHONY: revision-preflight revision-evaluation revision-resume revision-check revision-tests
revision-preflight:
	./run_all.sh --config "$(REVISION_CONFIG)" --preflight --output "$(PREFLIGHT_OUTPUT)"

revision-evaluation:
	./run_all.sh --config "$(REVISION_CONFIG)" --output "$(REVISION_OUTPUT)"

revision-resume:
	./run_all.sh --config "$(REVISION_CONFIG)" --resume --output "$(REVISION_OUTPUT)"

revision-check:
	$(PYTHON) -m revision.report_cli check --results-dir "$(REVISION_OUTPUT)" --verify-inputs

revision-tests:
	$(PYTHON) -m pytest -q tests/test_revision.py

# Optional local document targets; evaluation runs do not require this file.
-include manuscript.mk
