.DEFAULT_GOAL := manuscript
.DELETE_ON_ERROR:
.PHONY: manuscript paper revision revision-clean response clean-manuscript

PYTHON ?= python
LATEXMK ?= latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -halt-on-error -cd
REVISION_CONFIG ?= configs/revision.json
REVISION_OUTPUT ?= results/revision-expanded
PREFLIGHT_OUTPUT ?= results/revision-expanded-preflight

manuscript: paper revision revision-clean response

paper:
	$(LATEXMK) $(LATEXMK_FLAGS) manuscript/Paper/main.tex

revision:
	$(LATEXMK) $(LATEXMK_FLAGS) manuscript/Paper/Revision-1.tex

manuscript/Paper/Revision-1-clean.tex: templates/Revision-1-clean.tex
	cp $< $@

revision-clean: manuscript/Paper/Revision-1-clean.tex
	$(LATEXMK) $(LATEXMK_FLAGS) manuscript/Paper/Revision-1-clean.tex

response:
	$(LATEXMK) $(LATEXMK_FLAGS) manuscript/Response_Letter/resposeletter.tex

clean-manuscript:
	$(LATEXMK) -C -cd manuscript/Paper/main.tex
	$(LATEXMK) -C -cd manuscript/Paper/Revision-1.tex
	$(LATEXMK) -C -cd manuscript/Paper/Revision-1-clean.tex
	$(LATEXMK) -C -cd manuscript/Response_Letter/resposeletter.tex

.PHONY: revision-preflight revision-evaluation revision-resume revision-check revision-tests rebuttal-screenshots
revision-preflight:
	./run_all.sh --config "$(REVISION_CONFIG)" --preflight --output "$(PREFLIGHT_OUTPUT)"

revision-evaluation:
	./run_all.sh --config "$(REVISION_CONFIG)" --output "$(REVISION_OUTPUT)"

revision-resume:
	./run_all.sh --config "$(REVISION_CONFIG)" --resume --output "$(REVISION_OUTPUT)"

revision-check:
	$(PYTHON) 5_check_results.py --results-dir "$(REVISION_OUTPUT)" --verify-inputs

revision-tests:
	$(PYTHON) -m pytest -q tests/test_revision.py

rebuttal-screenshots:
	$(PYTHON) scripts/create_rebuttal_screenshots.py
	$(MAKE) response
