.PHONY: manuscript paper revision response clean-manuscript

LATEXMK ?= latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -halt-on-error -cd

manuscript: paper revision response

paper:
	$(LATEXMK) $(LATEXMK_FLAGS) manuscript/Paper/main.tex

revision:
	$(LATEXMK) $(LATEXMK_FLAGS) manuscript/Paper/Revision-1.tex

response:
	$(LATEXMK) $(LATEXMK_FLAGS) manuscript/Response_Letter/resposeletter.tex

clean-manuscript:
	$(LATEXMK) -C -cd manuscript/Paper/main.tex
	$(LATEXMK) -C -cd manuscript/Paper/Revision-1.tex
	$(LATEXMK) -C -cd manuscript/Response_Letter/resposeletter.tex

.PHONY: revision-preflight revision-evaluation revision-tests rebuttal-screenshots
revision-preflight:
	./run_all.sh --preflight --output results/revision-preflight

revision-evaluation:
	./run_all.sh --output results/revision-clinical-final

revision-tests:
	python -m pytest -q tests/test_revision.py

rebuttal-screenshots:
	python scripts/create_rebuttal_screenshots.py
	$(MAKE) response
