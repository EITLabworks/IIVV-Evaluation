#!/usr/bin/env python3
"""Classifier tuning: routed through the shared nested patient-level workflow.

The former stand-alone cohort-wide optimization has been replaced. Basic --task,
--metric, --model, --n_trials and --n_folds options remain aliases. See
code-revisions.md; the recommended command is ./run_revision.sh.
"""
from revision.compat import main

if __name__ == '__main__':
    raise SystemExit(main('Classifier tuning'))
