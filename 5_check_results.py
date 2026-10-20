#!/usr/bin/env python3
"""Inspect the new nested revision artifacts; see --help."""

from revision.report_cli import main

if __name__ == "__main__":
    raise SystemExit(main("check"))
