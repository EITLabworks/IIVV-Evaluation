"""Inspect revision manifests/results without reading stale legacy selection files."""

import argparse
import json
from pathlib import Path
import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Inspect patient-grouped nested revision outputs"
    )
    p.add_argument("kind", choices=["check", "evaluation", "tuning"])
    p.add_argument("--results-dir", default="results/revision-expanded")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--only", choices=["freq", "mask", "metrics"])
    p.add_argument(
        "--verify-inputs",
        action="store_true",
        help="Re-hash source/data inputs when checking completed results",
    )
    args = p.parse_args(argv)
    kind = args.kind
    root = Path(args.results_dir)
    manifest = root / "run_manifest.json"
    if not manifest.exists():
        print(
            f"No run manifest at {manifest}; execute ./run_revision.sh --preflight first."
        )
        return 2
    info = json.loads(manifest.read_text())
    print(f'Status: {info["status"]}; synthetic: {info.get("synthetic",False)}')
    if kind == "check":
        from .verification import verify_results

        try:
            print(
                json.dumps(
                    verify_results(root, verify_inputs=args.verify_inputs), indent=2
                )
            )
        except (
            ValueError,
            FileNotFoundError,
            KeyError,
            pd.errors.EmptyDataError,
        ) as exc:
            print(f"Result verification failed: {exc}")
            return 2
        print((root / "analysis_status.json").read_text())
        return 0
    if kind == "tuning" or args.only in ["freq", "mask"]:
        for path in sorted(root.glob("*/inner_search.csv")):
            frame = pd.read_csv(path)
            if args.only:
                frame = frame[
                    frame.stage == {"freq": "frequency", "mask": "mask"}[args.only]
                ]
            print(path)
            print(
                frame.sort_values("objective_auc", ascending=False)
                .head(args.top)
                .to_string(index=False)
            )
    else:
        path = root / "classification_outputs.csv"
        if not path.is_file():
            path = (
                root / "all_metrics.csv"
            )  # historical runs before the explicit metric export
        if not path.is_file():
            print(f"{path} has not been generated.")
            return 2
        print(pd.read_csv(path).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
