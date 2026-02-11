#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


TASKS = ["healthy_vs_cancer", "cancer_vs_lr_hr", "MC"]
METRICS = ["auc", "acc", "f1"]


def _read_ranking_table(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _format_mask_row(row: pd.Series) -> str:
    # Mask ranking CSV schema: IIVV, mean_score, std_score, n_runs, n_params
    iivv = str(row.get("IIVV", ""))
    mean_s = row.get("mean_score", None)
    std_s = row.get("std_score", None)
    n_runs = row.get("n_runs", None)
    n_params = row.get("n_params", None)

    mean_txt = f"{mean_s:.4f}" if pd.notna(mean_s) else "nan"
    std_txt = f"{std_s:.4f}" if pd.notna(std_s) else "nan"
    n_runs_txt = f"{int(n_runs)}" if pd.notna(n_runs) else "?"
    n_params_txt = f"{int(n_params)}" if pd.notna(n_params) else "?"

    return f"{iivv:<18}  mean={mean_txt}  std={std_txt}  runs={n_runs_txt}  params={n_params_txt}"


def _format_freq_row(row: pd.Series) -> str:
    # Freq ranking CSV schema: N_freqs, mean_score, std_score, n_runs
    n = row.get("N_freqs", None)
    mean_s = row.get("mean_score", None)
    std_s = row.get("std_score", None)
    n_runs = row.get("n_runs", None)

    n_txt = f"N{int(n):02d}" if pd.notna(n) else "N??"
    mean_txt = f"{mean_s:.4f}" if pd.notna(mean_s) else "nan"
    std_txt = f"{std_s:.4f}" if pd.notna(std_s) else "nan"
    n_runs_txt = f"{int(n_runs)}" if pd.notna(n_runs) else "?"

    return f"{n_txt:<6}  mean={mean_txt}  std={std_txt}  runs={n_runs_txt}"


def _print_block_header(title: str, width: int = 86):
    line = "─" * width
    print("\n" + line)
    print(title)
    print(line)


def _print_table(
    title: str,
    df: pd.DataFrame,
    formatter,
    top_k: int,
    width: int = 86,
):
    _print_block_header(title, width=width)

    if df is None or df.empty:
        print("No data found.")
        return

    # Ensure sorted best->worst (already should be)
    df_show = df.head(top_k).copy()

    for i, (_, row) in enumerate(df_show.iterrows(), start=1):
        print(f"{i:>2d}. {formatter(row)}")

    if len(df) > top_k:
        print(f"... ({len(df) - top_k} more)")


def _find_latest_results_dir(path: str) -> Path:
    return Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Print MaskSel and FreqSel rankings for all tasks/metrics."
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--top", type=int, default=5, help="Show top-k per ranking")
    parser.add_argument(
        "--only",
        choices=["all", "mask", "freq"],
        default="all",
        help="Print only mask or freq rankings",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="auc,acc,f1",
        help="Comma-separated metrics to print (default: auc,acc,f1)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="healthy_vs_cancer,cancer_vs_lr_hr,MC",
        help="Comma-separated tasks to print",
    )
    args = parser.parse_args()

    results_dir = _find_latest_results_dir(args.results_dir)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    width = 86
    print("\n" + "=" * width)
    print("RANKINGS SUMMARY (good → bad)")
    print(f"results_dir: {results_dir.resolve()}")
    print("=" * width)

    for task in tasks:
        _print_block_header(f"TASK: {task}", width=width)

        for metric in metrics:
            # MASK
            if args.only in ("all", "mask"):
                mask_csv = results_dir / f"MaskSel_ranking_{task}_{metric}.csv"
                df_mask = _read_ranking_table(mask_csv)
                _print_table(
                    title=f"MaskSel | metric={metric} | file={mask_csv.name}",
                    df=df_mask,
                    formatter=_format_mask_row,
                    top_k=args.top,
                    width=width,
                )

            # FREQ
            if args.only in ("all", "freq"):
                freq_csv = results_dir / f"FreqSel_ranking_{task}_{metric}.csv"
                df_freq = _read_ranking_table(freq_csv)
                _print_table(
                    title=f"FreqSel | metric={metric} | file={freq_csv.name}",
                    df=df_freq,
                    formatter=_format_freq_row,
                    top_k=args.top,
                    width=width,
                )

    print("\nDone.\n")


if __name__ == "__main__":

    # chmod +x print_rankings.py
    # ./print_rankings.py --top 10

    # ./print_rankings.py --only mask
    # ./print_rankings.py --only freq
    # ./print_rankings.py --metrics auc,f1 --tasks healthy_vs_cancer,MC --top 5

    main()
