#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd


def detect_score_col(df: pd.DataFrame) -> str:
    for c in ["score_mean", "score", "metric_mean", "val_score"]:
        if c in df.columns:
            return c
    raise ValueError(f"No score column found in columns={list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(description="Read & print all tuning CSV results.")
    ap.add_argument("--path", default="results", help="Folder with tuning CSV files")
    ap.add_argument(
        "--pattern", default="GPU_tuning*.csv", help="Glob pattern (default: *.csv)"
    )
    ap.add_argument("--top", type=int, default=4, help="How many top rows to print")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.path, args.pattern)))

    if not files:
        print(f"No CSV files found in {args.path} with pattern {args.pattern}")
        return

    print(f"\nFound {len(files)} CSV files\n")

    all_frames = []

    # ---- print each file ----
    for f in files:
        print("=" * 80)
        print(f"FILE: {f}")
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  -> Failed to read: {e}")
            continue

        if df.empty:
            print("  -> empty")
            continue

        try:
            score_col = detect_score_col(df)
            df_sorted = df.sort_values(score_col, ascending=False)
        except Exception:
            df_sorted = df

        print(df_sorted.head(args.top).to_string(index=False))
        print()

        df["__source_file"] = os.path.basename(f)
        all_frames.append(df)

    # ---- global summary ----
    if not all_frames:
        print("No readable CSV content.")
        return

    big = pd.concat(all_frames, ignore_index=True)

    print("\n" + "#" * 80)
    print("GLOBAL TOP RESULTS")
    print("#" * 80)

    try:
        score_col = detect_score_col(big)
        big_sorted = big.sort_values(score_col, ascending=False)
        print(big_sorted.head(args.top).to_string(index=False))
    except Exception as e:
        print(f"Could not compute global ranking: {e}")
        return

    # ---- grouped best ----
    group_cols = [
        c for c in ["model", "class_task", "fsel", "msel"] if c in big.columns
    ]

    if group_cols:
        print("\n" + "#" * 80)
        print("BEST PER (model, class_task, fsel, msel)")
        print("#" * 80)

        idx = big.groupby(group_cols)[score_col].idxmax()
        best_per_group = big.loc[idx].sort_values(score_col, ascending=False)
        print(best_per_group.head(args.top * 2).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
