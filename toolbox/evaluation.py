import pandas as pd
from typing import List, Sequence, Mapping, Tuple, Union, Optional
import numpy as np


def evaluate_combinations(
    df: pd.DataFrame,
    *,
    groupby_cols: Sequence[str] = (
        "Architecture",
        "Fold",
        "Random_state",
        "Mask",
        "Freq. selection",
        "Task",
    ),
    metric_cols: Sequence[str] = (
        "ROC AUC",
        "Sensitivity",
        "Specificity",
        "Recall",
        "F1-score",
    ),
    agg_funcs: Mapping[str, Union[str, List[str]]] = None,
    primary_metric: str = "ROC AUC",
    ascending: bool = False,
    round_digits: Optional[int] = 4,
) -> pd.DataFrame:
    """
    Summarise a results DataFrame by meaningful categorical combinations.

    Parameters
    ----------
    df : pd.DataFrame
        The raw results table. Must contain the columns listed in ``groupby_cols``
        and ``metric_cols``.
    groupby_cols : sequence of str, optional
        Columns that define a *combination* of experimental settings.
        The default list matches the columns you mentioned in the question.
    metric_cols : sequence of str, optional
        Numeric columns that hold the performance scores you want to aggregate.
    agg_funcs : mapping, optional
        Mapping ``metric_name -> aggregation``.  If ``None`` the function will
        automatically compute ``["mean", "std", "max"]`` for every metric.
        You can also pass a single string (e.g. ``"mean"``) or a list of strings.
    primary_metric : str, optional
        The metric used to sort the final table (default ``"ROC AUC"``).
    ascending : bool, optional
        Sort order for the primary metric. ``False`` (default) puts the best
        scores on top.
    round_digits : int or None, optional
        Number of decimal places to round the aggregated values. ``None`` leaves
        the original precision untouched.

    Returns
    -------
    pd.DataFrame
        One row per unique combination of ``groupby_cols`` with aggregated
        statistics for each metric.  Columns are named ``<metric>_<agg>`` (e.g.
        ``ROC AUC_mean``).

    Example
    -------
    >>> import pandas as pd
    >>> # suppose `results.csv` contains the columns you listed
    >>> df = pd.read_csv("results.csv")
    >>> summary = evaluate_combinations(df,
    ...                                 groupby_cols=["Architecture","Mask","Task"],
    ...                                 primary_metric="F1-score")
    >>> # show the top‑5 performing combos
    >>> print(summary.head())
    """
    missing = set(groupby_cols) - set(df.columns)
    if missing:
        raise KeyError(
            f"The following groupby columns are missing from the DataFrame: {missing}"
        )

    missing = set(metric_cols) - set(df.columns)
    if missing:
        raise KeyError(
            f"The following metric columns are missing from the DataFrame: {missing}"
        )

    if primary_metric not in metric_cols:
        raise ValueError(
            f"`primary_metric` must be one of {metric_cols}, got '{primary_metric}'."
        )

    # Build aggregation dictionary
    if agg_funcs is None:
        agg_funcs = {m: ["mean", "std", "max"] for m in metric_cols}
    else:
        # Normalise to a dict of lists
        norm = {}
        for m in metric_cols:
            if m not in agg_funcs:
                # default for missing metric
                norm[m] = ["mean", "std", "max"]
                continue
            val = agg_funcs[m]
            if isinstance(val, str):
                norm[m] = [val]
            else:
                norm[m] = list(val)
        agg_funcs = norm

    # Group‑by + aggregation
    grouped = df.groupby(list(groupby_cols)).agg(agg_funcs)

    # After aggregation pandas creates a MultiIndex column like
    # ('ROC AUC', 'mean').  Flatten it to a single level.
    grouped.columns = [f"{metric}_{agg}" for metric, agg in grouped.columns]

    # Reset index so the grouping columns become ordinary columns again
    result = grouped.reset_index()

    # Optional rounding
    if round_digits is not None:
        metric_agg_cols = [
            c for c in result.columns if any(m in c for m in metric_cols)
        ]
        result[metric_agg_cols] = result[metric_agg_cols].round(round_digits)

    # Sort by the primary metric (use the "mean" version if it exists,
    # otherwise the first aggregation that contains the metric name)
    sort_col_candidates = [
        f"{primary_metric}-mean",
        f"{primary_metric}-max",
        f"{primary_metric}-std",
    ]
    sort_col = next((c for c in sort_col_candidates if c in result.columns), None)
    if sort_col is None:
        # fallback: any column that starts with the metric name
        sort_col = next(c for c in result.columns if c.startswith(primary_metric))
    result = result.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

    return result


def load_top_Fsel_Msel(class_task):
    """
    return Freq_sel, Mask_sel
    """
    tmp = np.load(f"results/freq_fold_results_{class_task}.npz", allow_pickle=True)
    top_freqs_baseline = tmp["top_freqs_baseline"]

    Freq_sel = dict()
    for ft in range(1, len(top_freqs_baseline) + 1):
        Freq_sel[f"Top {ft}"] = top_freqs_baseline[:ft].tolist()

    # load masking selection
    tmp = np.load(f"results/Top_IIVV_masks_{class_task}.npz", allow_pickle=True)
    Mask_ranks = tmp["Mask_ranks"][:14]
    Mask_sel = {
        "All": "all",
        "Long a+": "vv_long_across",
        "Long a+ ext.": "vv_long_across_extended",
        "Med. a+ ext.": "vv_medium_across_extended",
        "Skip1 close": "vv_distance_close_skip_1",
        "Adj. close": "adjacent_close",
        "Med. adj.": "vv_distance_medium_adj",
        "Adj. far": "adjacent_far",
        "Skip1 medium": "skip1_medium",
        "Skip1 far": "skip1_far",
        "Opp. close": "opposite_close",
        "Opp. medium": "opposite_medium",
        "Opp. far": "opposite_far",
        "Z_threshold": None
    }
    Mask_sel = {k: v for k, v in Mask_sel.items() if k in Mask_ranks}
    return Freq_sel, Mask_sel