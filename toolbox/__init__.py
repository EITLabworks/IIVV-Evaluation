from .util import z_score, get_freqs, squeeze1shape
from .visualization import visualize_Category_hist
from .masking import euclidean_distance, Masking
from .processing import (
    assessment_wrapping,
    find_all_indices,
    augment_patient_and_cat_specific,
)

# please do not use compute_ROC anymore, because it is not bulletproof
from .roc_auc import combined_frequency_roc_analysis, compute_ROC_patient_split
from .decoding_encoding import Encoder, hot_coding
from .evaluation import evaluate_combinations

__all__ = [
    # util.py
    "get_freqs",
    "z_score",
    "squeeze1shape",
    # visualization.py
    "visualize_Category_hist",
    # masking.py
    "euclidean_distance",
    "Masking",
    # processing.py
    "find_all_indices",
    "augment_patient_and_cat_specific",
    # roc_auc.py
    "compute_ROC",
    "combined_frequency_roc_analysis",
    "compute_ROC_patient_split",
    # decoding_encoding.py
    "Encoder",
    "hot_coding",
    # evaluation.py
    "evaluate_combinations",
]
