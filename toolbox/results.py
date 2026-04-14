import numpy as np
from .loading import load_data
from .processing import find_all_indices, augment_patient_and_cat_specific
from .decoding_encoding import hot_coding
from .visualization import visualize_Category_hist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def Results_data_loading(class_task, data_augmentation=True):
    """
    return X, Y_labels, labels, Case, inverse_mapping, Y
    """
    (
        py_idx,
        patient_no,
        patient_sample,
        Zdata,
        Avg_Cutpats,
        LesionAssessment,
        labels_healthy_vs_cancer,
        labels_cancer_vs_lr_hr,
        labels_MC,
    ) = load_data(path="data", Cutpats_threshold=60, norm="freq")
    if class_task == "healthy_vs_cancer":
        print("Healthy vs. cancer\n")
        select_idx = find_all_indices(
            labels_healthy_vs_cancer, ["Healthy", "Cancer"], split=False
        )
        labels = labels_healthy_vs_cancer[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        # for SVD kernel boundary
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            1: "#003F5C",  # healthy
            0: "#FFA600",  # cancer
        }
        cp_l = {
            1: "healthy",  # healthy
            0: "cancer",  # cancer
        }
        visualize_Category_hist(labels)

    elif class_task == "cancer_vs_lr_hr":
        print("\nCancer vs. High risk vs. Low risk\n")
        select_idx = find_all_indices(
            labels_cancer_vs_lr_hr, ["Cancer", "High risk", "Low risk"], split=False
        )
        labels = labels_cancer_vs_lr_hr[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            0: "#FFA600",  # "Cancer
            1: "#FF6361",  # "High risk
            2: "#58508D",  # "Low risk
        }
        cp_l = {
            0: "Cancer",  # Cancer
            1: "High risk",  # High risk
            2: "Low risk",  # Low risk
        }
        visualize_Category_hist(labels)

    elif class_task == "MC":
        print("\nMC\n")
        select_idx = find_all_indices(
            labels_MC,
            ["Healthy", "Cancer", "High-grade dysplasia", "Non-malignant"],
            split=False,
        )
        labels = labels_MC[select_idx]
        Y, inverse_mapping = hot_coding(labels)
        Y_labels = np.argmax(Y, axis=1)
        cp = {
            3: "#58508D",  # Non-malignant
            0: "#FFA600",  # Cancer
            1: "#003F5C",  # Healthy
            2: "#FF6361",  # High-grade dysplasia
        }
        cp_l = {
            0: "Cancer",
            1: "Healthy",
            2: "High-grade dysplasia",
            3: "Non-malignant",
        }

        aug_dict = {
            "High-grade dysplasia": 70,
            "Non-malignant": 70,
        }
        visualize_Category_hist(labels)
    Case = patient_no[select_idx]
    X = Zdata[select_idx, ...]
    print("\nInverse mapping:", inverse_mapping)
    print(X.shape, Y.shape, Case.shape)

    if class_task == "MC" and data_augmentation:
        print("\n\t\tBefore augmentation")
        visualize_Category_hist(labels)
        X, Y_labels, labels, Case = augment_patient_and_cat_specific(
            X, Y_labels, labels, Case, aug_dict
        )
        print("\n\t\tAfter augmentation")
        visualize_Category_hist(labels)
    return X, Y_labels, labels, Case, inverse_mapping, Y


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def ResultsROCAUC(
    class_task,
    y_true,
    y_score,
    Y_labels=None,
    s_name=None,
    *,
    score_classes,
    eval_classes=None,
    pos_label=None,
    plot_micro=True,  # default: micro-average for multiclass
    npz_path=None,  # optional: save everything needed to recreate plot
    npz_extra_meta=None,  # optional dict with extra metadata to store
):
    """
    Plots ROC/AUC and (optionally) saves a .npz with everything needed to recreate
    the exact plot later.

    - Binary: plots ROC for pos_label (default: largest label in eval_classes).
    - Multiclass (OvR): plots per-class ROC + ONE average curve:
        * micro-average if plot_micro=True (default)
        * macro-average if plot_micro=False

    If npz_path is provided, saves:
      - metadata (class_task, eval_classes, score_classes, pos_label, plot_micro, labels)
      - per-class curves: fpr/tpr/thresholds + auc
      - average curve (micro or macro) if multiclass
    """

    # --- label/color maps (keep your mappings) ---
    label_roc_plot = {
        "healthy_vs_cancer": "Healthy vs. cancer",
        "cancer_vs_lr_hr": "Cancer vs. high risk vs. low risk",
        "MC": "Multiclass",
    }

    if class_task == "healthy_vs_cancer":
        cp = {1: "#003F5C", 0: "#FFA600"}
        cp_l = {1: "healthy", 0: "cancer"}
    elif class_task == "cancer_vs_lr_hr":
        cp = {0: "#FFA600", 1: "#FF6361", 2: "#003F5C"}
        cp_l = {0: "Cancer", 1: "High-grade dysplasia", 2: "Non-malignant"}
    elif class_task == "MC":
        cp = {3: "#58508D", 0: "#FFA600", 1: "#003F5C", 2: "#FF6361"}
        cp_l = {
            0: "Cancer",
            1: "Healthy",
            2: "High-grade dysplasia",
            3: "Non-malignant",
        }
    else:
        raise ValueError(f"Unknown class_task={class_task}")

    y_true = np.asarray(y_true)

    # Decide eval_classes (stable target order)
    if eval_classes is None:
        if Y_labels is not None:
            eval_classes = np.array(sorted(np.unique(Y_labels)))
        else:
            eval_classes = np.array(sorted(np.unique(y_true)))
    else:
        eval_classes = np.asarray(eval_classes)

    # y_score shape normalize
    y_score = np.asarray(y_score)
    if y_score.ndim == 1 and len(eval_classes) != 2:
        raise ValueError("y_score is 1D but eval_classes is not binary.")

    score_classes = np.asarray(score_classes)
    if score_classes.ndim != 1:
        raise ValueError("score_classes must be a 1D array-like of labels.")

    # Validate that y_true labels are known
    unknown_y = set(np.unique(y_true)) - set(eval_classes.tolist())
    if unknown_y:
        raise ValueError(
            f"y_true contains labels not in eval_classes: {sorted(unknown_y)}"
        )

    # Validate eval_classes exist in score_classes
    missing = [c for c in eval_classes if c not in set(score_classes.tolist())]
    if missing:
        raise ValueError(
            f"These classes are in eval_classes but missing in score_classes/y_score columns: {missing}. "
            "Pass score_classes=clf.classes_ (or the correct column order)."
        )

    # Build mapping from class -> column index in y_score
    score_col = {lbl: j for j, lbl in enumerate(score_classes)}

    # Align y_score columns to eval_classes order
    if y_score.ndim == 1:
        # Binary only: treat provided scores as scores for pos_label
        if pos_label is None:
            pos_label = eval_classes[-1]
        if pos_label not in eval_classes:
            raise ValueError(
                f"pos_label={pos_label} not in eval_classes={eval_classes.tolist()}"
            )

        other = [c for c in eval_classes if c != pos_label][0]
        s_pos = y_score.astype(float)
        s_other = 1.0 - s_pos
        y_score_aligned = np.column_stack(
            [s_pos if c == pos_label else s_other for c in eval_classes]
        )
    else:
        if y_score.shape[1] != len(score_classes):
            raise ValueError(
                f"y_score has {y_score.shape[1]} columns, but score_classes has {len(score_classes)} labels."
            )
        y_score_aligned = np.column_stack(
            [y_score[:, score_col[c]] for c in eval_classes]
        )

    # ---- Compute curves + build object to save/return ----
    roc_bundle = {
        "meta": {
            "class_task": class_task,
            "plot_micro": bool(plot_micro),
            "pos_label": None if pos_label is None else pos_label,
            "eval_classes": eval_classes.copy(),
            "score_classes": score_classes.copy(),
            "label_roc_plot": label_roc_plot.get(class_task, "ROC"),
            "cp_l": cp_l,
        },
        "per_class": {},  # class -> dict(fpr,tpr,thresholds,auc)
        "average": None,  # dict(type,fpr,tpr,auc) or None
    }

    # ----- Plot -----
    plt.figure(figsize=(5.5, 3))

    # ----- Binary -----
    if len(eval_classes) == 2:
        if pos_label is None:
            pos_label = eval_classes[-1]
        if pos_label not in eval_classes:
            raise ValueError(
                f"pos_label={pos_label} not in eval_classes={eval_classes.tolist()}"
            )

        pos_idx = int(np.where(eval_classes == pos_label)[0][0])
        y_true_bin = (y_true == pos_label).astype(int)

        fpr, tpr, thr = roc_curve(y_true_bin, y_score_aligned[:, pos_idx])
        roc_auc = auc(fpr, tpr)

        roc_bundle["meta"]["pos_label"] = pos_label
        roc_bundle["per_class"][pos_label] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thr,
            "auc": float(roc_auc),
        }

        color = cp.get(pos_label, "#003F5C")
        label_txt = f"{label_roc_plot.get(class_task, 'ROC')} (AUC = {roc_auc:.2f})"
        plt.plot(fpr, tpr, color=color, label=label_txt)
        print("AUC =", roc_auc)

    # ----- Multiclass OvR -----
    else:
        y_bin = label_binarize(y_true, classes=eval_classes)

        # per-class OvR
        for k, c in enumerate(eval_classes):
            fpr, tpr, thr = roc_curve(y_bin[:, k], y_score_aligned[:, k])
            roc_auc = auc(fpr, tpr)

            roc_bundle["per_class"][c] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thr,
                "auc": float(roc_auc),
            }

            plt.plot(
                fpr,
                tpr,
                color=cp.get(c, None),
                label=f"{cp_l.get(c, str(c))} (AUC = {roc_auc:.2f})",
            )

        # average curve (micro default OR macro if plot_micro=False)
        if plot_micro:
            fpr_avg, tpr_avg, thr_avg = roc_curve(
                y_bin.ravel(), y_score_aligned.ravel()
            )
            avg_auc = auc(fpr_avg, tpr_avg)
            avg_type = "micro"
            plt.plot(
                fpr_avg,
                tpr_avg,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Micro-average (AUC = {avg_auc:.2f})",
            )
            print("Micro-average AUC =", avg_auc)
        else:
            mean_fpr = np.linspace(0.0, 1.0, 200)
            tprs = []
            for k in range(len(eval_classes)):
                fpr_k, tpr_k, _ = roc_curve(y_bin[:, k], y_score_aligned[:, k])
                tpr_interp = np.interp(mean_fpr, fpr_k, tpr_k)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            fpr_avg, tpr_avg, thr_avg = mean_fpr, mean_tpr, None
            avg_auc = auc(fpr_avg, tpr_avg)
            avg_type = "macro"
            plt.plot(
                fpr_avg,
                tpr_avg,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Macro-average (AUC = {avg_auc:.2f})",
            )
            print("Macro-average AUC =", avg_auc)

        roc_bundle["average"] = {
            "type": avg_type,
            "fpr": fpr_avg,
            "tpr": tpr_avg,
            "thresholds": thr_avg,  # None for macro
            "auc": float(avg_auc),
        }

    # Formatting
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([-0.03, 1.03])
    plt.ylim([0.0, 1.03])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if s_name is not None:
        plt.savefig(s_name, bbox_inches="tight")

    # ---- Save NPZ (optional) ----
    if npz_path is not None:
        _save_roc_npz(npz_path, roc_bundle, extra_meta=npz_extra_meta)

    plt.show()
    return roc_bundle


def _save_roc_npz(npz_path, roc_bundle, extra_meta=None):
    """
    Internal helper: stores roc_bundle into a compressed NPZ.

    Uses one key per array, plus a small meta dict stored as an object array.
    """
    extra_meta = {} if extra_meta is None else dict(extra_meta)

    meta = dict(roc_bundle.get("meta", {}))
    meta["average_type"] = (
        None if roc_bundle.get("average") is None else roc_bundle["average"].get("type")
    )
    meta.update(extra_meta)

    arrays = {}

    # Store eval_classes/score_classes explicitly as arrays
    arrays["eval_classes"] = np.asarray(meta.get("eval_classes"))
    arrays["score_classes"] = np.asarray(meta.get("score_classes"))

    # Store meta as pickled object (safe if you control the file source)
    # If you want "no pickle", remove this and store only primitives/arrays.
    arrays["meta"] = np.array(meta, dtype=object)

    # Per-class curves
    for cls, d in roc_bundle.get("per_class", {}).items():
        cls_key = str(cls)
        arrays[f"per_class/{cls_key}/fpr"] = np.asarray(d["fpr"], dtype=float)
        arrays[f"per_class/{cls_key}/tpr"] = np.asarray(d["tpr"], dtype=float)
        arrays[f"per_class/{cls_key}/auc"] = np.asarray(d["auc"], dtype=float)
        thr = d.get("thresholds", None)
        if thr is not None:
            arrays[f"per_class/{cls_key}/thresholds"] = np.asarray(thr, dtype=float)

    # Average curve (if exists)
    avg = roc_bundle.get("average", None)
    if avg is not None:
        arrays["average/type"] = np.asarray(str(avg["type"]), dtype=object)
        arrays["average/fpr"] = np.asarray(avg["fpr"], dtype=float)
        arrays["average/tpr"] = np.asarray(avg["tpr"], dtype=float)
        arrays["average/auc"] = np.asarray(avg["auc"], dtype=float)
        if avg.get("thresholds", None) is not None:
            arrays["average/thresholds"] = np.asarray(avg["thresholds"], dtype=float)

    # os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(npz_path, **arrays)


def PlotROCAUCFromNPZ(
    *path_parts,
    show=True,
    save_path=None,
):
    """
    Load ROC bundle from an NPZ created by ResultsROCAUC(..., npz_path=...),
    recreate the plot, and return a dict with:
      - 'meta'
      - 'per_class': {class: {'fpr','tpr','thresholds','auc'}}
      - 'average': {'type','fpr','tpr','thresholds','auc'} or None

    Usage:
        bundle = PlotROCAUCFromNPZ("results", "fold0", "roc_plot.npz")
        # or:
        bundle = PlotROCAUCFromNPZ("/abs/path/to/roc_plot.npz")
    """
    npz_path = os.path.join(*path_parts) if len(path_parts) > 1 else path_parts[0]

    with np.load(npz_path, allow_pickle=True) as z:
        meta = z["meta"].item() if "meta" in z else {}
        eval_classes = z["eval_classes"] if "eval_classes" in z else np.array([])
        score_classes = z["score_classes"] if "score_classes" in z else np.array([])

        class_task = meta.get("class_task", "ROC")
        plot_title = meta.get("label_roc_plot", "ROC")
        cp_l = meta.get("cp_l", {})  # may be missing; that's fine

        # Rebuild dicts
        per_class = {}
        for key in z.files:
            if not key.startswith("per_class/"):
                continue
            _, cls_str, field = key.split("/", 2)
            per_class.setdefault(cls_str, {})
            per_class[cls_str][field] = z[key]

        # Convert AUC scalar arrays to floats
        for cls_str, d in per_class.items():
            if "auc" in d:
                d["auc"] = float(np.asarray(d["auc"]).reshape(()))

        average = None
        if "average/fpr" in z and "average/tpr" in z and "average/auc" in z:
            average = {
                "type": (
                    str(z["average/type"].item())
                    if "average/type" in z
                    else meta.get("average_type", "micro")
                ),
                "fpr": z["average/fpr"],
                "tpr": z["average/tpr"],
                "auc": float(np.asarray(z["average/auc"]).reshape(())),
                "thresholds": (
                    z["average/thresholds"] if "average/thresholds" in z else None
                ),
            }

    # ---- Plot ----
    plt.figure(figsize=(5.5, 3))

    # Try to recover colors for known tasks (optional). If unknown, Matplotlib default colors are used.
    def _colors_for_task(task):
        if task == "healthy_vs_cancer":
            return {1: "#003F5C", 0: "#FFA600"}
        if task == "cancer_vs_lr_hr":
            return {0: "#FFA600", 1: "#FF6361", 2: "#003F5C"}
        if task == "MC":
            return {3: "#58508D", 0: "#FFA600", 1: "#003F5C", 2: "#FF6361"}
        return {}

    cp = _colors_for_task(class_task)

    # Plot per-class
    # Note: classes were saved as string keys; we attempt to cast to int if possible for color/label lookup.
    for cls_str, d in per_class.items():
        try:
            cls_key = int(cls_str)
        except Exception:
            cls_key = cls_str

        fpr = d.get("fpr")
        tpr = d.get("tpr")
        auc_val = d.get("auc", None)

        if fpr is None or tpr is None:
            continue

        label_name = cp_l.get(cls_key, cls_str)
        if auc_val is not None:
            label = f"{label_name} (AUC = {auc_val:.2f})"
        else:
            label = f"{label_name}"

        plt.plot(fpr, tpr, color=cp.get(cls_key, None), label=label)

    # Plot average (if present)
    if average is not None:
        avg_type = average.get("type", "micro")
        plt.plot(
            average["fpr"],
            average["tpr"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"{avg_type.capitalize()}-average (AUC = {average['auc']:.2f})",
        )

    # Formatting
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([-0.03, 1.03])
    plt.ylim([0.0, 1.03])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    # Return a clean bundle (use original eval/score classes from file)
    out = {
        "meta": {
            **(meta or {}),
            "eval_classes": eval_classes,
            "score_classes": score_classes,
            "npz_path": npz_path,
        },
        "per_class": per_class,  # keys are strings (saved form)
        "average": average,
    }
    return out
