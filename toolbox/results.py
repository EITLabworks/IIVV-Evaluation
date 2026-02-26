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
    plot_macro=True,
):
    """
    Robust ROC/AUC plotting that NEVER silently swaps labels.

    Parameters
    ----------
    y_true : array-like (N,)
        True labels (any hashable / ints).
    y_score : array-like (N, K) or (N,)
        Probabilities/scores per class in the order given by score_classes.
    score_classes : array-like (K,)
        The class labels corresponding to columns in y_score.
        This MUST be provided (e.g. clf.classes_).
    eval_classes : array-like (K,), optional
        The stable class order to evaluate/plot. If None, uses sorted unique labels from Y_labels if given,
        else from y_true.
    pos_label : label, optional
        For binary ROC: explicitly define the positive class.
        If None: uses the *largest* label in eval_classes.
    plot_macro : bool
        For multiclass: whether to plot macro-average ROC.

    Raises
    ------
    ValueError if classes are missing/misaligned.
    """

    # --- label/color maps (keep your existing mappings if you want) ---
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
        # cp_l = {0: "Cancer", 1: "High risk", 2: "Low risk"}
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
    if y_score.ndim == 1:
        # Only allowed for binary: score for ONE class.
        if len(eval_classes) != 2:
            raise ValueError("y_score is 1D but eval_classes is not binary.")
        # Interpret as score for pos_label (must be provided or default below)
        # We'll expand to (N,2) after pos_label is known.
        pass

    score_classes = np.asarray(score_classes)
    if score_classes.ndim != 1:
        raise ValueError("score_classes must be a 1D array-like of labels.")

    # Validate that y_true labels are known
    unknown_y = set(np.unique(y_true)) - set(eval_classes.tolist())
    if unknown_y:
        raise ValueError(
            f"y_true contains labels not in eval_classes: {sorted(unknown_y)}"
        )

    # Validate that eval_classes exist in score_classes
    missing = [c for c in eval_classes if c not in set(score_classes.tolist())]
    if missing:
        raise ValueError(
            f"These classes are in eval_classes but missing in score_classes/y_score columns: {missing}. "
            "Pass score_classes=clf.classes_ (or the correct column order)."
        )

    # Build mapping from class -> column index in y_score
    score_col = {lbl: j for j, lbl in enumerate(score_classes)}

    # Align y_score columns to eval_classes order (this is the key!)
    if y_score.ndim == 1:
        # Binary only: treat provided scores as scores for pos_label
        if pos_label is None:
            pos_label = eval_classes[-1]
        if pos_label not in eval_classes:
            raise ValueError(
                f"pos_label={pos_label} not in eval_classes={eval_classes.tolist()}"
            )
        # Build (N,2) aligned: column0 = other class, column1 = pos_label (in eval order)
        other = [c for c in eval_classes if c != pos_label][0]
        s_pos = y_score.astype(float)
        s_other = 1.0 - s_pos
        # Now align to eval_classes order:
        aligned = np.column_stack(
            [s_pos if c == pos_label else s_other for c in eval_classes]
        )
        y_score_aligned = aligned
    else:
        if y_score.shape[1] != len(score_classes):
            raise ValueError(
                f"y_score has {y_score.shape[1]} columns, but score_classes has {len(score_classes)} labels."
            )
        y_score_aligned = np.column_stack(
            [y_score[:, score_col[c]] for c in eval_classes]
        )

    plt.figure(figsize=(5.5, 3))

    # ----- Binary case -----
    if len(eval_classes) == 2:
        if pos_label is None:
            pos_label = eval_classes[-1]  # deterministic default: "largest" label
        if pos_label not in eval_classes:
            raise ValueError(
                f"pos_label={pos_label} not in eval_classes={eval_classes.tolist()}"
            )

        pos_idx = int(np.where(eval_classes == pos_label)[0][0])
        y_true_bin = (y_true == pos_label).astype(int)

        fpr, tpr, _ = roc_curve(y_true_bin, y_score_aligned[:, pos_idx])
        roc_auc = auc(fpr, tpr)

        color = cp.get(pos_label, "#003F5C")
        label_txt = f"{label_roc_plot.get(class_task, 'ROC')} (AUC = {roc_auc:.2f})"
        plt.plot(fpr, tpr, color=color, label=label_txt)
        print("AUC =", roc_auc)

    # ----- Multiclass OvR -----
    else:
        y_bin = label_binarize(y_true, classes=eval_classes)

        # per-class OvR
        for k, c in enumerate(eval_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, k], y_score_aligned[:, k])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=cp.get(c, None),
                label=f"{cp_l.get(c, str(c))} (AUC = {roc_auc:.2f})",
            )

        # macro-average (optional)
        if plot_macro:
            # gemeinsames FPR-Gitter
            mean_fpr = np.linspace(0.0, 1.0, 200)

            tprs = []
            aucs = []

            for k in range(len(eval_classes)):
                fpr_k, tpr_k, _ = roc_curve(y_bin[:, k], y_score_aligned[:, k])
                auc_k = auc(fpr_k, tpr_k)
                aucs.append(auc_k)

                # TPR auf gemeinsames Gitter interpolieren
                tpr_interp = np.interp(mean_fpr, fpr_k, tpr_k)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

            # Mittelwert der interpolierten TPRs
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0

            # AUC der gemittelten Kurve
            macro_auc = auc(mean_fpr, mean_tpr)

            plt.plot(
                mean_fpr,
                mean_tpr,
                color="black",
                linestyle="--",
                label=f"Micro-average (AUC = {macro_auc:.2f})",
            )

            print("Micro-average AUC =", macro_auc)

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
    plt.show()
