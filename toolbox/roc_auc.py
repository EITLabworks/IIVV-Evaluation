import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from .visualization import color_palette_full


# To maintain compatibility with the original function name
def compute_ROC(
    data, labels, patient_ids=None, test_size=0.3, plot=True, random_state=42
):
    print("Outdated, please use: compute_ROC_patient_split")
    """
    Computes ROC curves and AUC with either patient-based or random train/test splitting.

    Parameters:
        data        : Feature data array
        labels      : One-hot encoded class labels (samples, n_classes)
        patient_ids : Array of patient IDs (if None, will use random sample splitting)
        test_size   : Proportion for testing (default 0.3)
        plot        : Whether to plot ROC curves (default True)
        random_state: Random seed for reproducibility

    Returns:
        avg_roc_auc : Average AUC across all classes
        roc_auc_dict: Dictionary of AUC values for each class
    """
    # If patient_ids are provided, use patient-based splitting
    if patient_ids is not None:
        return compute_ROC_patient_split(
            data, labels, patient_ids, test_size, plot, random_state
        )

    # Original function implementation for sample-based splitting
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    data = np.nan_to_num(data)

    ohe_into = "labels should be in one hot encoding mode.\nUse: `from toolbox import hot_coding`"
    assert labels.shape[1] == np.unique(labels, axis=0).shape[1], ohe_into
    n_classes = labels.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    if plot:
        print("NaN are replaced by 0.")
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print("Note: Using random sample splitting (not patient-based)")

    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=random_state)
    )

    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    roc_auc_scores = []
    roc_auc_dict = dict()
    for i in range(n_classes):
        roc_auc_dict[f"roc_auc_{i}"] = roc_auc[i]
        score = roc_auc_score(y_test[:, i], y_score[:, i])
        roc_auc_scores.append(score)

    avg_roc_auc = np.mean(roc_auc_scores)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            "--",
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        )

        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        print(f"{avg_roc_auc=}")

    return avg_roc_auc, roc_auc_dict


def compute_ROC_patient_split(
    data,
    labels,
    patient_ids,
    test_size=0.3,
    plot=True,
    random_state=42,
    inverse_mapping=None,
):
    """
    Computes ROC curves and AUC with patient-based train/test splitting.

    Parameters:
        data            : Feature data array
        labels          : One-hot encoded class labels (samples, n_classes)
        patient_ids     : Array of patient IDs corresponding to each sample
        test_size       : Proportion of patients to use for testing (default 0.3)
        plot            : Whether to plot ROC curves (default True)
        random_state    : Random seed for reproducibility
        inverse_mapping : Dict mapping one-hot tuple -> class name (optional)

    Returns:
        avg_roc_auc : Average AUC across all classes
        roc_auc_dict: Dictionary of AUC values for each class label
    """
    # Reshape data if needed
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    # Replace NaN values with 0
    data = np.nan_to_num(data)

    # Verify one-hot encoding
    assert (
        labels.shape[1] == np.unique(labels, axis=0).shape[1]
    ), "labels should be in one hot encoding mode.\nUse: `from toolbox import hot_coding`"
    n_classes = labels.shape[1]

    # Get unique patient IDs
    unique_patients = np.unique(patient_ids)
    np.random.seed(random_state)

    # Split patients into train and test sets
    train_patient_indices, test_patient_indices = train_test_split(
        np.arange(len(unique_patients)), test_size=test_size, random_state=random_state
    )

    train_patients = unique_patients[train_patient_indices]
    test_patients = unique_patients[test_patient_indices]
    print("Train patients:", train_patients)
    print("Test patients:", test_patients)

    # Create masks for samples based on patient assignment
    train_mask = np.isin(patient_ids, train_patients)
    test_mask = np.isin(patient_ids, test_patients)

    # Split data using patient-based masks
    X_train = data[train_mask]
    X_test = data[test_mask]
    y_train = labels[train_mask]
    y_test = labels[test_mask]

    train_patient_count = len(train_patients)
    test_patient_count = len(test_patients)

    if plot:
        print("NaN values are replaced by 0.")
        print(
            f"Training: {X_train.shape[0]} samples from {train_patient_count} patients"
        )
        print(f"Testing:  {X_test.shape[0]} samples from {test_patient_count} patients")
        print(
            f"Data shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}"
        )

    # Train the classifier
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=random_state)
    )

    # Fit and predict
    try:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    except Exception as e:
        print(f"Error during classifier fitting: {e}")
        return 0.5, {}

    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception as e:
            print(f"Error computing ROC for class {i}: {e}")
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 0])
            roc_auc[i] = 0.5

    # Compute micro-average ROC curve and ROC area
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception as e:
        print(f"Error computing micro-average ROC: {e}")
        fpr["micro"] = np.array([0, 1])
        tpr["micro"] = np.array([0, 0])
        roc_auc["micro"] = 0.5

    # Calculate per-class AUC scores
    roc_auc_scores = []
    roc_auc_dict = {}
    for i in range(n_classes):
        try:
            score = roc_auc_score(y_test[:, i], y_score[:, i])
            # Key-Namen aus inverse_mapping statt class index
            if inverse_mapping:
                class_name = inverse_mapping.get(
                    tuple(labels[np.argmax(labels[:, i])]), f"Class_{i}"
                )
            else:
                class_name = f"Class_{i}"
            roc_auc_dict[f"roc_auc_{class_name}"] = score
            roc_auc_scores.append(score)
        except Exception as e:
            if inverse_mapping:
                class_name = inverse_mapping.get(
                    tuple(labels[np.argmax(labels[:, i])]), f"Class_{i}"
                )
            else:
                class_name = f"Class_{i}"
            roc_auc_dict[f"roc_auc_{class_name}"] = 0.5
            roc_auc_scores.append(0.5)

    # Calculate average AUC
    avg_roc_auc = np.mean(roc_auc_scores)

    # Plot ROC curves
    if plot:
        plt.figure(figsize=(6, 4))

        # Micro-average curve
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            "--",
            label=f"Micro-average ROC curve (area = {roc_auc['micro']:.3f})",
            linewidth=1.5,
            color="C0",  # oder eigene Farbe für Micro-average
        )

        # Per-class curves
        for i in range(n_classes):
            if inverse_mapping:
                class_name = inverse_mapping.get(
                    tuple(labels[np.argmax(labels[:, i])]), f"Class_{i}"
                )
            else:
                class_name = f"Class {i}"

            # take from color_palette_full
            color = color_palette_full.get(class_name, None)

            plt.plot(
                fpr[i],
                tpr[i],
                label=f"ROC {class_name} (area = {roc_auc[i]:.3f})",
                color=color,
            )

        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"Average ROC AUC = {avg_roc_auc:.4f}")
        print(
            f"Patient-based split: {train_patient_count} training patients, {test_patient_count} testing patients"
        )

    return avg_roc_auc, roc_auc_dict


def combined_frequency_roc_analysis(
    X,
    Y,
    LesionCase,
    max_freqs=31,
    n_folds=5,
    random_state=42,
    s_name=None,
):
    """
    Combines PCA-based frequency selection with ROC analysis using the same patient-based train/test splits.

    Parameters:
        X           : np.ndarray of shape (samples, freqs, patterns)
        Y           : Binary class labels in one-hot encoding (samples, n_classes)
        LesionCase  : Patient IDs for each sample
        max_freqs   : Max number of frequencies to test (default 31)
        n_folds     : Number of folds for cross-validation (default 5)
        random_state: Random seed for reproducibility
        s_name      : Save name of the pictures (default None)

    Returns:
        final_aucs  : Average AUCs across folds for each number of frequencies
        composite_freq_indices : Composite ordered frequency indices
        all_fold_rankings : List of frequency rankings from each fold
        fold_aucs   : Dictionary of AUC results for each fold
    """
    # Get unique patient IDs
    unique_patients = np.unique(LesionCase)
    np.random.seed(random_state)

    # Check if Y is in one-hot encoding
    if len(Y.shape) == 1 or Y.shape[1] == 1:
        raise ValueError("Labels (Y) must be in one-hot encoding format")

    n_classes = Y.shape[1]

    # Initialize arrays to store fold-specific rankings and overall frequency scores
    all_fold_rankings = []
    frequency_score_accumulator = np.zeros(X.shape[1])

    # Store AUC results for each fold
    fold_aucs = {i: [] for i in range(n_folds)}
    final_aucs = np.zeros(max_freqs)

    # Custom splitting to ensure 70/30 ratio
    patient_indices = np.arange(len(unique_patients))
    np.random.shuffle(patient_indices)

    # Split these indices into n_folds
    fold_size = len(patient_indices) // n_folds
    folds = []
    for i in range(n_folds):
        if i == n_folds - 1:
            # Last fold gets all remaining patients
            test_indices = patient_indices[i * fold_size :]
        else:
            test_indices = patient_indices[i * fold_size : (i + 1) * fold_size]

        # Training indices are all other indices
        train_indices = np.setdiff1d(patient_indices, test_indices)
        folds.append((train_indices, test_indices))

    print(
        f"Performing {n_folds}-fold cross-validation with PCA-based frequency ranking and ROC analysis..."
    )

    # Function to remove NaN rows for PCA
    def remove_nan_rows_for_pca(data, verbose=False):
        """Remove rows with NaN values for PCA"""
        nan_mask = np.isnan(data).any(axis=1)
        if verbose:
            print(f"Removed {nan_mask.sum()} rows with NaN values")
        return data[~nan_mask]

    # Loop through each fold
    for fold_idx, (train_patients_idx, test_patients_idx) in enumerate(folds):
        fold = fold_idx + 1

        # Get patient IDs for training and testing
        train_patients = unique_patients[train_patients_idx]
        test_patients = unique_patients[test_patients_idx]
        print("Train patients:", train_patients)
        print("Test patients:", test_patients)

        # Create masks for training and testing samples
        train_mask = np.isin(LesionCase, train_patients)
        test_mask = np.isin(LesionCase, test_patients)

        # Verify split is close to 70/30
        train_pct = train_mask.mean() * 100
        test_pct = test_mask.mean() * 100
        print(f"Fold {fold}: {train_pct:.1f}% train / {test_pct:.1f}% test")

        # Count samples and patients in each set
        train_samples = np.sum(train_mask)
        test_samples = np.sum(test_mask)
        train_patient_count = len(train_patients)
        test_patient_count = len(test_patients)

        print(f"Fold {fold}: {train_pct:.1f}% train / {test_pct:.1f}% test")
        print(
            f"  Training: {train_samples} samples from {train_patient_count} patients"
        )
        print(f"  Testing:  {test_samples} samples from {test_patient_count} patients")

        # Get training and testing data
        X_train = X[train_mask]
        X_test = X[test_mask]
        Y_train = Y[train_mask]
        Y_test = Y[test_mask]

        # Step 1: Reshape and clean training data for PCA
        X_train_reshaped = X_train.transpose(0, 2, 1).reshape(
            -1, X_train.shape[1]
        )  # (train_samples * patterns, freqs)
        X_train_clean = remove_nan_rows_for_pca(X_train_reshaped, verbose=False)

        # Step 2: Run PCA and rank frequencies using only training data
        pca = PCA(
            n_components=min(max_freqs, X_train_clean.shape[0], X_train_clean.shape[1])
        )
        pca.fit(X_train_clean)
        loading_matrix = pca.components_  # (components, freqs)
        frequency_scores = np.sum(loading_matrix[:10] ** 2, axis=0)
        fold_top_freq_indices = np.argsort(frequency_scores)[::-1]  # descending order

        # Store the frequency rankings for this fold
        all_fold_rankings.append(fold_top_freq_indices)

        # Accumulate scores for composite ranking
        # Add points based on rank (max_freqs - rank) so higher ranks get more points
        for rank, idx in enumerate(fold_top_freq_indices[:max_freqs]):
            frequency_score_accumulator[idx] += max_freqs - rank

        # Print the frequency ranking for this fold
        print(f"Fold {fold} frequency ranking (top 10): {fold_top_freq_indices[:10]}")

        # Step 3: ROC analysis for this fold using increasing numbers of top frequencies
        fold_auc_results = []

        for n in range(1, max_freqs + 1):
            selected_freqs = fold_top_freq_indices[:n]

            # Filter data to use only selected frequencies
            X_train_filtered = X_train[:, selected_freqs, :]
            X_test_filtered = X_test[:, selected_freqs, :]

            # Reshape for classifier
            X_train_flat = X_train_filtered.reshape(X_train_filtered.shape[0], -1)
            X_test_flat = X_test_filtered.reshape(X_test_filtered.shape[0], -1)

            # Replace NaN values with 0
            X_train_flat = np.nan_to_num(X_train_flat)
            X_test_flat = np.nan_to_num(X_test_flat)

            # Train classifier
            classifier = OneVsRestClassifier(
                svm.SVC(kernel="linear", probability=True, random_state=random_state)
            )

            # Fit and predict
            y_score = classifier.fit(X_train_flat, Y_train).predict_proba(X_test_flat)

            # Compute ROC AUC for each class
            roc_auc_scores = []
            for i in range(n_classes):
                try:
                    score = roc_auc_score(Y_test[:, i], y_score[:, i])
                    roc_auc_scores.append(score)
                except Exception as e:
                    print(
                        f"Warning: Could not calculate AUC for class {i} in fold {fold}: {e}"
                    )

            # Calculate average AUC across classes
            avg_auc = np.mean(roc_auc_scores) if roc_auc_scores else 0
            fold_auc_results.append(avg_auc)

            # Add to running total for final average
            final_aucs[n - 1] += avg_auc

            # print(f"Fold {fold}, Top {n} freq(s): AUC = {avg_auc:.4f}")

        # Store this fold's AUC results
        fold_aucs[fold_idx] = fold_auc_results

    # Calculate composite ranking based on accumulated scores
    composite_freq_indices = np.argsort(frequency_score_accumulator)[
        ::-1
    ]  # descending order

    # Calculate average AUC across folds
    final_aucs = final_aucs / n_folds

    print(
        "\nComposite frequency ranking from all folds (top 10):",
        composite_freq_indices[:10],
    )
    print("\nAverage AUC results across all folds:")
    for n in range(1, max_freqs + 1):
        print(f"Top {n} frequencies: Avg AUC = {final_aucs[n-1]:.4f}")

    # Create a heatmap visualization showing frequency ranks across folds
    plt.figure(figsize=(12, 8))

    # Select frequencies to display (top frequencies from composite ranking)
    top_freqs_to_show = min(30, X.shape[1])
    freqs_to_show = composite_freq_indices[:top_freqs_to_show]

    # Initialize the rank matrix: rows=frequencies, columns=folds
    rank_matrix = np.full((len(freqs_to_show), n_folds), np.nan)

    # Fill the matrix with rank positions
    for fold_idx, fold_rankings in enumerate(all_fold_rankings):
        # For each frequency in our display list, find its rank position in this fold
        for i, freq_idx in enumerate(freqs_to_show):
            # Find where this frequency appears in this fold's ranking
            rank_pos = np.where(fold_rankings == freq_idx)[0]
            if len(rank_pos) > 0:
                # Add 1 to make ranks 1-based instead of 0-based
                rank_matrix[i, fold_idx] = rank_pos[0] + 1
            # If not found, leave as nan

    # Create a masked array for better visualization
    masked_rank_matrix = np.ma.masked_invalid(rank_matrix)

    # Create the heatmap
    cmap = plt.cm.viridis_r  # Reversed colormap so lower ranks (better) are darker
    im = plt.imshow(masked_rank_matrix, aspect="auto", cmap=cmap)

    # Add a colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Rank Position (lower is better)")

    # Set up the axes
    plt.xlabel("Fold")
    plt.ylabel("Frequency Index")
    plt.yticks(range(len(freqs_to_show)), freqs_to_show)
    plt.xticks(range(n_folds), [f"Fold {i+1}" for i in range(n_folds)])

    # Add title
    plt.title("Frequency Rank Positions Across Folds")

    # Add text annotations with rank values
    for i in range(len(freqs_to_show)):
        for j in range(n_folds):
            if not np.ma.is_masked(masked_rank_matrix[i, j]):
                # Format as integer
                text = f"{int(masked_rank_matrix[i, j])}"
                plt.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=(
                        "white" if masked_rank_matrix[i, j] > max_freqs / 2 else "black"
                    ),
                )

    plt.tight_layout()
    if s_name != None:
        plt.savefig(s_name + "HeatMaps.pdf", bbox_inches="tight")
    plt.show()

    # Plot AUC vs number of frequencies
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, max_freqs + 1),
        final_aucs,
        marker="x",
        color="steelblue",
        linewidth=2,
        label="Average AUCs",
    )

    # Add individual fold lines
    for fold_idx in range(n_folds):
        plt.plot(
            range(1, max_freqs + 1),
            fold_aucs[fold_idx],
            "o-",
            alpha=0.5,
            label=f"Fold {fold_idx+1}",
        )

    plt.xlabel("Number of Frequencies Used")
    plt.ylabel("Average AUC")
    plt.title("AUC vs. Number of Frequencies Used")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if s_name != None:
        plt.savefig(s_name + "aucs.pdf", bbox_inches="tight")
    plt.show()

    return final_aucs, composite_freq_indices, all_fold_rankings, fold_aucs
