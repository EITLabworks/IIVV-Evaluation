import warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def Model(
    architecture: str,
    X_train,
    Y_train,
    X_test,
    Y_test,
    random_state,
    inverse_mapping,
    fold,
    Msel,
    Fsel,
    class_task,
):
    """
    Train and evaluate a binary-class, multi‑class (or multi‑label) classifier.

    The function builds a scikit‑learn estimator according to the requested
    ``architecture`` (e.g. SVM, Random Forest, Gradient Boosting, Logistic Regression,
    Decision Tree).  For most models a ``OneVsRestClassifier`` wrapper is used so that
    the routine works with the one‑hot encoded label matrices that are typical for
    the inverse EIS task.  Optional ``StandardScaler`` preprocessing can be added
    (e.g. ``SVMscaler`` or ``LRscaler``) to perform z‑score normalisation before
    fitting.

    After fitting, the model predicts class probabilities on ``X_test`` and
    computes three performance metrics:

    * **ROC‑AUC** (area under the ROC curve, computed with ``roc_auc_score`` on the
      one‑hot encoded ground‑truth and predicted probabilities)
    * **Recall** (weighted‑average recall across all classes)
    * **F1‑score** (weighted‑average F1 across all classes)

    For the special case where the label matrix has exactly two columns
    (binary classification), the function also builds a confusion matrix using
    the provided ``inverse_mapping`` to translate one‑hot tuples back to the
    original integer labels.

    Parameters
    ----------
    architecture : str
        Identifier of the model to train.  Supported values are:

        * ``"SVMscaler"`` - SVM with RBF kernel + ``StandardScaler``.
        * ``"SVM"`` - SVM with RBF kernel (no scaling).
        * ``"DecisionTreeClassifier"`` - Decision‑Tree wrapped in ``OneVsRest``.
        * ``"GradientBoostingClassifier"`` - Gradient Boosting wrapped in ``OneVsRest``.
        * ``"RF"`` - Random Forest (max depth = 5) wrapped in ``OneVsRest``.
        * ``"RFscaler"`` - Same as ``"RF"`` but with ``StandardScaler``.
        * ``"LR"`` - Logistic Regression (max_iter = 1000) wrapped in ``OneVsRest``.
        * ``"LRscaler"`` - Same as ``"LR"`` but with ``StandardScaler``.

    X_train : array‑like of shape (n_samples_train, n_features)
        Training feature matrix.

    Y_train : array‑like of shape (n_samples_train, n_classes)
        One‑hot encoded training labels.

    X_test : array‑like of shape (n_samples_test, n_features)
        Test feature matrix.

    Y_test : array‑like of shape (n_samples_test, n_classes)
        One‑hot encoded test labels.

    random_state : int or ``np.random.RandomState``, optional
        Seed for reproducibility of stochastic estimators (e.g. Random Forest,
        Gradient Boosting, SVM probability estimates).

    inverse_mapping : dict
        Mapping from one‑hot tuple representation to the original integer label.
        Required only when ``Y_train.shape[1] == 2`` (binary case) to build a
        human‑readable confusion matrix.

    fold : int or str
        Identifier of the cross‑validation fold (useful when the function is
        called inside a loop over folds).

    Msel : any
        Mask selection identifier - passed through to the output dictionary
        unchanged.  The exact type depends on the surrounding pipeline.

    Fsel : any
        Frequency‑selection identifier - also passed through to the output
        dictionary unchanged.

    class_task : str
        Description of the classification task (e.g. ``"binary"``, ``"multiclass"``,
        ``"multilabel"``).  Stored in the result for bookkeeping.

    Returns
    -------
    performance : dict
        Dictionary summarising the run.  Keys include:

        * ``"Architecture"`` - the model identifier used.
        * ``"Fold"`` - the fold identifier supplied.
        * ``"Random_state"`` - the random seed used.
        * ``"ROC AUC"`` - ROC‑AUC score (float).
        * ``"Recall"`` - weighted recall (float).
        * ``"F1-score"`` - weighted F1 score (float).
        * ``"Mask"`` - the ``Msel`` argument (passed through).
        * ``"Freq. selection"`` - the ``Fsel`` argument (passed through).
        * ``"Task"`` - the ``class_task`` argument (passed through).

        The dictionary can be directly appended to a pandas ``DataFrame`` for
        later analysis.

    Raises
    ------
    ValueError
        If ``architecture`` is not one of the supported identifiers.

    Notes
    -----
    * The function suppresses all warnings globally via ``warnings.filterwarnings("ignore")``.
      Consider removing that line if you need to debug convergence or deprecation warnings.
    * ``roc_auc_score`` expects either binary or multilabel‑indicator format; the
      implementation works for both because ``Y_test`` is kept in its original
      one‑hot form.
    * For binary classification the confusion matrix is computed using the
      original integer labels obtained via ``inverse_mapping``.  The matrix is
      not returned but the intermediate variables ``TP, TN, FP, FN`` are
      available for debugging inside the function.

    Examples
    --------
    >>> perf = Model(
    ...     architecture="SVMscaler",
    ...     X_train=X_tr,
    ...     Y_train=Y_tr,
    ...     X_test=X_te,
    ...     Y_test=Y_te,
    ...     random_state=42,
    ...     inverse_mapping={(0, 1): 20, (1, 0): 400},
    ...     fold=1,
    ...     Msel="mask_01",
    ...     Fsel="freq_05",
    ...     class_task="binary",
    ... )
    >>> print(perf["ROC AUC"])
    0.93
    """

    if architecture == "SVMscaler":
        print("\nNew Supported vector machine (SVM) with StandardScaler\n")
        m = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                SVC(kernel="rbf", probability=True, random_state=random_state)
            ),
        )
    elif architecture == "SVM":
        print("\nSupported vector machine (SVM)\n")
        m = OneVsRestClassifier(
            SVC(kernel="rbf", probability=True, random_state=random_state)
        )

    elif architecture == "DecisionTreeClassifier":
        print("\nDecisionTreeClassifier\n")
        m = OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state))

    elif architecture == "GradientBoostingClassifier":
        print("\nGradientBoostingClassifier (GBC)\n")
        m = OneVsRestClassifier(GradientBoostingClassifier(random_state=random_state))

    elif architecture == "RF":
        print("\nRandom Forest (RF) - RandomForestClassifier\n")
        m = make_pipeline(
            OneVsRestClassifier(
                RandomForestClassifier(max_depth=5, random_state=random_state)
            ),
        )
    elif architecture == "RFscaler":
        print("\nRandom Forest (RF) with StandardScaler - RandomForestClassifier\n")
        m = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                RandomForestClassifier(max_depth=5, random_state=random_state)
            ),
        )

    elif architecture == "LR":
        print("\nLogistic Regression (LR)\n")
        m = make_pipeline(
            OneVsRestClassifier(
                LogisticRegression(max_iter=1000, random_state=random_state)
            ),
        )
    elif architecture == "LRscaler":
        print("\nLogistic Regression (LR) - with StandardScaler\n")
        m = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(max_iter=1000, random_state=random_state)
            ),
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    y_score = m.fit(X_train, Y_train).predict_proba(X_test)
    Y_pred = m.predict(X_test)

    if Y_train.shape[1] == 2:
        y_test_tuples = [tuple(row) for row in Y_test.astype(int)]
        y_pred_tuples = [tuple(row) for row in Y_pred.astype(int)]
        # inverse mapping
        y_test_labels = [inverse_mapping[t] for t in y_test_tuples]
        y_pred_labels = [inverse_mapping[t] for t in y_pred_tuples]
        class_labels = list(inverse_mapping.values())
        # confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_labels)

        TP = cm[0, 0]  # true positives (TP)
        TN = cm[1, 1]  # true negatives (TN)
        FN = cm[0, 1]  # false negatives (FN)
        FP = cm[1, 0]  # false positives (FP)

        class_labels = list(inverse_mapping.values())

    score = roc_auc_score(Y_test, y_score)
    recall = recall_score(Y_test, Y_pred, labels=None, average="weighted")
    f1 = f1_score(Y_test, Y_pred, average="weighted")

    print("ROC AUC :", score)
    print("Recall  :", recall)
    print("F1 score:", f1)

    performance = {
        "Architecture": architecture,
        "Fold": fold,
        "Random_state": random_state,
        "ROC AUC": score,
        "Recall": recall,
        "F1-score": f1,
        "Mask": Msel,
        "Freq. selection": Fsel,
        "Task": class_task,
    }
    return performance
