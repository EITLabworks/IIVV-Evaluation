#!/usr/bin/env python3
import os
from itertools import product

RESULT_DIR = "results"

class_tasks = ["healthy_vs_cancer", "cancer_vs_lr_hr", "MC"]

metrics = [
    "f1",
    "auc",
    "acc",
]

models = [
    "svm",
    "rf",
    "lr",
]

# build header
header = ["Task", "Model"] + metrics
col_widths = [18, 8] + [10] * len(metrics)


def fmt_row(row):
    return " ".join(str(val).ljust(w) for val, w in zip(row, col_widths))


def main():
    # ----------------------------------------------------------------------- #
    print("\n1.1 z-threshold\n")

    def build_filename():
        return f"results/z_thresholds_best.json"

    # print a checkmark if the file exists
    path = build_filename()
    if os.path.exists(path):
        print("✅ z_thresholds_best.json file found.")
    else:
        print("❌ z_thresholds_best.json file not found.")

    def build_filename(task, metric):
        return f"zthres_tuning_best_thresholds_ALL_{metric}.csv"

    status = {}

    header_nomodel = ["Task"] + metrics
    col_widths_nomodel = [18, 5] + [8] * len(metrics)
    for task, metric in product(class_tasks, metrics):
        filename = build_filename(task, metric)
        path = os.path.join(RESULT_DIR, filename)
        status[(task, metric)] = os.path.exists(path)

    print("\n1.1 z-threshold\n")
    print(fmt_row(header_nomodel))
    print("-" * sum(col_widths_nomodel))

    for task in class_tasks:
        row = [task]
        for metric in metrics:
            ok = status[(task, metric)]
            row.append("✅" if ok else "❌")

        print(fmt_row(row))

    # ----------------------------------------------------------------------- #

    # 1.2
    def build_filename(task, model, metric):
        return f"MaskSel_ranking_{task}_{metric}.csv"

    status = {}

    for task, model, metric in product(class_tasks, models, metrics):
        filename = build_filename(task, model, metric)
        path = os.path.join(RESULT_DIR, filename)
        status[(task, model, metric)] = os.path.exists(path)

    print("\n1.2 Mask-analysis\n")
    print(fmt_row(header))
    print("-" * sum(col_widths))

    for task in class_tasks:
        for model in models:
            row = [task, model]

            for metric in metrics:
                ok = status[(task, model, metric)]
                row.append("✅" if ok else "❌")

            print(fmt_row(row))

    # 1.3
    def build_filename(task, model, metric):
        return f"FREQ-analysis_{task}_{metric}_gpu.csv"

    status = {}

    for task, model, metric in product(class_tasks, models, metrics):
        filename = build_filename(task, model, metric)
        path = os.path.join(RESULT_DIR, filename)
        status[(task, model, metric)] = os.path.exists(path)

    print("\n1.3 Freq-analysis\n")
    print(fmt_row(header))
    print("-" * sum(col_widths))

    for task in class_tasks:
        for model in models:
            row = [task, model]

            for metric in metrics:
                ok = status[(task, model, metric)]
                row.append("✅" if ok else "❌")

            print(fmt_row(row))

    # 2
    def build_filename(task, model, metric):
        return f"GPU_tuning_{model}_{task}_{metric}.csv"

    status = {}

    for task, model, metric in product(class_tasks, models, metrics):
        filename = build_filename(task, model, metric)
        path = os.path.join(RESULT_DIR, filename)
        status[(task, model, metric)] = os.path.exists(path)

    print("\n2. GPU tuning\n")
    print(fmt_row(header))
    print("-" * sum(col_widths))

    for task in class_tasks:
        for model in models:
            row = [task, model]

            for metric in metrics:
                ok = status[(task, model, metric)]
                row.append("✅" if ok else "❌")

            print(fmt_row(row))

    # 3.
    def build_filename(task, model, metric):
        return f"plots_{task}_{metric}/ROC_{task}_{model}_{metric}.pdf"

    status = {}

    for task, model, metric in product(class_tasks, models, metrics):
        filename = build_filename(task, model, metric)
        path = os.path.join(RESULT_DIR, filename)
        status[(task, model, metric)] = os.path.exists(path)

    print("\n3. Final results\n")
    print(fmt_row(header))
    print("-" * sum(col_widths))

    for task in class_tasks:
        for model in models:
            row = [task, model]

            for metric in metrics:
                ok = status[(task, model, metric)]
                row.append("✅" if ok else "❌")

            print(fmt_row(row))

    # 4.
    def build_filename(task, model, metric):
        return f"plots_{task}_baseline/ROC_baseline_{task}_{model}.pdf"

    status = {}

    for task, model, metric in product(class_tasks, models, metrics):
        filename = build_filename(task, model, metric)
        path = os.path.join(RESULT_DIR, filename)
        status[(task, model, metric)] = os.path.exists(path)

    print("\n4. Baseline results\n")
    print(fmt_row(header))
    print("-" * sum(col_widths))

    for task in class_tasks:
        for model in models:
            row = [task, model]

            for metric in metrics:
                ok = status[(task, model, metric)]
                row.append("✅" if ok else "❌")

            print(fmt_row(row))


if __name__ == "__main__":
    main()
