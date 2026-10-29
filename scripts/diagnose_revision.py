"""Read saved fold timings/progress without importing or training project models."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys


PROGRESS = re.compile(r"(\w+)/(nested_cv_repeat\d+_fold\d+): (\d+) candidates completed")
FINISHED = re.compile(r"(\w+): (nested_cv_repeat\d+_fold\d+) completed \((\d+) candidates\)")
REQUIRED = {"splits.json", "selections.json", "candidate_failures.json",
            "outer_predictions.csv", "inner_search.csv", "task_inclusion.csv",
            "task_label_audit.csv", "classes.json"}


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def log_progress(path):
    progress = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            match = PROGRESS.search(line) or FINISHED.search(line)
            if match:
                key, count = (match[1], match[2]), int(match[3])
                previous = progress.get(key, {})
                progress[key] = dict(count=count, reset=previous.get("reset", False)
                                     or count < previous.get("count", 0))
    return progress


def checkpoint(folder, task, key):
    marker = read_json(folder / "complete.json")
    if marker.get("task") != task or marker.get("fold_key") != key:
        raise ValueError(f"Checkpoint-Identität stimmt nicht: {folder}")
    inventory = marker.get("artifacts", {})
    if not REQUIRED <= inventory.keys():
        raise ValueError(f"Checkpoint-Inventar unvollständig: {folder}")
    for name, expected in inventory.items():
        artifact = (folder / name).resolve()
        if not artifact.is_relative_to(folder.resolve()):
            raise ValueError(f"Ungültiger Artefaktpfad: {name}")
        if digest(artifact) != expected:
            raise ValueError(f"Checkpoint-Prüfsumme stimmt nicht: {artifact}")
    elapsed = float(marker["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"Ungültige Laufzeit: {folder}")
    return elapsed / 3600, read_json(folder / "selections.json")


def saved_fits(records):
    # Multiple objectives can reuse the same outer fit; count its timing once.
    return list({json.dumps(row["candidate"], sort_keys=True): row for row in records}.values())


def target(value):
    try:
        task, repeat, fold = value.split(":")
        repeat, fold = int(repeat), int(fold)
        if not task or repeat < 0 or fold < 0:
            raise ValueError
        return task, repeat, fold
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Format: MC:0:2 oder healthy_vs_cancer:0:4") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/revision-expanded"))
    parser.add_argument("--log", type=Path, default=Path("logs/revision-expanded.log"))
    parser.add_argument("--folds", type=target, nargs="+",
                        default=[("MC", 0, 2), ("healthy_vs_cancer", 0, 4)])
    args = parser.parse_args(argv)
    try:
        manifest = read_json(args.results_dir / "run_manifest.json")
        schedule = read_json(args.results_dir / "parallel_execution.json")["fold_order"]
        progress = log_progress(args.log)
        now = datetime.now(timezone.utc)
        started = datetime.fromisoformat(manifest["started_utc"])
        if started.tzinfo is None or started > now:
            raise ValueError("Ungültige Manifest-Startzeit")
        age_hours = (now - started).total_seconds() / 3600
        print(f"Run: {args.results_dir}; Status: {manifest['status']}")
        print(f"Diagnosezeit: {now.isoformat(timespec='seconds')}")
        print(f"Log-Dateialter: {(now.timestamp()-args.log.stat().st_mtime)/3600:.2f} h "
              "(kein Zeitpunkt des letzten Fold-Fortschritts)")
        print("Nur gespeicherte Daten; keine Fits, Vorhersagen oder Resampling.")
        print("Innere Fit-Zeiten/Warnungen fehlen; Kandidatenzähler ist keine Prozentanzeige.")
        scheduled = {(item["task"], *item["key"]) for item in schedule}
        summaries = []
        for task, repeat, fold in args.folds:
            if (task, repeat, fold, "nested_cv") not in scheduled:
                raise ValueError(f"Fold nicht im Run-Zeitplan: {task}:{repeat}:{fold}")
            wanted = f"nested_cv_repeat{repeat}_fold{fold}"
            print(f"\n{task}, Repeat {repeat}, Fold {fold}:")
            timings, fits, completed_hours = [], [], None
            for item in schedule:
                if item["task"] != task or item["key"][2] != "nested_cv":
                    continue
                r, f, evaluation = item["key"]
                name = f"{evaluation}_repeat{r}_fold{f}"
                folder = args.results_dir / "fold-checkpoints" / task / name
                if not (folder / "complete.json").is_file():
                    continue
                hours, selections = checkpoint(folder, task, item["key"])
                timings.append(hours)
                print(f"  Fertig: Repeat {r}, Fold {f}: {hours:.2f} h")
                if name == wanted:
                    completed_hours = hours
                fits.extend(saved_fits(selections))
            latest = progress.get((task, wanted))
            print(f"  Letzter Kandidatenzähler: {latest['count'] if latest else 'unbekannt'}")
            if latest and latest["reset"]:
                print("  Zähler zurückgesetzt: Log enthält wahrscheinlich mehrere Versuche.")
            warned = [row for row in fits if row.get("convergence_warnings")]
            print(f"  Gespeicherte äußere Fits mit Warnungen: {len(warned)}/{len(fits)}")
            for row in warned[:3]:
                print(f"    Fold {row['outer_fold']}, {row['model']}: {row['convergence_warnings']}")
            timed = [row for row in fits if isinstance(row.get("model_fit_seconds"), (int, float))
                     and math.isfinite(row["model_fit_seconds"]) and row["model_fit_seconds"] >= 0]
            print("  Langsamste gespeicherte äußere Modell-Fits (keine inneren Suchzeiten):")
            for row in sorted(timed, key=lambda row: row["model_fit_seconds"], reverse=True)[:3]:
                print(f"    Fold {row['outer_fold']}: {row['model_fit_seconds']:.2f} s, "
                      f"Features={row.get('input_features')}, "
                      f"Kandidat={json.dumps(row['candidate'], sort_keys=True)}")
            reference = (f"Fertige Vergleichsfolds min/Median/max: "
                         f"{min(timings):.2f}/{median(timings):.2f}/{max(timings):.2f} h"
                         if timings else "Keine fertigen Vergleichsfolds")
            state = (f"FERTIG in {completed_hours:.2f} h" if completed_hours is not None
                     else "ohne vollständigen Checkpoint; exakte bisherige Fold-Laufzeit unbekannt")
            summaries.append(f"{task}:{repeat}:{fold}: {state}. {reference}.")
        print("\nZEITEN AM ENDE")
        print(f"Seit Manifeststart: {age_hours:.2f} h = {age_hours/24:.2f} Tage.")
        print("Das ist das Run-Alter, keine gemessene Laufzeit offener Folds; "
              "Kopien/Resumes erlauben keinen Rückschluss auf deren tatsächliche Rechenzeit.")
        print("\n".join(summaries))
        config = manifest["config"]
        print(f"Threads: fold_workers={config.get('fold_workers')}, "
              f"rf_jobs={config.get('rf_jobs')}, blas_threads={config.get('blas_threads')}.")
        print("Restlaufzeit: nicht belastbar bestimmbar. Zeitgestützter Kandidatenfortschritt fehlt; "
              "Vergleichsfolds begrenzen die Laufzeit offener Folds nicht. "
              "Zusammenführung und finale Statistik kommen danach hinzu.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Diagnose abgebrochen: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
