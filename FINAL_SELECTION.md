# Reduzierte finale IIVV-/Frequenzsuche

Start: `./run_final.sh`. Konfiguration: `configs/revision-final.json`.
`configs/revision.json`, frühere Ergebnisse und Manuskripte bleiben unverändert.

## Protokoll und Grenzen

Unverändert: vier Tasks, SVM/RF/LR, AUC/Accuracy/F1/Sensitivität als vier
Auswahlziele, Seed 1729, **5 äußere × 4 innere Patientenfolds**, eine Partition.
Das sind 20 neue äußere Berechnungen einschließlich MC/Fold 2. Es ist
patientengruppierte nested CV, **kein echtes Leave-one-patient-out**.
Alte Checkpoints werden weder übernommen noch mit neuen Folds vermischt.

Der neue Suchraum ist nach bisherigen Erfahrungen mit derselben Kohorte
eingeschränkt. Das ist eine adaptive Folgeanalyse, keine unabhängige Bestätigung
und kein Nachweis gleicher Leistung wie beim großen Suchraum. Frühere äußere
Testscores werden vom neuen Lauf nicht gelesen. Alle konkreten Auswahlentscheidungen
fallen erneut anhand der aktuellen inneren Trainings-/Validierungsdaten.

Die Screening-Stufen vergleichen jeweils eine Suchachse bei fester Referenz:
Magnitude, globale Skalierung, Mittelwert-Imputation, balancierte Klassen und
voller Qualitätsbereich. Das ist eine schrittweise Suchheuristik, keine
erschöpfende gemeinsame Optimierung. Die Referenz-Baselines behalten ihren
bisherigen 60%-Cutoff und frequenzweise Skalierung; Abdeckung und Vergleiche auf
gemeinsam auswertbaren Fällen werden wie bisher separat ausgewiesen.

## Was die vorhandenen Daten tatsächlich zeigen

Quelle: `results/revision-finetuning-20261102-122955`. Die Zählungen betreffen
**227 gespeicherte Modell-/Ziel-Gewinner aus innerer Auswahl**, nicht 227 unabhängige
Experimente. Wiederholte Gewinner können dieselbe Konfiguration verwenden.
Dateihashes und Zählungen stehen in `configs/revision-final-evidence.json`.

- PCA: 116/227; Mutual Information: 69/227. Deshalb bleibt MI erhalten.
- Magnitude: 134/227; Magnitude/Phase: 57/227. Beide bleiben erhalten.
- Impedanz-Schwellenmasken: 53/227. Deshalb wird diese Möglichkeit nicht gestrichen.
- Global/frequenzweise skaliert: 204/227. Die zwei protokollierten LR-Ausfälle
  betreffen unskalierte PCA-Kandidaten in Task 2.
- Cutoff 80% oder voller Bereich: 200/227. Das beweist keine klinische Sicherheit.
- RF: 100 Bäume in 46/76, Tiefe 8 in 63/76; die drei Featureanteile wurden
  ähnlich häufig gewählt (27/23/26). Alle drei Anteile bleiben erhalten.

## Jede eingeschränkte Suchachse

| Ausgelassene Option | Verbleibend | Kurze Begründung |
|---|---|---|
| RFE | PCA, MI | Wiederholte Hilfsmodell-Fits; 24/227 Gewinner. Budgetentscheidung, keine bewiesene Unterlegenheit. |
| Permutation als Frequenzselector | PCA, MI | Zusätzliches RF-Modell und wiederholte Permutationen; 18/227 Gewinner. |
| `real_imag` | Magnitude, Magnitude/Phase | Zweikanal-Alternative zusätzlich zur Phase; 24/227 Gewinner. |
| `ratio` | Magnitude, Magnitude/Phase | Zusätzliche Referenzbildung; nur 6/227 Gewinner. |
| `circuit` | Magnitude, Magnitude/Phase | Nichtlineare Ersatzschaltbild-Fits pro Fall; 6/227 Gewinner. |
| Cutoffs 40%, 60% im optimierten Suchraum | 80%, 100.000001% | Weniger Ausschlüsse, zwei statt vier Screening-Optionen; 200/227 bisherige Gewinner liegen hier. 60% bleibt Referenz-Baseline/Audit. |
| Keine Skalierung | Global, frequenzweise | Unskalierte Größen erschweren numerische Konditionierung; genau dort liegen die zwei protokollierten LR-Ausfälle. |
| Median-/Null-Imputation | Mittelwert | Eine Strategie statt drei; Mittelwert in 158/227 Gewinnern. Es wird keine Gleichwertigkeit behauptet. |
| Frequenzzahlen 2, 4, 6–9, 11–30 | 1, 3, 5, 10, 31 | Grobes Raster von sehr klein bis vollständig; der vollständige Frequenzraum bleibt als Schutz gegen erzwungene Reduktion erhalten. |
| Vier Frequenz-/Geometriegewinner je Ziel | Je ein innerer Gewinner | Beseitigt den Hauptteil des kartesischen Produkts; kann gute Zweitplatzierte verwerfen. |
| SVM `poly`, `sigmoid` | `linear`, `rbf` | Zwei Kernel statt vier begrenzen das Suchbudget. Die gestrichenen Kernel waren in 23/76 SVM-Gewinnern enthalten, also nicht nachweislich nutzlos. |
| SVM C=100, 1000 | 0.001, 0.01, 0.1, 1, 10 | Extrem schwache Regularisierung entfällt zugunsten eines begrenzten Suchbudgets. |
| RF 300/800 Bäume | 100 | Weniger Baum-Fits; 100 war in 46/76 RF-Gewinnern gewählt. |
| RF Tiefe 16/unbegrenzt | 8 | Begrenzte Baumkomplexität; Tiefe 8 in 63/76 RF-Gewinnern, unbegrenzt in keinem. |
| LR C=10, 100, 1000 | 0.001, 0.01, 0.1, 1 | Schwerpunkt auf stärkerer Regularisierung; 45/75 LR-Gewinner hatten C=0.001. |
| Zusätzlicher C-Nachlauf wie im letzten Fine-Tuning | Festes obiges Raster | Kein weiterer Nachbarschaftslauf nach der neuen gemeinsamen Suche. |

Nicht gestrichen: alle 13 günstigen Geometriemasken im Screening, fünf
Impedanzquantile, beide Klassengewichtungen, Patientengewichte, alle drei Modelle,
alle vier Auswahlziele, Kalibrierung, SN-/SP-Betriebspunkte, Missingness-Kontrolle,
verfügbare Kovariaten und ihr gematchter EIS-Kontrolllauf. Drei unterschiedliche
Schwellen dürfen nicht verwechselt werden: Qualitäts-Cutoff, Impedanzmaske und
Entscheidungsschwelle auf Trainings-OOF-Wahrscheinlichkeiten. Letztere bleibt
vollständig trainingsintern erhalten; sie verursacht kein großes Modellgitter.
Bereits der alte Code übernahm nur das beste Impedanzquantil in das gemeinsame
Tuning. Diese Regel bleibt bestehen; die Verringerung von fünf auf zwei
Maskenplätze entsteht durch weniger geometrische Zweitplatzierte, nicht durch
das Entfernen eines früheren Fünffachprodukts der Quantile.

## Anzahl der Berechnungen und Performance

Das alte Verfahren bildet **nicht** ein vollständiges Produkt aller JSON-Listen.
Nach eindimensionalem Screening kombiniert es bis zu 4 Frequenzumfänge ×
(4 Geometriemasken + 1 Impedanzmaske) × 62 Modellkonfigurationen × 4 Ziele.
Die 19 vorhandenen großen Folds enthalten 79.936 erfolgreiche einzigartige
Kandidaten, also 319.744 innere Modell-Fits ohne äußere Refits und ohne den
separat berechneten gematchten LR-Kontrolllauf.

Neu: 35 Screening-Kandidaten; höchstens 1 Frequenzumfang × 2 Masken ×
(10 SVM + 3 RF + 4 LR) = **34 Kandidaten je Ziel**. Einschließlich Baselines und
klinischer Kontrollen höchstens **177 innere Kandidaten pro äußerem Fold**,
also **14.160 innere Fits für alle 20 Folds** vor Wiederverwendung identischer
Kandidaten. Das ist eine ungefähr 22,6-fache Verringerung gegenüber den bisher
berechneten 19 großen Folds, **keine zugesagte Laufzeitbeschleunigung**.
Fehlgeschlagene Fits können vor Abschluss aller vier inneren Folds abbrechen.

Der letzte C-Nachlauf war noch kleiner: 511 erfolgreiche Kandidaten über 19 Folds und etwa
0,91 Stunden Gesamtlaufzeit. Der neue Lauf sucht Frequenzen/Masken erneut und
berechnet Vergleichsauswertungen; er ist deshalb kein bloßer Wiederholungslauf
dieser 511 Kandidaten. Eine belastbare Laufzeitschätzung benötigt erste neue
Fold-Zeiten. Die alte maximale Fold-Zeit von 170,5 Stunden ist kein Maßstab für
die bereits vektorisierte und jetzt begrenzte Implementierung.

- Standard: CPUs **224–255**, 16 äußere Worker × höchstens 2 RF-/BLAS-Threads.
  Affinität, Scheduler- und cgroup-Limits können die Workerzahl weiter begrenzen.
- Round-robin startet Folds aller Tasks früh; die aggregierte Reihenfolge bleibt
  deterministisch. Alle 20 Folds werden fertig gerechnet, ohne zeitabhängige Auswahl.
- Trainingsfeatures werden je exakter Trainings-/Testpartition wiederverwendet.
  Zusätzlich wird das trainingsinterne SVM-Gram-Produkt über C-/Kernelvarianten
  derselben Features wiederverwendet. Testdaten bestimmen weder Ranking noch Fit.
- Trainingsraum-LR und BLAS-Kernel bleiben aktiv. RF-Vorhersagen laufen weiterhin
  mit einem Thread für feste Summationsreihenfolge.
- SVM maximal 100.000, LR maximal 20.000 Iterationen, wie beim letzten Nachlauf.
  Das sind Iterations-, keine Zeitlimits. Konvergenzwarnungen führen zur Ablehnung
  des Fits; andere Kandidaten/Folds laufen weiter. Fehlende Streams werden
  ausgewiesen und erhalten keine gepoolten Leistungswerte.
- Der 8-GiB-Featurecache je Worker erlaubt bis zu 128 GiB Cachekapazität,
  zusätzlich zu Eingangsdaten und temporären Arrays. Das ist kein gemessener RAM-Peak.
- Berichtserstellung folgt erst nach Ende der Fit-Worker und verwendet vier Task-Worker.
  Bootstrap und Permutationen bleiben bei jeweils 1000; Statistikauswertung wird nicht verkürzt.

## Ein Startskript auf dem Server

Die aktualisierten Dateien einschließlich neuer Module, Tests und JSON-Dateien
müssen auf dem Server liegen. Die bisherigen `data/`-/`source/`-Dateien genügen;
alte Ergebnisordner werden für den Lauf nicht benötigt.

```bash
tmux new -s iivv-final
cd /home/jt292/IIVV-Evaluation
source ./setup_environment.sh
./run_final.sh
```

Das Skript legt einen neuen Zeitstempelordner an, führt Regressionstests,
Quellprüfung/Audits, alle Folds, Statistiken und Plots aus und prüft anschließend
die Resultate. Es überschreibt keine alten Runs. Mit `Ctrl-b`, dann `d` löst man
sich von tmux; `tmux attach -t iivv-final` verbindet erneut.

```bash
tail -f logs/revision-final-latest.log
```

Das Log nennt `OUTPUT=...`. Es enthält Pipeline-Stufen, alle 60 Sekunden den
Stand der aktiven Jobs, Kandidatenzähler, inneren Fold und aktuelle Fit-Dauer.
`fold-checkpoints/<task>/<fold>/progress.json` ergänzt PID und genaue Parameter.
Eine lange Kandidatenzeit bedeutet nicht automatisch einen Aufhänger.

Nur einen explizit fehlgeschlagenen/unterbrochenen **neuen** Final-Lauf fortsetzen:

```bash
./run_final.sh --resume --output results/revision-final-<Zeitstempel>
```

Für eine andere tatsächlich zugewiesene CPU-Menge: `CPUSET=0-31 ./run_final.sh`.
Ein gleichzeitiger Start mehrerer Kopien beschleunigt diesen Lauf nicht.
Optional: `./run_final.sh --preflight --output results/revision-final-preflight`;
dabei werden Audits/Repeatability-Bootstraps, aber keine Klassifikatoren trainiert.

## Ergebnisse und Reviewer-Text

Die bisherigen Dateien bleiben erhalten: `classification_outputs.csv`,
`all_metrics.csv`, `BaselineResults.csv`, `FinalResults.csv`,
`corrected_resampled_tests_holm.csv`, Klassenmetriken, ROC/Confusion-Matrizen,
Kalibrierung, Betriebspunkte, Patient-/Pathologie-/Demografietabellen und Timings.
Zusätzlich: `review_selected_parameters.csv`, `unavailable_results.csv`,
`final_search_budget.csv`, `final_selection_protocol.json`, `REVIEW_RESULTS.md`.
Der kompatible interaktive `parameter-explorer.html` wird am Ende automatisch
erstellt und lässt sich lokal im Browser öffnen.
`inner_search.csv` enthält sämtliche erfolgreich ausgewerteten Kandidaten und
ihre Rechenzeit; `candidate_failures.json` dokumentiert Ablehnungen.
Am Logende werden die tatsächlichen Laufzeiten der einzelnen äußeren Jobs
ausgegeben; `final_search_budget.csv` speichert sie zusätzlich in Sekunden.
Für gestrichene Selector-/Darstellungsoptionen entsteht kein neuer Ablationsvergleich;
ihr Fehlen belegt keine Unterlegenheit. Begründung und Ausgangszählungen werden als
`FINAL_SELECTION.md` und `revision-final-evidence.json` in den Ergebnisordner kopiert.

Für vorhandene allgemeine Auswertungsnotebooks den neuen Ordner über
`IIVV_RESULTS=results/revision-final-<Zeitstempel>` auswählen, z. B.
`IIVV_RESULTS=results/revision-final-<Zeitstempel> jupyter lab Figures-and-Tables.ipynb`.
Das separate Fine-Tuning-Notebook/-Prüfskript erwartet weiterhin einen alten
Shortlist-Lauf und ist nicht der Prüfer dieses neuen Protokolls.

Textbaustein **zur Methodenänderung**, keine vorweggenommene Ergebnisbehauptung:

> We reduced the computational search budget while retaining the original
> patient-grouped outer and inner validation protocol. Each outer-training
> population screens the retained alternatives and selects the frequency count,
> IIVV mask, preprocessing, classifier and hyperparameters using inner validation
> only. PCA and mutual information remain available, as do geometric and
> training-derived impedance masks. Calibration and operating thresholds use
> inner out-of-fold predictions and are frozen before outer testing. The omitted
> alternatives reflect a computational budget, not demonstrated inferiority.
> Because this design follows earlier analyses of the same cohort, we interpret
> it as an adaptive follow-up rather than independent confirmatory validation.

Lokal werden hierfür keine Modelle oder Evaluationen gestartet. Die neuen
Regressionstests und die tatsächliche Laufzeit müssen auf dem Server geprüft werden.
