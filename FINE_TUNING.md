# Reduziertes Fine-Tuning für die Revision

## Ausgangspunkt und Grenzen

Quelle: `results/revision-expanded`, Basis-Seed 1729, fünf äußere und vier innere
patientengetrennte Folds, eine äußere Partition. Die Konfiguration friert die
aktuell verfügbaren **19 Folds** ein: HC, Cancer-vs-LR/HR und High-risk-vs-Low-risk
je fünf, MC nur 0, 1, 3 und 4. **MC_2 wird weder abgewartet noch ersetzt.** Auch
wenn sein ursprünglicher Checkpoint später erscheint, bleibt diese Konfiguration
bei vier MC-Folds. Eine spätere vollständige Variante braucht eine neue Config
und einen neuen Ergebnisordner.

Die Auswahl stammt für jeden Fold ausschließlich aus dessen bereits gespeicherten
`auc_<model>`, `acc_<model>`, `f1_<model>` und `sensitivity_<model>`-Gewinnern.
Äußere AUC/F1-Werte werden nicht zur Kandidatenauswahl gelesen. Kombinationen
anderer äußerer Folds werden nicht übertragen, weil deren Training die aktuellen
Testpatienten enthalten kann. Dies ist eine zusätzliche adaptive Analyse auf
derselben Partition, **keine neue unabhängige Bestätigung**. Die ursprünglichen
Ergebnisse bleiben erhalten; die neue Analyse soll sie nicht nachträglich ersetzen.

## Was wird gerechnet?

* Drei Modelle: SVM, RF, LR; vier Auswahlziele: AUC, Accuracy, F1, Sensitivität.
* Frequenzzahl, Maskentyp, Selektor, Repräsentation, Completeness-Cutoff,
  Normalisierungs- und Imputationsstrategie sowie Klassengewichtung bleiben bei
  der jeweiligen fold-internen Gewinnerkombination.
* SVM/LR: C/2, C und 2C innerhalb des ursprünglichen Bereichs 0,001–1.000.
  RF: nur seine gespeicherten Gewinnerparameter; keine neue RF-Gittersuche.
* Konkrete Frequenz-/Patternidentitäten, Imputations-/Skalierungsparameter,
  Klassifikator, OOF-Kalibrierung und Schwellen werden neu aus den jeweiligen
  Trainingsdaten gelernt. Es werden keine alten Vorhersagen als neue ausgegeben.
* Drei Baselines und verfügbare klinische/EIS-Kontrollen bleiben erhalten.
  Bereits vorhandene Feature-Caches und Kandidaten-/Outer-Fit-Wiederverwendung
  vermeiden Mehrfacharbeit zwischen den vier Zielen.
* Breites Screening und Ablationen entfallen. Fehlende Ablationen werden als
  unavailable dokumentiert; hierfür die ursprüngliche Auswertung verwenden.

Die SVM erhält auch in der Baseline `max_iter=100000`. LR erhält im Fine-Tuning
`lr_max_iter=20000` (auch für Baselines und LR-Kontrollen), weil bei 5.000 bereits
alle LR-Kandidaten eines Folds das Limit erreichten. Der ursprüngliche breite
Modus bleibt bei 5.000. Das höhere Limit verändert weder C noch Skalierung,
Toleranz oder Zielfunktion und wird nur bei Bedarf ausgeschöpft. Es garantiert
keine Konvergenz. Bei geändertem Budget einen neuen Ergebnisordner verwenden,
z. B. `bash run_finetuning.sh --output results/revision-finetuning-lr20k`;
ein Resume mit abweichender Konfiguration wird bewusst abgelehnt.
Konvergenzwarnungen bei Modell-Fits führen im neuen Modus zur Ablehnung des
betreffenden Kandidaten. Wenn ein Modell keinen gültigen Kandidaten hat, wird
sein Stream übersprungen; andere Modelle und Folds werden weitergerechnet.
Nicht konvergierte äußere Fits und LR-Kontrollen werden ebenfalls dokumentiert
und ausgelassen. Daten-/Splitfehler brechen weiterhin ab.
Das SVM-Limit ist **kein Zeitlimit** und garantiert
keine bestimmte Laufzeit; diese Solveränderung ist im Methodenbericht anzugeben.
Der ursprüngliche breite Modus behält sein bisheriges Verhalten.

## Gemessener Suchumfang der vorhandenen Dateien

Zählung der gespeicherten erfolgreichen `inner_search.csv`-Zeilen und der
deduplizierten neuen Kombinationen, ohne Fits auszuführen:

| Task | Folds | Bisherige Kandidaten | Neue inkl. Baselines | Neue innere Fits |
|---|---:|---:|---:|---:|
| healthy_vs_cancer | 5 | 18.670 | 113 | 452 |
| cancer_vs_lr_hr | 5 | 22.266 | 140 | 560 |
| MC | 4 | 19.214 | 110 | 440 |
| highrisk_vs_lowrisk | 5 | 19.786 | 112 | 448 |
| Summe | 19 | 79.936 | 475 | 1.900 |

Das sind rund **99,4 % weniger Kandidaten**. Die alte Zahl zählt erfolgreiche
Kandidaten und lässt Fehlversuche aus; die neue enthält bereits drei Baselines
pro Fold. Klinische Kontrollen, äußere Fits, Feature-Selektion, Bootstrap und
Permutation sind nicht in der Fit-Spalte enthalten. Sie verhindern eine direkte
Umrechnung dieses Faktors in Stunden. Eine ehrliche Wall-Time-Prognose benötigt
Messungen des neuen Laufs; insbesondere ein großer RF oder schwieriger SVM-Fit
kann weiterhin dominieren.

## Serverstart und Performance

Alle neuen/geänderten Python-Dateien, `run_finetuning.sh` und
`configs/revision-finetuning.json` auf den Server übertragen. Den alten Prozess
nicht neu starten und dessen Ordner nicht überschreiben. Nur unveränderte, durch
`complete.json` abgeschlossene Quellfolds werden akzeptiert. Eingabedatei-Hashes,
wissenschaftliche Grundeinstellungen und äußere/innere Patientenaufteilungen
müssen übereinstimmen. Die neue Ausgabe enthält den eingefrorenen Plan samt
Original-Checkpointinventaren in `fine_tuning_plan.json`; Resume lehnt Änderungen ab.

Standardmäßig CPUs **224–255**, also 32 logische CPUs; acht Fold-Worker mit vier
RF-Threads, ein BLAS-Thread, bis zu 64 GiB Feature-Caches plus Datenspeicher und
temporäre Arrays. Die CPU-IDs garantieren weder 32 physische Kerne noch exklusive
Nutzung; andere Runs auf denselben CPUs können diesen Lauf verlangsamen.

Auf dem Server zuerst die vorbereiteten Tests ausführen (sie wurden hier nicht
ausgeführt):

```sh
cd /home/jt292/IIVV-Evaluation
conda activate iivv-revision
python -m pytest -q tests/test_finetuning.py tests/test_revision.py
```

Danach optional Eingabe-/Planprüfung; dabei werden auch die bestehenden
Qualitäts-/Repeatability-Audits einschließlich ihrer Bootstraps berechnet:

```sh
bash run_finetuning.sh --preflight --output results/revision-finetuning-preflight
```

Neuer vollständiger Lauf über die 19 freigegebenen Folds:

```sh
tmux new -s revision-finetuning
cd /home/jt292/IIVV-Evaluation
conda activate iivv-revision
mkdir -p logs
bash run_finetuning.sh > logs/revision-finetuning.log 2>&1
```

Session verlassen: Ctrl+B, D. Wiedereinstieg: `tmux attach -t revision-finetuning`.
Fortschritt: `tail -f logs/revision-finetuning.log`. Nur bei unterbrochenem oder
fehlgeschlagenem neuen Lauf: `bash run_finetuning.sh --resume`.

Nach Abschluss:

```sh
python scripts/check_finetuning.py --results-dir results/revision-finetuning
```

Das prüft Struktur, Daten-/Checkpoint-Hashes und Patiententrennung, keine klinische
Eignung. Der normale `revision-check` und das Finalergebnis-Notebook verweigern
weiterhin einen unvollständigen Gesamtversuch. Diese Schutzfunktion bleibt bestehen.

## Dateien für die Reviewer

### Fertigen Serverlauf lokal anzeigen

Den vollständigen Ordner kopieren, einschließlich `fold-checkpoints/`, Plan,
Manifesten, Tabellen und Abbildungen. Für den am 2. November abgeschlossenen Run
auf dem lokalen PC im Repository ausführen:

```sh
mkdir -p results logs
scp -r jt292@excalibur.iae.ief.uni-rostock.de:/home/jt292/IIVV-Evaluation/results/revision-finetuning-20261102-122955 ./results/
scp 'jt292@excalibur.iae.ief.uni-rostock.de:/home/jt292/IIVV-Evaluation/logs/revision-finetuning*.log' ./logs/
conda activate iivv-revision
jupyter lab Fine-Tuning-Results.ipynb
```

Erst nach Abschluss des Kopierens im Notebook alle Zellen ausführen. Es liest nur
vorhandene Dateien, prüft die kopierten Checkpoint-Hashes und zeigt Metriken/CIs,
Parameter, Fehler, CV-Statistik, Demografie und gespeicherte Plots. Es startet kein
Training, Bootstrap oder neue Statistik. Für einen anderen Ergebnisordner:
`IIVV_FINETUNING_RESULTS=results/anderer-run jupyter lab Fine-Tuning-Results.ipynb`.
`completed_partial` ist für diesen 19-Fold-Lauf der erwartete Abschlussstatus.
Das Notebook macht daraus keinen vollständigen MC-Endpunkt.

Unter `results/revision-finetuning/`:

* `REVIEW_RESULTS.md`: Abdeckung und Grenzen; MC ausdrücklich vorläufig.
* `review_selected_parameters.csv`: Modelle, Parameter, Frequenzen,
  Featurezahl und äußere Fit-Zeiten pro Fold/Ziel.
* `fine_tuning_budget.csv`: alte/neue Kandidatenzahl und geplante innere Fits
  pro Fold, ohne Kontrollmodelle.
* `classification_outputs.csv`: AUC, F1, Precision/PPV, SN, SP, NPV und die
  bestehenden patientenweisen Bootstrap-Konfidenzintervalle (1.000 Ziehungen).
* `all_metrics.csv`, `BaselineResults.csv`, `FinalResults.csv`: Vergleichstabellen.
* `unavailable_results.csv`: nicht verfügbare Streams. Bei fehlenden Modellfolds
  werden gepoolte Modellkennzahlen vollständig zurückgehalten; es wird keine
  Auswahl nur erfolgreicher Folds berichtet. Korrigierte Tests behalten ihre
  Hypothesen bei, fehlende Foldpaare führen zu NaN statt scheinbarer Signifikanz.
* `corrected_resampled_tests_holm.csv`: korrigierte CV-Tests und Holm-Korrektur
  der 288 eindeutigen Hypothesen; PPV ist ein Alias von Precision. Bei MC nur
  vier Folds/df=3 und explorativer Geltungsbereich.
* Task-Unterordner: Klassenmetriken, Demografie/Pathologie der Folds,
  Frequenz-/Patternidentitäten, ROC/Confusion-Matrix-PDFs, Kalibrierung,
  Schwellenanalysen, Konvergenzfehler und Rechenzeiten.

CSV-Berichte tragen `evidence_status`, `folds_available` und `folds_expected`.
MC-Abbildungen sind mit `PARTIAL EXPLORATORY` beschriftet. Der Gesamtstatus lautet
bewusst `completed_partial`, obwohl alle **geplanten 19** Jobs fertig sind.
MC-Kennzahlen und Intervalle beziehen sich nur auf die vier vorhandenen Folds;
sie erlauben keine endgültige Aussage für die gesamte MC-Kohorte. Bei den übrigen
drei Tasks ist die ursprüngliche äußere Partition vollständig, die neue Analyse
bleibt dennoch eine adaptive Nachanalyse. Es werden weder Paper noch Response
Letter automatisch überschrieben.
