# A Machine-Learning Approach to Optimize Electrical Impedance Spectroscopy-based Classification of In Vivo Oral Lesions

Repository for the IEEE-TBME paper: _A Machine-Learning Approach to Optimize Electrical Impedance Spectroscopy-based Classification of In Vivo Oral Lesions_

## IIVV selection

Configure the following shell script for the metric and start it to run it for all tasks. F1-score is preferrec in medical applications.

    ./1_1run_tmux_iivv.sh

To see the tuning results, run

    ./print_eval_results.py --only mask

## Frequency and z-threshold rankings

Configure the following shell script for the metric and start it to run it for all tasks. F1-score is preferrec in medical applications.

    ./1_2run_tmux_freq.sh

To see the tuning results, run

    ./print_eval_results.py --only freq

## Model and parameter tuning

Configure the following shell script for the metric and start it to run it for all tasks. F1-score is preferrec in medical applications.

    ./2_run_tmux_model_tuning.sh

To see the tuning results, run

    ./print_tuning_results.py

## Show the best performing models

    ./print_tuning_results.py --top 4

or of an specific file

    ./print_tuning_results.py --pattern "GPU_tuning_*AGG.csv" --top 4

## Create plots and final results

    ./3_final_results.py --task MC --metric f1 --model svm

## Create baseline results and plots

    ./4_results_baseline.py --task healthy_vs_cancer
    ./4_results_baseline.py --task cancer_vs_lr_hr
    ./4_results_baseline.py --task MC

## Check the available/created results

    ./5_check_results.py
___


**Documentation of the script options**

_script.py_ and   -h, --help            shows help message and exit

___

For referencing this work, please cite:

```
TBD:Citation paper
```

and the repository:

```
@software{IIVV_Evaluation_2025,
    author = {Thönes, Jacob P., Sophie A. Lloyd},
    license = {MIT},
    month = sep,
    title = {{IIVV Evaluation}},
    url = {https://github.com/EITLabworks/IIVV-Evaluation},
    version = {0.1.0},
    year = {2025}
}
```