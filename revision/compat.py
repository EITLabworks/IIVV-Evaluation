"""Compatibility entry points; independent cohort-wide selection is no longer valid."""
import sys


def main(stage):
    from run_revision import main as run
    argv=list(sys.argv[1:])
    aliases={'--data_path':'--data','--data-path':'--data','--results_dir':'--output',
             '--results-path':'--output','--outdir':'--output','--n_folds':'--outer-folds',
             '--n_trials':'--outer-repeats'}
    argv=[aliases.get(v,v) for v in argv]
    if '--task' in argv:
        i=argv.index('--task')
        if i+1<len(argv) and argv[i+1]=='ALL':del argv[i:i+2]
        else:argv[i]='--tasks'
    if '--metric' in argv:argv[argv.index('--metric')]='--objectives'
    if '--model' in argv:
        i=argv.index('--model')
        if i+1<len(argv):argv[i+1]=argv[i+1].lower()
        argv[i]='--models'
    print(f'{stage}: selection, tuning and scoring now run together inside patient-grouped nested CV. '
          'Use run_revision.sh for the complete revision workflow.',file=sys.stderr)
    return run(argv)
