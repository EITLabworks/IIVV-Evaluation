"""Inspect revision manifests/results without reading stale legacy selection files."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main(kind):
    p=argparse.ArgumentParser(description='Inspect patient-grouped nested revision outputs')
    p.add_argument('--results-dir',default='results/revision')
    p.add_argument('--top',type=int,default=10)
    p.add_argument('--only',choices=['freq','mask','metrics'])
    args=p.parse_args();root=Path(args.results_dir)
    manifest=root/'run_manifest.json'
    if not manifest.exists():
        print(f'No run manifest at {manifest}; execute ./run_revision.sh --preflight first.');return 2
    info=json.loads(manifest.read_text());print(f'Status: {info["status"]}; synthetic: {info.get("synthetic",False)}')
    if kind=='check':
        if info['status']!='completed':print(info.get('error','Run is not completed.'));return 2
        required=[root/'all_metrics.csv',root/'BaselineResults.csv',root/'FinalResults.csv',root/'paired_comparisons_holm.csv',root/'analysis_status.json']
        for task in info['config']['tasks']:
            required.extend(root/task/name for name in ['outer_predictions.csv','inner_search.csv','splits.json','selections.json','classes.json','per_class_metrics.csv','frequency_and_pattern_identities.json'])
        missing=[str(v) for v in required if not v.is_file()]
        if missing:print('Missing required artifacts:\n'+'\n'.join(missing));return 2
        print(f'Validated existence of {len(required)} result artifacts.');print((root/'analysis_status.json').read_text());return 0
    if kind=='tuning' or args.only in ['freq','mask']:
        for path in sorted(root.glob('*/inner_search.csv')):
            frame=pd.read_csv(path)
            if args.only:frame=frame[frame.stage=={'freq':'frequency','mask':'mask'}[args.only]]
            print(path);print(frame.sort_values('objective_auc',ascending=False).head(args.top).to_string(index=False))
    else:
        path=root/'all_metrics.csv'
        if not path.is_file():print(f'{path} has not been generated.');return 2
        print(pd.read_csv(path).to_string(index=False))
    return 0
