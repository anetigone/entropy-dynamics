"""Read existing Tent tasks and check event coverage against saved risk curves."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(folder):
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    groups = {}
    tasks = []
    mismatches = 0
    count = 0
    for path in sorted(folder.glob('task_*.npz')):
        with np.load(path) as d:
            meta = json.loads(str(d['metadata']))
            curves = d['risk_changes']
            initial = d['initial_risks']
            features = d['features']
            if curves.shape != (protocol['futures'],protocol['horizon']+1):
                raise ValueError(f'Risk shape mismatch: {path.name}')
            if not all(np.isfinite(x).all() for x in (curves,initial,features)):
                raise ValueError(f'Nonfinite values: {path.name}')
            if not np.allclose(curves[:,0],0):
                raise ValueError(f'Risk origin mismatch: {path.name}')
            events = curves[:,1:].max(1)>protocol['epsilon']
            mismatches += int(np.sum(events != d['events']))
            key = (meta['corruption'],meta['severity'],meta['lr'],meta['start_step'])
            groups.setdefault(key,[]).append((curves.copy(), initial.copy(),events))
            first_hit = [int(np.flatnonzero(row[1:]>protocol['epsilon'])[0]+1)
                         if event else None for row,event in zip(curves,events)]
            tasks.append(dict(task=meta['task'],corruption=key[0],severity=key[1],
                lr=key[2],start_step=key[3],repetition=meta['repetition'],
                events=int(events.sum()),futures=len(events),first_hit_steps=first_hit,
                initial_error_mean=float(initial.mean()),
                final_change_mean=float(curves[:,-1].mean()),
                max_deterioration=float(curves[:,1:].max())))
            count += 1
    rows = []
    for key, values in sorted(groups.items()):
        r = np.concatenate([v[0] for v in values])
        initial = np.concatenate([v[1] for v in values])
        events = np.concatenate([v[2] for v in values])
        # Rebound from a later attained minimum is distinct from the declared event.
        rebound = r-np.minimum.accumulate(r,axis=1)
        rows.append(dict(corruption=key[0],severity=key[1],lr=key[2],start_step=key[3],
            starts=len(values),starts_with_events=sum(bool(v[2].any()) for v in values),
            futures=len(r),events=int(events.sum()),
            initial_error_mean=float(initial.mean()),final_change_mean=float(r[:,-1].mean()),
            final_change_min=float(r[:,-1].min()),final_change_max=float(r[:,-1].max()),
            max_deterioration=float(r[:,1:].max()),max_rebound=float(rebound.max()),
            mean_curve=r.mean(0).tolist()))
    return dict(protocol=protocol,observed_tasks=count,declared_tasks=protocol['task_count'],
        event_mismatches=mismatches,total_events=sum(r['events'] for r in rows),
        total_futures=sum(r['futures'] for r in rows),groups=rows,tasks=tasks)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    result = json.dumps(summarize(args.tasks),indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(result,encoding='utf-8')
    else:
        print(result,end='')


if __name__ == '__main__':
    main()
