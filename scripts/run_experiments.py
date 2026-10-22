"""Run the full study of the paper (resumable). Appends one JSON line per run to results/results.jsonl.

  python scripts/run_experiments.py            # everything (about 4 h on one CPU core)
  python scripts/run_experiments.py --data ai4i --seeds 0 1
"""
import argparse, json, pickle, traceback
from _paths import RESULTS, CKPT
from nff import train

MAIN = ['ENFT', 'Transformer', 'NeuralTSK', 'MLP', 'HGB']
ABL = ['ENFT-noEvolve', 'ENFT-noAttnMod', 'ENFT-noResidual', 'ENFT-rawTokens']

def plan(datasets, seeds):
    p = []
    for d in datasets:
        ms = MAIN + (['LSTM'] if d != 'ai4i' else [])
        p += [(d, m, s) for s in seeds for m in ms]
        if d != 'FD003': p += [(d, m, s) for s in seeds if s < 3 for m in ABL]
        if d != 'ai4i': p += [(d, 'ENFT-rawTokens', s) for s in seeds if s >= 3 or d == 'FD003']
    seen, out = set(), []
    for x in p:
        if x not in seen: seen.add(x); out.append(x)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', default=['ai4i', 'FD001', 'FD003'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    out = RESULTS / 'results.jsonl'
    done = set()
    if out.exists():
        for l in open(out): r = json.loads(l); done.add((r['data'], r['model'], r['seed']))
    for d, m, s in plan(a.data, a.seeds):
        if (d, m, s) in done: continue
        try: r = train.run(d, m, s)
        except Exception: traceback.print_exc(); continue
        if m == 'ENFT' and s == 0:
            pickle.dump({k: r[k] for k in ('_params', '_mask', '_cfg') if k in r}, open(CKPT / f'enft_{d}_s0.pkl', 'wb'))
        rec = {k: v for k, v in r.items() if not k.startswith('_')}; rec.update(data=d, model=m, seed=s)
        open(out, 'a').write(json.dumps(rec, default=float) + '\n'); print(d, m, s, rec['test'], flush=True)
