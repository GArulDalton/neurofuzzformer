"""Aggregate results/results.jsonl into the paper tables.

Writes results/summary.json (all metrics per dataset, model and seed), results/selection.json
(validation based choice between fuzzy augmented and raw value tokens), results/pvals.json
(paired t tests against NeuroFuzzFormer) and results/tables.json (Tables 4 to 11).
Run scripts/explain_rules.py for ai4i and FD001 first (needed for Table 8).

  python scripts/analyze_results.py
"""
from _paths import RESULTS
import json, numpy as np, pandas as pd
from scipy import stats
R = [json.loads(l) for l in open(RESULTS / 'results.jsonl')]
rows = []
for r in R:
    d = dict(data=r['data'], model=r['model'], seed=r['seed'], **r['test'], **{'conf_'+k: v for k, v in r['conf'].items()},
             params=r.get('params'), lat=r.get('latency_ms'), train_s=r.get('train_s'), fid=r.get('fidelity'), fid_corr=r.get('fid_corr'),
             rules=r.get('rules'), gate=r.get('gate'))
    rows.append(d)
df = pd.DataFrame(rows)
def ms(x, p=3): return f"{np.mean(x):.{p}f} ± {np.std(x, ddof=1):.{p}f}" if len(x) > 1 else f"{np.mean(x):.{p}f}"
out = {}
for d, g in df.groupby('data'):
    out[d] = {}
    for m, gm in g.groupby('model'):
        out[d][m] = {c: [float(v) for v in gm.sort_values('seed')[c].values if v is not None and not (isinstance(v, float) and np.isnan(v))]
                     for c in gm.columns if c not in ('data', 'model')}
json.dump(out, open(RESULTS / 'summary.json', 'w'), indent=1)

# ---- validation selection and paired tests ----

val = {}
for r in R:
    if r['model'] in ('ENFT', 'ENFT-rawTokens'): val.setdefault((r['data'], r['model']), {})[r['seed']] = max(h['val'] for h in r['hist'])
sel = {}
for d in ['ai4i', 'FD001', 'FD003']:
    a, b = val.get((d, 'ENFT'), {}), val.get((d, 'ENFT-rawTokens'), {})
    common = sorted(set(a) & set(b))
    sel[d] = 'ENFT-rawTokens' if common and np.mean([b[s] for s in common]) > np.mean([a[s] for s in common]) else 'ENFT'
    print(d, sel[d], {s: (round(a[s], 3), round(b[s], 3)) for s in common})
json.dump(sel, open(RESULTS / 'selection.json', 'w'))
S = out; P = {}
for d in S:
    for m in S[d]:
        for k in ['F1', 'AUPRC', 'RMSE']:
            if k in S[d][m] and S[d][m][k] and 'ENFT' in S[d]:
                a = np.array(S[d]['ENFT'][k]); b = np.array(S[d][m][k]); n = min(len(a), len(b))
                if n >= 2 and m != 'ENFT': P[f'{d}|{m}|{k}'] = float(stats.ttest_rel(a[:n], b[:n]).pvalue)
json.dump({'ttest': P}, open(RESULTS / 'pvals.json', 'w'))

# ---- tables ----
S = out
NAME = {'ENFT': 'NeuroFuzzFormer (proposed)', 'Transformer': 'Transformer', 'NeuralTSK': 'NeuralTSK', 'MLP': 'MLP', 'LSTM': 'LSTM', 'HGB': 'HGB',
        'ENFT-noEvolve': 'without rule evolution', 'ENFT-noAttnMod': 'without attention modulation', 'ENFT-noResidual': 'without residual path', 'ENFT-rawTokens': 'raw value tokens'}
def ms(x, p=3): x = np.array(x, float); return f"{x.mean():.{p}f} ± {x.std(ddof=1):.{p}f}"
def pt(d, m, k, seeds=5):
    a = np.array(S[d]['ENFT'][k][:seeds]); b = np.array(S[d][m][k][:seeds]); n = min(len(a), len(b))
    t, p = stats.ttest_rel(a[:n], b[:n]); return float(a[:n].mean()-b[:n].mean()), float(p)
T = {}
order = ['ENFT', 'Transformer', 'NeuralTSK', 'MLP', 'HGB']
T['ai4i'] = [[NAME[m], ms(S['ai4i'][m]['F1']), ms(S['ai4i'][m]['AUPRC']), ms(S['ai4i'][m]['AUROC']), ms(S['ai4i'][m]['bacc'])] for m in order]
for d in ['FD001', 'FD003']:
    if d not in S: continue
    T[d] = [[NAME[m], ms(S[d][m]['RMSE'], 2), ms(S[d][m]['MAE'], 2), ms(S[d][m]['Score'], 0)] for m in order + ['LSTM'] if m in S[d] and len(S[d][m]['seed']) >= 2]
# conformal
C = []
for d in ['ai4i', 'FD001', 'FD003']:
    if d not in S: continue
    for m in [x for x in ['ENFT', 'Transformer', 'HGB'] if x in S[d] and len(S[d][x]['seed']) >= 2]:
        o = S[d][m]
        C.append([{'ai4i': 'AI4I', 'FD001': 'FD001', 'FD003': 'FD003'}[d], NAME[m], ms(o['conf_cov']), ms(o['conf_size']) if d == 'ai4i' else ms(o['conf_width'], 1) + ' cycles'])
T['conf'] = C
# ablation (3 seeds)
A = []
for d, k, p in [('ai4i', 'F1', 3), ('ai4i', 'AUPRC', 3), ('FD001', 'RMSE', 2)]:
    pass
for m in ['ENFT', 'ENFT-noEvolve', 'ENFT-noAttnMod', 'ENFT-noResidual', 'ENFT-rawTokens']:
    row = [NAME[m]]
    for d, k, pr in [('ai4i', 'F1', 3), ('ai4i', 'AUPRC', 3), ('ai4i', 'fid', 3), ('FD001', 'RMSE', 2), ('FD001', 'fid_corr', 3)]:
        v = S.get(d, {}).get(m, {}).get(k, [])[:3]
        row.append(ms(v, pr) if len(v) >= 2 else '–')
    row.append(f"{np.mean(S['ai4i'][m]['rules'][:3]):.1f} / {np.mean(S['FD001'][m]['rules'][:3]):.1f}" if m in S.get('FD001', {}) and S['FD001'][m]['rules'] else '–')
    A.append(row)
T['abl'] = A
# efficiency
E = []
for m in ['ENFT', 'Transformer', 'NeuralTSK', 'MLP', 'LSTM', 'HGB']:
    r = [NAME[m]]
    for d in ['ai4i', 'FD001']:
        o = S[d].get(m)
        if not o: r += ['–', '–', '–']; continue
        r += [f"{int(o['params'][0]):,}" if o['params'] else 'tree ensemble', ms(o['lat'], 2), f"{np.mean(o['train_s']):.0f}"]
    E.append(r)
T['eff'] = E
# stats
ST = []
for d, k in [('ai4i', 'F1'), ('ai4i', 'AUPRC'), ('FD001', 'RMSE'), ('FD003', 'RMSE')]:
    if d not in S: continue
    for m in ['Transformer', 'NeuralTSK', 'MLP', 'HGB'] + (['LSTM'] if d != 'ai4i' else []):
        if m in S[d] and len(S[d][m]['seed']) >= 2 and len(S[d]['ENFT']['seed']) >= 2:
            diff, p = pt(d, m, k); ST.append([{'ai4i': 'AI4I', 'FD001': 'FD001', 'FD003': 'FD003'}[d], k, NAME[m], f"{diff:+.3f}" if k != 'RMSE' else f"{diff:+.2f}", f"{p:.3f}" if p >= 0.001 else '< 0.001'])
T['stats'] = ST
# fidelity / rules
F = []
for d in ['ai4i', 'FD001', 'FD003']:
    if d in S:
        o = S[d]['ENFT']
        F.append([{'ai4i': 'AI4I'}.get(d, d), ms(o['fid']), ms(o['fid_corr']) if o.get('fid_corr') else '–', f"{np.mean(o['rules']):.1f} ({int(min(o['rules']))}–{int(max(o['rules']))})", ms(o['gate'])])
T['fid'] = F
json.dump(T, open(RESULTS / 'tables.json', 'w'), indent=1, ensure_ascii=False)

# combined RUL table incl. raw-token variant
RU = []
for m in ['ENFT', 'Transformer', 'NeuralTSK', 'MLP', 'LSTM', 'HGB', 'ENFT-rawTokens']:
    r = [NAME[m] if m != 'ENFT-rawTokens' else 'NeuroFuzzFormer, raw value tokens (validation selected)']
    for d in ['FD001', 'FD003']:
        o = S.get(d, {}).get(m)
        if o and len(o['seed']) >= 2: r += [ms(o['RMSE'], 2), ms(o['MAE'], 2), ms(o['Score'], 0)]
        else: r += ['–', '–', '–']
    RU.append(r)
T['rul'] = RU
RL = []
for d, lab in [('ai4i', 'AI4I'), ('FD001', 'FD001')]:
    ex = json.load(open(RESULTS / f'explain_{d}.json'))
    for r in ex['rules']:
        cons = r['cons']
        if 'RUL' in cons:
            v = float(cons.split('≈')[1].split()[0]); cons = f'RUL ≈ {max(v, 0):.0f} cycles'
        RL.append([lab, r['ante'], cons, f"{r['support']:.3f}"])
T['rules'] = RL
json.dump(T, open(RESULTS / 'tables.json', 'w'), indent=1, ensure_ascii=False)
for d in ['FD001', 'FD003']:
    for m in ['Transformer', 'LSTM', 'HGB']:
        a = np.array(S[d]['ENFT-rawTokens']['RMSE']); b = np.array(S[d][m]['RMSE'])
        T['stats'].append([d, 'RMSE', f'{NAME[m]} (vs. validation selected variant)', f'{a.mean()-b.mean():+.2f}', (lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}')(stats.ttest_rel(a, b).pvalue)])
json.dump(T, open(RESULTS / 'tables.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    for d in ['ai4i', 'FD001', 'FD003']:
        print('==', d)
        for m, o in out[d].items():
            key = 'F1' if d == 'ai4i' else 'RMSE'
            print(f"  {m:18s} n={len(o['seed'])} {key} {ms(o[key], 3 if d == 'ai4i' else 2)}")
