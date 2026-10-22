"""Extract linguistic rules and an example explanation from the seed 0 NeuroFuzzFormer checkpoint.

  python scripts/explain_rules.py ai4i      ->  results/explain_ai4i.json
  python scripts/explain_rules.py FD001     ->  results/explain_FD001.json
"""
import pickle, numpy as np, jax, jax.numpy as jnp, json, sys
from _paths import RESULTS, CKPT, FIGS
from nff import models as Mm, data as Dm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
TERMS = ['LOW', 'MEDIUM', 'HIGH']
def load(d):
    k = pickle.load(open(CKPT / f'enft_{d}_s0.pkl', 'rb')); return k['_params'], k['_mask'], k['_cfg']
def rules_table(d, top=6, nvar=3):
    p, mask, cfg = load(d); D = Dm.load(d, 0); names = D['names']; X = D['Xte']
    out, aux = Mm.forward(p, jnp.asarray(mask), jnp.asarray(X[:4096]), cfg)
    wbar = np.asarray(aux['wbar']); s = np.asarray(aux['s'])
    P = np.asarray(jax.nn.softmax(p['rules']['P'], -1)); A = np.asarray(jax.nn.softplus(p['rules']['A'])); C = np.asarray(p['rules']['C'])
    act = np.where(mask > 0)[0]; sup = wbar.mean(0)
    risk = (np.asarray(jax.nn.softmax(C, -1))[:, 1] if D['task'] == 'clf' else -C[:, 0])
    order = list(act[np.argsort(-sup[act])][:3]) + [r for r in act[np.argsort(-risk[act])] if r not in act[np.argsort(-sup[act])][:3]][:3]; rows = []
    for r in order:
        js = np.argsort(-A[r])[:nvar]
        ante = ' AND '.join(f'{names[j]} is {TERMS[int(P[r, j].argmax())]}' for j in js)
        if D['task'] == 'clf': cons = f'failure probability {float(jax.nn.softmax(C[r])[1]):.2f}'
        else: cons = f'RUL ≈ {float(C[r, 0])*D["cap"]:.0f} cycles'
        rows.append(dict(rule=int(r), support=float(sup[r]), ante=ante, cons=cons))
    # example: a failure / low RUL test window
    yt = D['yte'][:4096]; full = np.asarray(out)
    i = int(np.where(yt == 1)[0][0]) if D['task'] == 'clf' else int(np.argmin(yt[:len(full)]))
    ex = []
    for r in np.argsort(-wbar[i])[:2]:
        js = np.argsort(-A[r] * s[i] * len(names))[:3]
        ex.append(dict(rule=int(r), w=float(wbar[i, r]), ante=' AND '.join(f'{names[j]} is {TERMS[int(P[r, j].argmax())]}' for j in js),
                       cons=(f'failure probability {float(jax.nn.softmax(C[r])[1]):.2f}' if D['task']=='clf' else f'RUL ≈ {float(C[r,0])*D["cap"]:.0f} cycles')))
    exinfo = dict(index=i, y=float(yt[i]), pred=(float(jax.nn.softmax(full[i])[1]) if D['task']=='clf' else float(full[i,0])*D['cap']), top=ex,
                  rel={names[j]: float(s[i, j]*len(names)) for j in np.argsort(-s[i])[:4]})
    rel = s.mean(0) * len(names)
    return rows, dict(zip(names, rel.tolist())), int(len(act)), exinfo
def mf_fig(d, vars_, path):
    p, mask, cfg = load(d); D = Dm.load(d, 0)
    c, sg = Mm.centers(p['fz']); c, sg = np.asarray(c), np.asarray(sg)
    xs = np.linspace(-0.05, 1.05, 300)
    fig, axs = plt.subplots(1, len(vars_), figsize=(3.4*len(vars_), 2.7))
    for ax, j in zip(axs, vars_):
        mu = np.exp(-0.5*((xs[:, None]-c[j])/sg[j])**2); mu = mu/mu.sum(1, keepdims=True)
        for m, (lab, col) in enumerate(zip(TERMS, ['#1f77b4', '#2ca02c', '#d62728'])): ax.plot(xs, mu[:, m], color=col, label=lab)
        for m in range(3): ax.axvline([0.15, 0.5, 0.85][m], color='grey', ls=':', lw=0.8)
        ax.set_title(D['names'][j], fontsize=10); ax.set_xlabel('normalized value'); ax.set_ylim(0, 1.05)
    axs[0].set_ylabel('membership'); axs[0].legend(fontsize=8, loc='center left')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
if __name__ == '__main__':
    d = sys.argv[1]; rows, rel, nact, ex = rules_table(d); print(ex)
    print(nact); [print(r) for r in rows]; print(sorted(rel.items(), key=lambda x: -x[1]))
    json.dump(dict(rules=rows, rel=rel, active=nact, example=ex), open(RESULTS / f'explain_{d}.json', 'w'), indent=1)
