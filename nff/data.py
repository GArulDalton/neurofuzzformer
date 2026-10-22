"""Data loading for the AI4I 2020 and NASA C-MAPSS benchmarks.

All windows are min-max normalised with statistics of the training split only.
Set the environment variable NFF_DATA to use a different data folder.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

D = os.environ.get('NFF_DATA', str(Path(__file__).resolve().parents[1] / 'data'))

def _mm(tr, *others):
    lo, hi = tr.min(axis=tuple(range(tr.ndim-1))), tr.max(axis=tuple(range(tr.ndim-1)))
    sc = np.where(hi-lo > 1e-9, hi-lo, 1.0)
    f = lambda a: np.clip((a-lo)/sc, -0.25, 1.25).astype(np.float32)
    return [f(tr)] + [f(o) for o in others]

def load_ai4i(seed):
    d = pd.read_csv(f'{D}/ai4i2020.csv')
    num = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
    t = d['Type'].map({'L':0.0,'M':0.5,'H':1.0}).values[:,None]
    X = np.concatenate([d[num].values, t], 1).astype(np.float64)
    y = d['Machine failure'].values.astype(np.int32)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.35, stratify=y, random_state=seed)
    va, rest2 = train_test_split(rest, test_size=25/35, stratify=y[rest], random_state=seed)
    ca, te = train_test_split(rest2, test_size=0.6, stratify=y[rest2], random_state=seed)
    Xtr, Xva, Xca, Xte = _mm(X[tr], X[va], X[ca], X[te])
    names = ['Air temperature','Process temperature','Rotational speed','Torque','Tool wear','Product quality']
    r = lambda a: a[:, None, :]
    return dict(task='clf', K=2, names=names, Xtr=r(Xtr), ytr=y[tr], Xva=r(Xva), yva=y[va], Xca=r(Xca), yca=y[ca], Xte=r(Xte), yte=y[te])

SENS = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
def load_cmapss(fd, seed, W=30, cap=125):
    cols = ['unit','cycle','o1','o2','o3'] + [f's{i}' for i in range(1,22)]
    rd = lambda f: pd.read_csv(f'{D}/{f}', sep=r'\s+', header=None, names=cols)
    tr = rd(f'train_{fd}.txt'); te = rd(f'test_{fd}.txt')
    rul = np.loadtxt(f'{D}/RUL_{fd}.txt')
    sc = [f's{i}' for i in SENS]
    units = tr.unit.unique(); rng = np.random.RandomState(seed); rng.shuffle(units)
    nh = len(units)//10
    sets = {'va': units[:nh], 'ca': units[nh:2*nh], 'tr': units[2*nh:]}
    def build(df, us):
        X, y = [], []
        for u in us:
            g = df[df.unit == u]; a = g[sc].values; mx = g.cycle.max()
            r = np.minimum(mx - g.cycle.values, cap)
            for i in range(len(a)-W+1): X.append(a[i:i+W]); y.append(r[i+W-1])
        return np.array(X), np.array(y, np.float32)
    Xtr, ytr = build(tr, sets['tr']); Xtr, ytr = Xtr[::2], ytr[::2]
    Xva, yva = None, None
    Xva, yva = build(tr, sets['va']); Xca, yca = build(tr, sets['ca'])
    Xte, yte = [], []
    for j, u in enumerate(sorted(te.unit.unique())):
        a = te[te.unit == u][sc].values
        if len(a) < W: a = np.vstack([np.repeat(a[:1], W-len(a), 0), a])
        Xte.append(a[-W:]); yte.append(min(rul[j], cap))
    Xte = np.array(Xte); yte = np.array(yte, np.float32)
    Xtr, Xva, Xca, Xte = _mm(Xtr, Xva, Xca, Xte)
    names = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32']
    return dict(task='reg', K=1, cap=cap, names=names, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xca=Xca, yca=yca, Xte=Xte, yte=yte)

def load(name, seed):
    """name in {'ai4i', 'FD001', 'FD003'}; seed controls the split."""
    if name == 'ai4i': return load_ai4i(seed)
    if name in ('FD001','FD003'): return load_cmapss(name, seed)
