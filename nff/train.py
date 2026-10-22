"""Training, evaluation, conformal calibration and latency measurement.

Model names (paper name in brackets):
  ENFT            NeuroFuzzFormer, default configuration (fuzzy augmented tokens)
  ENFT-rawTokens  NeuroFuzzFormer with raw value tokens (validation selected on C-MAPSS)
  ENFT-noEvolve, ENFT-noAttnMod, ENFT-noResidual   ablations
  Transformer, NeuralTSK, MLP, LSTM, HGB           baselines
"""
import jax, jax.numpy as jnp, numpy as np, optax, time, json, sys
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from . import models as Mm, data as Dm

BASE = dict(d=32, L=2, Rmax=32, R0=8, fuzzy_tok=True, use_tf=True, use_rules=True, attn_mod=True, residual=True,
            evolve=True, lam_res=0.01, lam_a=1e-3)
VARIANTS = {
 'ENFT': {},
 'ENFT-noEvolve': dict(evolve=False, R0=16),
 'ENFT-noAttnMod': dict(attn_mod=False),
 'ENFT-noResidual': dict(residual=False),
 'ENFT-rawTokens': dict(fuzzy_tok=False),
 'Transformer': dict(use_rules=False, fuzzy_tok=False),
 'NeuralTSK': dict(use_tf=False, fuzzy_tok=False, evolve=False, R0=16),
}

def batches(n, bs, rng):
    idx = rng.permutation(n); return [idx[i:i+bs] for i in range(0, n, bs)]

def predict(fn, params, X, bs=2048):
    outs = [np.asarray(fn(params, X[i:i+bs])) for i in range(0, len(X), bs)]
    return np.concatenate(outs)

def tune_thr(y, out):
    pr = np.asarray(jax.nn.softmax(np.asarray(out), -1))[:, 1]
    ts = np.linspace(0.05, 0.95, 91); f = [f1_score(y, pr >= t) for t in ts]; return float(ts[int(np.argmax(f))])

def metrics(task, y, out, cap=125, thr=0.5):
    if task == 'clf':
        pr = np.asarray(jax.nn.softmax(out, -1)); yh = pr.argmax(1); K = pr.shape[1]
        if K == 2: yh = (pr[:, 1] >= thr).astype(int)
        m = dict(acc=accuracy_score(y, yh), bacc=balanced_accuracy_score(y, yh), macroF1=f1_score(y, yh, average='macro'))
        if K == 2:
            m.update(F1=f1_score(y, yh), AUROC=roc_auc_score(y, pr[:, 1]), AUPRC=average_precision_score(y, pr[:, 1]))
        return m
    yp = np.clip(out.reshape(-1)*cap, 0, cap); e = yp - y
    score = np.sum(np.where(e < 0, np.exp(-e/13)-1, np.exp(e/10)-1))
    return dict(RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))), Score=float(score))

def conformal(task, out_ca, y_ca, out_te, y_te, alpha=0.1, cap=125):
    if task == 'clf':
        pc = np.asarray(jax.nn.softmax(out_ca, -1)); pt = np.asarray(jax.nn.softmax(out_te, -1))
        sc = 1 - pc[np.arange(len(y_ca)), y_ca]; n = len(sc)
        q = np.quantile(sc, min(1, np.ceil((n+1)*(1-alpha))/n), method='higher')
        sets = (1 - pt) <= q
        return dict(cov=float(sets[np.arange(len(y_te)), y_te].mean()), size=float(sets.sum(1).mean()),
                    singleton=float((sets.sum(1) == 1).mean()))
    rc = np.abs(np.clip(out_ca.reshape(-1)*cap, 0, cap) - y_ca); n = len(rc)
    q = np.quantile(rc, min(1, np.ceil((n+1)*(1-alpha))/n), method='higher')
    yp = np.clip(out_te.reshape(-1)*cap, 0, cap)
    return dict(cov=float((np.abs(yp-y_te) <= q).mean()), width=float(2*q))

def run(dname, mname, seed, epochs=None, log=print):
    D = Dm.load(dname, seed); task, K = D['task'], D['K']; cap = D.get('cap', 125)
    Xtr, ytr = D['Xtr'], D['ytr']; N, W, J = Xtr.shape
    t0 = time.time()
    if mname == 'HGB':
        f = lambda X: X.reshape(len(X), -1)
        if task == 'clf':
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight='balanced' if K == 2 else None, random_state=seed,
                                               early_stopping=True)
            m.fit(f(Xtr), ytr); lg = lambda X: np.log(m.predict_proba(f(X)) + 1e-9)
        else:
            m = HistGradientBoostingRegressor(max_iter=500, random_state=seed, early_stopping=True); m.fit(f(Xtr), ytr/cap)
            lg = lambda X: m.predict(f(X))[:, None]
        ott, oca = lg(D['Xte']), lg(D['Xca']); thr = tune_thr(D['yva'], lg(D['Xva'])) if task == 'clf' and K == 2 else 0.5
        t1 = time.perf_counter()
        for i in range(200): lg(D['Xte'][i % len(D['Xte'])][None])
        lat = (time.perf_counter()-t1)/200*1e3
        res = dict(test=metrics(task, D['yte'], ott, cap, thr), conf=conformal(task, oca, D['yca'], ott, D['yte'], cap=cap),
                   params=None, latency_ms=lat, train_s=time.time()-t0)
        return res
    cfg = dict(BASE); cfg.update(VARIANTS.get(mname, {}))
    key = jax.random.PRNGKey(seed); rng = np.random.RandomState(seed)
    if mname == 'MLP': params = Mm.init_mlp(key, W*J, K); mask = jnp.ones(1)
    elif mname == 'LSTM': params = Mm.init_lstm(key, J, K); mask = jnp.ones(1)
    else: params = Mm.init_model(key, cfg, J, W, K); mask = np.zeros(cfg['Rmax'], np.float32)
    is_rule = mname not in ('MLP', 'LSTM') and cfg['use_rules']

    def fwd(p, mask, x):
        if mname == 'MLP': return Mm.run_mlp(p, x), {}
        if mname == 'LSTM': return Mm.run_lstm(p, x), {}
        return Mm.forward(p, mask, x, cfg)

    if task == 'clf':
        cw = np.ones(K, np.float32)
        cw = jnp.array(cw)
    def loss_fn(p, mask, x, y):
        out, aux = fwd(p, mask, x)
        if task == 'clf':
            lp = jax.nn.log_softmax(out); per = -lp[jnp.arange(len(y)), y] * cw[y]; L = per.sum()/cw[y].sum()
        else:
            per = (out[:, 0] - y/cap)**2; L = per.mean()
        if 'res' in aux: L = L + cfg['lam_res'] * jnp.mean(aux['res']**2)
        if is_rule: L = L + cfg['lam_a'] * jnp.mean(jax.nn.softplus(p['rules']['A']) * mask[:, None])
        return L, per
    # rule init by k-means on fuzzified window-tail memberships
    if is_rule:
        def ubar_of(p, x): return Mm.fuzzify(p['fz'], x)[:, -min(W, 5):].mean(1)
        ub = np.asarray(ubar_of(params, Xtr[rng.choice(N, min(N, 4000), replace=False)]))
        sub = rng.choice(N, min(N, 4000), replace=False); ub = np.asarray(ubar_of(params, Xtr[sub]))
        km = KMeans(cfg['R0'], n_init=3, random_state=seed).fit(ub.reshape(len(ub), -1))
        P = np.array(params['rules']['P']); C = np.array(params['rules']['C'])
        for r in range(cfg['R0']):
            P[r] = np.log(km.cluster_centers_[r].reshape(J, -1) + 1e-3)
            ys = ytr[sub][km.labels_ == r]
            if task == 'clf': C[r] = 2.0*(np.bincount(ys, minlength=K)/max(len(ys), 1) - 1.0/K)
            else: C[r] = ys.mean()/cap if len(ys) else 0.5
            mask[r] = 1
        params['rules']['P'] = jnp.array(P); params['rules']['C'] = jnp.array(C)
    epochs = epochs or {'ai4i': 60, 'FD001': 30, 'FD003': 30}[dname]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(2e-3, epochs*int(np.ceil(N/256))), weight_decay=5e-4))
    st = opt.init(params)
    @jax.jit
    def step(p, st, mask, x, y):
        (L, _), g = jax.value_and_grad(loss_fn, has_aux=True)(p, mask, x, y)
        up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, L
    out_fn = jax.jit(lambda p, mask, x: fwd(p, mask, x)[0])
    aux_fn = jax.jit(lambda p, mask, x: fwd(p, mask, x))
    per_fn = jax.jit(lambda p, mask, x, y: loss_fn(p, mask, x, y)[1])
    def val_score(p, mask):
        o = predict(lambda pp, x: out_fn(pp, jnp.asarray(mask), x), p, D['Xva']); m = metrics(task, D['yva'], o, cap)
        return (m.get('AUPRC', m.get('macroF1')) if task == 'clf' else -m['RMSE'])
    best, bestp, bestmask, hist = -1e9, None, None, []
    for ep in range(epochs):
        for b in batches(N, 256, rng):
            params, st, L = step(params, st, jnp.asarray(mask), Xtr[b], ytr[b])
        ev = None
        if is_rule and cfg['evolve'] and ep % 4 == 3 and ep < 0.75*epochs:
            sub = rng.choice(N, min(N, 4000), replace=False)
            _, aux = aux_fn(params, jnp.asarray(mask), Xtr[sub]); per = np.asarray(per_fn(params, jnp.asarray(mask), Xtr[sub], ytr[sub]))
            wb = np.asarray(aux['wbar']).mean(0); cov = np.asarray(aux['cover']); ub = np.asarray(aux['ubar'])
            act = np.where(mask > 0)[0]; ra = len(act)
            prune = [r for r in act if wb[r] < 0.15/ra][:max(0, ra-4)]
            for r in prune: mask[r] = 0
            # growth: poorly covered AND poorly predicted samples
            crit = (cov < np.quantile(cov, 0.1)) & (per > np.quantile(per, 0.75))
            cand = np.where(crit)[0]; free = list(np.where(mask == 0)[0]); added = 0
            P = np.array(params['rules']['P']); A = np.array(params['rules']['A']); C = np.array(params['rules']['C'])
            chosen = []
            for i in cand[np.argsort(-per[cand])]:
                if not free or added >= 3: break
                if any(np.sum((ub[i]-ub[j])**2) < 0.5 for j in chosen): continue
                r = free.pop(0); chosen.append(i)
                P[r] = np.log(ub[i] + 1e-3); A[r] = 0.0
                yi = ytr[sub][i]
                C[r] = 2.0*(np.eye(K)[yi] - 1.0/K) if task == 'clf' else yi/cap
                mask[r] = 1; added += 1
            params['rules'].update(P=jnp.array(P), A=jnp.array(A), C=jnp.array(C))
            # reset Adam moments of re-initialised slots
            reset = np.zeros(cfg['Rmax'], bool); reset[[r for r in range(cfg['Rmax']) if mask[r] == 0]] = True
            for r in prune: reset[r] = True
            def zero_slots(t):
                if isinstance(t, dict) and 'rules' in t:
                    t = dict(t); t['rules'] = {k: jnp.where(reset.reshape((-1,)+(1,)*(v.ndim-1)), 0, v) for k, v in t['rules'].items()}
                return t
            st = jax.tree_util.tree_map(lambda s: s, st)
            st = tuple(_zero(s, zero_slots) for s in st)
            ev = dict(pruned=len(prune), added=added, active=int(mask.sum()))
        vs = val_score(params, mask)
        hist.append(dict(ep=ep, loss=float(L), val=float(vs), active=int(mask.sum()) if is_rule else None, **({'ev': ev} if ev else {})))
        if vs > best: best, bestp, bestmask = vs, jax.tree_util.tree_map(lambda a: a, params), mask.copy()
    params, mask = bestp, bestmask; jm = jnp.asarray(mask)
    o_te = predict(lambda p, x: out_fn(p, jm, x), params, D['Xte']); o_ca = predict(lambda p, x: out_fn(p, jm, x), params, D['Xca'])
    thr = tune_thr(D['yva'], predict(lambda p, x: out_fn(p, jm, x), params, D['Xva'])) if task == 'clf' and K == 2 else 0.5
    res = dict(test=metrics(task, D['yte'], o_te, cap, thr), conf=conformal(task, o_ca, D['yca'], o_te, D['yte'], cap=cap),
               train_s=time.time()-t0, hist=hist)
    nparam = sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(params))
    if is_rule:
        nparam -= int((cfg['Rmax'] - mask.sum()) * (J*Mm.M_TERMS + J + K))
        _, aux = aux_fn(params, jm, D['Xte'][:2048])
        ro = np.asarray(aux['rule_out']); full = np.asarray(out_fn(params, jm, D['Xte'][:2048]))
        if task == 'clf': fid = float((ro.argmax(1) == full.argmax(1)).mean())
        else: fid = float(np.mean(np.abs(ro[:, 0]-full[:, 0])*cap <= 10.0)); res['fid_corr'] = float(np.corrcoef(ro[:, 0], full[:, 0])[0, 1])
        res.update(fidelity=fid, rules=int(mask.sum()), gate=float(jax.nn.sigmoid(params['gate'])) if 'gate' in params else None)
        res['_params'] = params; res['_mask'] = mask; res['_cfg'] = cfg
    res['params'] = nparam
    one = jnp.asarray(D['Xte'][:1]); out_fn(params, jm, one).block_until_ready()
    t1 = time.perf_counter()
    for i in range(300): out_fn(params, jm, jnp.asarray(D['Xte'][i % len(D['Xte'])][None])).block_until_ready()
    res['latency_ms'] = (time.perf_counter()-t1)/300*1e3
    return res

def _zero(s, fn):
    if hasattr(s, 'mu'): return s._replace(mu=fn(s.mu), nu=fn(s.nu))
    if isinstance(s, tuple) and not hasattr(s, '_fields'): return tuple(_zero(x, fn) for x in s)
    return s
