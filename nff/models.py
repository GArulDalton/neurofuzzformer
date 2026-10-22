"""JAX implementation of NeuroFuzzFormer and the neural baselines.

Components: adaptive fuzzy tokenizer with ordered Gaussian memberships (Eqs. 2-3),
transformer encoder with CLS-token sensor relevance (Eqs. 5-6), attention modulated
Takagi-Sugeno rule layer with slot masks (Eqs. 7-8), gated residual head.
"""
import jax, jax.numpy as jnp, numpy as np
M_TERMS = 3
def dense(k, i, o):
    return {'w': jax.random.normal(k, (i, o)) * np.sqrt(2.0/(i+o)), 'b': jnp.zeros(o)}
def ap(p, x): return x @ p['w'] + p['b']
def ln(x, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True); return (x-m)/jnp.sqrt(v+eps)

# ---------- fuzzy tokenizer ----------
def init_fuzzy(k, J):
    c0 = jnp.full((J,), 0.15); inc = jnp.full((J, M_TERMS-1), np.log(np.expm1(0.35)))
    return {'c0': c0, 'inc': inc, 'sig': jnp.full((J, M_TERMS), np.log(np.expm1(0.15)))}
def centers(fp):
    c = jnp.concatenate([fp['c0'][:, None], fp['c0'][:, None] + jnp.cumsum(jax.nn.softplus(fp['inc']), -1)], -1)
    return c, jax.nn.softplus(fp['sig']) + 0.02
def fuzzify(fp, x):            # x (B,W,J) -> (B,W,J,M), normalized memberships
    c, s = centers(fp)
    u = jnp.exp(-0.5 * ((x[..., None] - c) / s) ** 2)
    return u / (u.sum(-1, keepdims=True) + 1e-6)

# ---------- transformer ----------
def init_tf(k, J, tok_in, d, L, K):
    ks = jax.random.split(k, 4 + 4*L)
    p = {'tok': dense(ks[0], tok_in, d), 'vemb': jax.random.normal(ks[1], (J, d))*0.02,
         'cls': jax.random.normal(ks[2], (1, d))*0.02, 'head': dense(ks[3], d, K), 'layers': []}
    for l in range(L):
        a, b, c, e = ks[4+4*l:8+4*l]
        p['layers'].append({'qkv': dense(a, d, 3*d), 'o': dense(b, d, d), 'f1': dense(c, d, 2*d), 'f2': dense(e, 2*d, d)})
    return p
def run_tf(p, tok_feats, H=4):
    B, J, _ = tok_feats.shape
    z = ap(p['tok'], tok_feats) + p['vemb']
    z = jnp.concatenate([jnp.broadcast_to(p['cls'], (B, 1, z.shape[-1])), z], 1)
    d = z.shape[-1]; dh = d // H; att = None
    for lp in p['layers']:
        q, k, v = jnp.split(ap(lp['qkv'], ln(z)), 3, -1)
        sh = lambda t: t.reshape(B, J+1, H, dh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh), -1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, J+1, d)
        z = z + ap(lp['o'], o)
        z = z + ap(lp['f2'], jax.nn.gelu(ap(lp['f1'], ln(z))))
    h = ln(z[:, 0]); s = att[:, :, 0, 1:].mean(1)       # CLS->variable attention, (B,J)
    return h, s

# ---------- evolving fuzzy rule layer ----------
def init_rules(R, J, K):
    return {'P': jnp.zeros((R, J, M_TERMS)), 'A': jnp.zeros((R, J)), 'C': jnp.zeros((R, K))}
def rule_forward(rp, mask, ubar, s):
    # ubar (B,J,M); s (B,J) variable relevance (sums to J) ; mask (R,)
    P = jax.nn.softmax(rp['P'], -1); a = jax.nn.softplus(rp['A'])
    d2 = ((ubar[:, None] - P[None]) ** 2).sum(-1)              # (B,R,J)
    dist = (a[None] * s[:, None, :] * d2).sum(-1)               # (B,R)
    logw = -dist + jnp.log(mask + 1e-12)[None]
    wbar = jax.nn.softmax(logw, -1)
    cover = jnp.exp(jnp.max(logw, -1))
    return wbar @ rp['C'], wbar, cover

# ---------- full model family ----------
# cfg flags: fuzzy_tok, use_tf, use_rules, attn_mod, residual
def init_model(key, cfg, J, W, K):
    ks = jax.random.split(key, 5); p = {}
    if cfg['use_tf']:
        tok_in = W*(M_TERMS+1) if cfg['fuzzy_tok'] else W
        p['tf'] = init_tf(ks[0], J, tok_in, cfg['d'], cfg['L'], K)
    if cfg['use_rules'] or cfg['fuzzy_tok']: p['fz'] = init_fuzzy(ks[1], J)
    if cfg['use_rules']: p['rules'] = init_rules(cfg['Rmax'], J, K); p['gate'] = jnp.array(-1.0)
    return p

def forward(p, mask, x, cfg):
    B, W, J = x.shape; out = {}
    u = fuzzify(p['fz'], x) if 'fz' in p else None
    if cfg['use_tf']:
        tf_in = jnp.concatenate([u, x[..., None]], -1).transpose(0, 2, 1, 3).reshape(B, J, -1) if cfg['fuzzy_tok'] else x.transpose(0, 2, 1)
        h, s = run_tf(p['tf'], tf_in); res = ap(p['tf']['head'], h)
    else:
        s = jnp.ones((B, J)) / J; res = None
    if not cfg['use_rules']:
        return res, out
    ubar = u[:, -min(W, 5):].mean(1)
    sj = s * J if (cfg['use_tf'] and cfg['attn_mod']) else jnp.ones((B, J))
    ro, wbar, cover = rule_forward(p['rules'], mask, ubar, sj)
    out.update(rule_out=ro, wbar=wbar, cover=cover, s=s, ubar=ubar)
    if cfg['use_tf'] and cfg['residual']:
        g = jax.nn.sigmoid(p['gate']); out['res'] = g * res
        return ro + g * res, out
    return ro, out

# ---------- baselines: MLP / LSTM ----------
def init_mlp(k, inp, K, hid=128):
    a, b, c = jax.random.split(k, 3)
    return {'l1': dense(a, inp, hid), 'l2': dense(b, hid, hid), 'o': dense(c, hid, K)}
def run_mlp(p, x):
    z = x.reshape(x.shape[0], -1)
    z = jax.nn.relu(ap(p['l1'], z)); z = jax.nn.relu(ap(p['l2'], z)); return ap(p['o'], z)
def init_lstm(k, J, K, hid=64):
    a, b = jax.random.split(k)
    return {'g': dense(a, J+hid, 4*hid), 'o': dense(b, hid, K)}
def run_lstm(p, x):
    B = x.shape[0]; hid = p['o']['w'].shape[0]
    def step(c, xt):
        h, cc = c; z = ap(p['g'], jnp.concatenate([xt, h], -1)); i, f, g, o = jnp.split(z, 4, -1)
        cc = jax.nn.sigmoid(f+1.0)*cc + jax.nn.sigmoid(i)*jnp.tanh(g); h = jax.nn.sigmoid(o)*jnp.tanh(cc); return (h, cc), None
    (h, _), _ = jax.lax.scan(step, (jnp.zeros((B, hid)), jnp.zeros((B, hid))), x.transpose(1, 0, 2))
    return ap(p['o'], h)
