"""Regenerate the three paper figures from the seed 0 checkpoints.

  python scripts/make_figures.py
"""
import pickle, numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _paths import FIGS, CKPT
from nff import models as Mm, data as Dm
from explain_rules import mf_fig

def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(12, 5.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis('off')
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=c, ec='#333', lw=1.1))
        ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=9.2, wrap=True)
    def arr(x1, y1, x2, y2, t=None):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=1.3, color='#333'))
        if t: ax.text((x1+x2)/2, (y1+y2)/2+0.12, t, fontsize=8, ha='center', color='#444')
    box(0.1, 2.0, 1.6, 1.2, 'Sensor window\n$X \\in R^{W \\times J}$\n(asset digital twin\nstate stream)', '#e8eef7')
    box(2.1, 2.0, 1.8, 1.2, 'Adaptive fuzzy\ntokenizer\nordered Gaussian\nLOW / MED / HIGH', '#fdf1dc')
    box(4.4, 3.2, 2.2, 1.3, 'Transformer encoder\n(variable tokens + CLS)\nmulti-head self attention', '#e6f4ea')
    box(4.4, 0.6, 2.2, 1.3, 'Evolving fuzzy rule base\nTakagi Sugeno rules\n(masked slots, grow / prune)', '#f6e4ee')
    box(7.1, 3.2, 1.9, 1.3, 'Residual head\n$\\gamma\\, g(h)$\n(penalised, gated)', '#e6f4ea')
    box(7.1, 0.6, 1.9, 1.3, 'Rule output\n$\\sum_r \\bar{\\phi}_r c_r$\n+ linguistic trace', '#f6e4ee')
    box(9.5, 2.0, 2.3, 1.2, 'Prediction $\\hat{y}$\nfailure class / RUL\n+ conformal set or\ninterval $\\Gamma_\\alpha(X)$', '#eceaf6')
    arr(1.7, 2.6, 2.1, 2.6)
    arr(3.9, 2.85, 4.4, 3.8, 'tokens')
    arr(3.9, 2.35, 4.4, 1.3); ax.text(3.55, 1.45, 'window tail\nmemberships', fontsize=8, ha='center', color='#444')
    arr(5.5, 3.2, 5.5, 1.9, 'attention relevance $\\hat{s}_j$')
    arr(6.6, 3.85, 7.1, 3.85, 'h'); arr(6.6, 1.25, 7.1, 1.25)
    arr(9.0, 3.85, 9.5, 2.9); arr(9.0, 1.25, 9.5, 2.3)
    ax.text(5.5, 0.15, 'Training loop: every 4 epochs prune low utility rules and add rules at poorly covered, high loss samples', ha='center', fontsize=8.5, style='italic')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def fig_rul(path):
    k = pickle.load(open(CKPT / 'enft_FD001_s0.pkl', 'rb')); p, mask, cfg = k['_params'], jnp.asarray(k['_mask']), k['_cfg']
    D = Dm.load('FD001', 0)
    f = lambda X: np.clip(np.asarray(Mm.forward(p, mask, jnp.asarray(X), cfg)[0])[:, 0] * 125, 0, 125)
    yc = f(D['Xca']); rc = np.abs(yc - D['yca']); n = len(rc); q = np.quantile(rc, np.ceil((n + 1) * 0.9) / n, method='higher')
    yt = f(D['Xte']); o = np.argsort(D['yte'])
    fig, ax = plt.subplots(figsize=(7.5, 3.4)); x = np.arange(100)
    ax.fill_between(x, yt[o] - q, yt[o] + q, color='#9ecae1', alpha=0.6, label=f'90% conformal interval (±{q:.1f} cycles)')
    ax.plot(x, D['yte'][o], 'k-', lw=1.6, label='True RUL'); ax.plot(x, yt[o], 'o', ms=3, color='#d62728', label='Predicted RUL')
    ax.set_xlabel('Test engine (sorted by true RUL)'); ax.set_ylabel('RUL (cycles)'); ax.legend(fontsize=8, loc='upper left'); ax.set_ylim(-20, 175)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

if __name__ == '__main__':
    fig_architecture(FIGS / 'Fig1_architecture.png')
    mf_fig('ai4i', [2, 3, 4], FIGS / 'Fig2_membership_ai4i.png')
    fig_rul(FIGS / 'Fig3_rul_fd001.png')
    print('figures written to', FIGS)
