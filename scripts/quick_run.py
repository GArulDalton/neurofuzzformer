"""Train and evaluate a single model on one seed.  Example: python scripts/quick_run.py ai4i ENFT 0 --epochs 60"""
import argparse, json
from _paths import ROOT
from nff import train
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('model'); ap.add_argument('seed', type=int); ap.add_argument('--epochs', type=int)
a = ap.parse_args()
r = train.run(a.data, a.model, a.seed, epochs=a.epochs)
print(json.dumps({k: v for k, v in r.items() if not k.startswith('_') and k != 'hist'}, indent=1, default=float))
