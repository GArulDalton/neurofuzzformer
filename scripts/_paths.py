import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS, FIGS, CKPT = ROOT / 'results', ROOT / 'figures', ROOT / 'checkpoints'
for p in (RESULTS, FIGS, CKPT): p.mkdir(exist_ok=True)
