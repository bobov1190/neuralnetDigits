"""Softmax baseline vs neural net on the same three test sets as train_nn.py.

  python scripts/compare.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data_aug as da  # noqa: E402
from train_nn import predict, softmax_acc  # noqa: E402

p = dict(np.load(ROOT / "model" / "nn_weights.npz"))
base = np.load(ROOT / "model" / "softmax_baseline.npy")

d = load_digits()
X8, y = d.data, d.target
_, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

cx, cy = [], []
for i in idx_te:
    v = da.to_grid(da.canvas_from_grid(d.images[i]))
    if v is not None:
        cx.append(v)
        cy.append(y[i])
fx, fy = da.font_set(da.fonts(da.HELD_OUT_FONTS), per_digit=60)

print("accuracy               softmax   neural net")
for name, Xe, ye in [("dataset test 8x8", X8[idx_te], y[idx_te]),
                     ("test as canvas", np.array(cx), np.array(cy)),
                     ("held-out fonts", fx, fy)]:
    print(f"{name:20s}  {softmax_acc(base, Xe, ye):8.1%}  {(predict(p, Xe) == ye).mean():11.1%}")
