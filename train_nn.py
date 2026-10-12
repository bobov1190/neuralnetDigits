"""Train a small neural network (64 -> 128 -> 10, ReLU) in plain numpy.

Same data recipe as the softmax model (see data_aug.py), so the comparison is fair:
only the model changes. The softmax baseline lives in model/softmax_baseline.npy.

  pip install scikit-learn pillow
  python train_nn.py      (needs Windows fonts, a few minutes)  ->  model/nn_weights.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import data_aug as da

ROOT = Path(__file__).parent
OUT = ROOT / "model" / "nn_weights.npz"
BASELINE = ROOT / "model" / "softmax_baseline.npy"

HIDDEN, EPOCHS, BATCH, LR, WD = 128, 80, 256, 2e-3, 1e-4
rng = np.random.default_rng(1)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(p: dict, X: np.ndarray):
    h = np.maximum(0.0, X @ p["W1"] + p["b1"])
    return h, softmax(h @ p["W2"] + p["b2"])


def predict(p: dict, X16: np.ndarray) -> np.ndarray:
    return forward(p, X16 / 16.0)[1].argmax(1)


def softmax_acc(W: np.ndarray, X16: np.ndarray, y: np.ndarray) -> float:
    A = np.hstack([np.ones((len(X16), 1)), X16 / 16.0])
    return float((softmax(A @ W.T).argmax(1) == y).mean())


def main() -> None:
    digits = load_digits()
    X8, y = digits.data, digits.target
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

    tr_fonts, ho_fonts = da.fonts(da.TRAIN_FONTS), da.fonts(da.HELD_OUT_FONTS)
    print(f"train fonts: {len(tr_fonts)}   held-out fonts: {[p.stem for p in ho_fonts]}")

    # ----- training data -----
    Xs, ys = [X8[idx_tr]], [y[idx_tr]]
    aug_X, aug_y = [], []
    for i in idx_tr:
        for _ in range(da.AUG_PER_IMAGE):
            v = da.to_grid(da.canvas_from_grid(digits.images[i]))
            if v is not None:
                aug_X.append(v)
                aug_y.append(y[i])
    Xs.append(np.array(aug_X)); ys.append(np.array(aug_y))
    fx, fy = da.font_set(tr_fonts, per_digit=40)
    Xs.append(fx); ys.append(fy)
    X_tr, y_tr = np.vstack(Xs), np.concatenate(ys)
    print(f"training samples: {len(y_tr)}  (plain {len(idx_tr)}, augmented {len(aug_y)}, fonts {len(fy)})")

    # ----- evaluation data (never trained on) -----
    te_clean, te_clean_y = X8[idx_te], y[idx_te]
    cx, cy = [], []
    for i in idx_te:
        v = da.to_grid(da.canvas_from_grid(digits.images[i]))
        if v is not None:
            cx.append(v); cy.append(y[i])
    te_canvas, te_canvas_y = np.array(cx), np.array(cy)
    te_fx, te_fy = da.font_set(ho_fonts, per_digit=60)

    # ----- training: Adam, mini-batches, cosine learning-rate decay -----
    X = X_tr / 16.0
    Y = np.eye(10)[y_tr]
    p = {
        "W1": rng.normal(0, np.sqrt(2 / 64), (64, HIDDEN)), "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, 10)), "b2": np.zeros(10),
    }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    t = 0
    for epoch in range(EPOCHS):
        lr = LR * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
        order = rng.permutation(len(X))
        for s in range(0, len(X), BATCH):
            b = order[s : s + BATCH]
            xb, yb = X[b], Y[b]
            h, P = forward(p, xb)
            d2 = (P - yb) / len(b)
            d1 = (d2 @ p["W2"].T) * (h > 0)
            g = {"W2": h.T @ d2 + WD * p["W2"], "b2": d2.sum(0),
                 "W1": xb.T @ d1 + WD * p["W1"], "b1": d1.sum(0)}
            t += 1
            for k in p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                p[k] -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v2[k] / (1 - 0.999**t)) + 1e-8)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            loss = -np.log(forward(p, X)[1][np.arange(len(y_tr)), y_tr] + 1e-12).mean()
            acc = (predict(p, te_canvas) == te_canvas_y).mean()
            print(f"epoch {epoch:3d}  loss {loss:.4f}  canvas-test acc {acc:.1%}")

    # ----- comparison -----
    base = np.load(BASELINE)
    print("\naccuracy               softmax   neural net")
    for name, Xe, ye in [("dataset test 8x8", te_clean, te_clean_y),
                         ("test as canvas", te_canvas, te_canvas_y),
                         ("held-out fonts", te_fx, te_fy)]:
        print(f"{name:20s}  {softmax_acc(base, Xe, ye):8.1%}  {(predict(p, Xe) == ye).mean():11.1%}")

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, **p)
    n_params = sum(v.size for v in p.values())
    print(f"\nsaved {OUT}  ({n_params} parameters)")


if __name__ == "__main__":
    main()
