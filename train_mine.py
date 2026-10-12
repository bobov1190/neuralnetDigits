"""Train softmax regression AND the neural net on the SAME data, then test both on YOUR drawings.

Why: the old models were trained only on fake drawings (distorted dataset images + fonts), and
every test we had was fake too. Real mouse-drawn digits behave differently. So now:

  1. you collect real drawings:   COLLECT=1 uvicorn app.main:app      (saved to data/mine/<digit>/)
  2. ~30% of YOUR drawings are locked away as a test set (chosen by a hash of the file name,
     so adding more drawings later never moves an old drawing from test into training)
  3. the other ~70% are added to the fake training data (each one also jittered MINE_AUG times)
  4. both models are trained on exactly that data, then compared on the locked-away drawings

  python train_mine.py            full run (a few minutes), saves the new weights
  python train_mine.py --quick    tiny smoke test, saves nothing
"""
from __future__ import annotations

import argparse
import os
import shutil
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import data_aug as da
from app.preprocess import DrawingError, decode_image, to_vector

ROOT = Path(__file__).parent
MODEL = ROOT / "model"
MINE_DIR = Path(os.environ.get("MINE_DIR", ROOT / "data" / "mine"))
CACHE = ROOT / "data" / "synth_cache.npz"
HIDDEN, BATCH = 128, 256
rng = np.random.default_rng(3)


# ---------------------------------------------------------------- models (plain numpy)
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def init(kind: str) -> dict:
    if kind == "softmax":
        return {"W": np.zeros((64, 10)), "b": np.zeros(10)}
    return {"W1": rng.normal(0, np.sqrt(2 / 64), (64, HIDDEN)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, 10)), "b2": np.zeros(10)}


def probs(kind: str, p: dict, X: np.ndarray) -> np.ndarray:
    if kind == "softmax":
        return softmax(X @ p["W"] + p["b"])
    return softmax(np.maximum(0, X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"])


def accuracy(kind: str, p: dict, X16: np.ndarray, y: np.ndarray) -> float:
    return float((probs(kind, p, X16 / 16.0).argmax(1) == y).mean())


def train(kind: str, X16: np.ndarray, y: np.ndarray, epochs: int, lr0: float, wd: float = 1e-4) -> dict:
    X, Y = X16 / 16.0, np.eye(10)[y]
    p = init(kind)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    t, total = 0, epochs * int(np.ceil(len(X) / BATCH))
    for ep in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), BATCH):
            b = order[i:i + BATCH]
            xb, yb = X[b], Y[b]
            if kind == "softmax":
                d = (softmax(xb @ p["W"] + p["b"]) - yb) / len(b)
                g = {"W": xb.T @ d + wd * p["W"], "b": d.sum(0)}
            else:
                h = np.maximum(0, xb @ p["W1"] + p["b1"])
                d2 = (softmax(h @ p["W2"] + p["b2"]) - yb) / len(b)
                d1 = (d2 @ p["W2"].T) * (h > 0)
                g = {"W2": h.T @ d2 + wd * p["W2"], "b2": d2.sum(0),
                     "W1": xb.T @ d1 + wd * p["W1"], "b1": d1.sum(0)}
            t += 1
            lr = lr0 * 0.5 * (1 + np.cos(np.pi * t / total))  # cosine decay
            for k in p:  # Adam
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print(f"  {kind:8s} epoch {ep + 1:3d}/{epochs}  train acc {accuracy(kind, p, X16, y):.1%}")
    return p


# ---------------------------------------------------------------- fake data (cached)
def build_synth(aug_per: int, font_per: int) -> tuple[np.ndarray, np.ndarray]:
    d = load_digits()
    idx_tr, _ = train_test_split(np.arange(len(d.target)), test_size=0.2, random_state=42, stratify=d.target)
    X, y = [d.data[idx_tr]], [d.target[idx_tr]]
    ax, ay = [], []
    for i in idx_tr:
        for _ in range(aug_per):
            v = da.to_grid(da.canvas_from_grid(d.images[i]))
            if v is not None:
                ax.append(v)
                ay.append(d.target[i])
    fx, fy = da.font_set(da.fonts(da.TRAIN_FONTS), per_digit=font_per)
    X += [np.array(ax), fx]
    y += [np.array(ay), fy]
    return np.vstack(X), np.concatenate(y)


def synth_train(quick: bool) -> tuple[np.ndarray, np.ndarray]:
    if quick:
        return build_synth(2, 5)
    if CACHE.exists():
        c = np.load(CACHE)
        return c["X"], c["y"]
    print("building the fake training set (once, then cached) ...")
    X, y = build_synth(da.AUG_PER_IMAGE, 30)  # 30 per digit per font = 4500, as in train_nn.py
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y)
    return X, y


def old_tests(quick: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d = load_digits()
    _, idx = train_test_split(np.arange(len(d.target)), test_size=0.2, random_state=42, stratify=d.target)
    idx = idx[:60] if quick else idx
    cx, cy = [], []
    for i in idx:
        v = da.to_grid(da.canvas_from_grid(d.images[i]))
        if v is not None:
            cx.append(v)
            cy.append(d.target[i])
    fx, fy = da.font_set(da.fonts(da.HELD_OUT_FONTS), per_digit=5 if quick else 60)
    return {"fake: dataset 8x8": (d.data[idx], d.target[idx]),
            "fake: as canvas": (np.array(cx), np.array(cy)),
            "fake: unseen fonts": (fx, fy)}


# ---------------------------------------------------------------- YOUR drawings
def jitter(ink: np.ndarray) -> np.ndarray | None:
    """A slightly different version of a real drawing: small tilt, shear, brush thickness."""
    im = Image.fromarray((ink * 255).astype(np.uint8))
    im = da.affine(im, rng.uniform(-8, 8), rng.uniform(-0.12, 0.12))
    r = rng.random()
    if r < 0.2:
        im = im.filter(ImageFilter.MaxFilter(3))
    elif r < 0.4:
        im = im.filter(ImageFilter.MinFilter(3))
    return da.to_grid(np.asarray(im, dtype=np.float32) / 255.0)


def load_mine() -> tuple[list, list]:
    train_items, test_items = [], []
    for digit in range(10):
        for f in sorted((MINE_DIR / str(digit)).glob("*.png")):
            ink = decode_image(f.read_bytes())
            try:
                grid = to_vector(ink).ravel()
            except DrawingError:
                continue
            is_test = zlib.crc32(f.name.encode()) % 10 < 3
            (test_items if is_test else train_items).append((digit, ink, grid))
    return train_items, test_items


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mine-aug", type=int, default=40, help="jittered copies of each training drawing")
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    epochs, mine_aug = (4, 3) if a.quick else (a.epochs, a.mine_aug)

    tr_items, te_items = load_mine()
    per_digit = [sum(1 for d, _, _ in te_items if d == k) for k in range(10)]
    print(f"your drawings: {len(tr_items)} for training, {len(te_items)} locked away for testing")
    print("  test drawings per digit:", per_digit)
    if len(te_items) < 10 or min(per_digit) == 0:
        raise SystemExit("Not enough drawings: draw more (at least ~10 per digit) so every digit has test drawings.")

    Xs, ys = synth_train(a.quick)
    mx, my = [], []
    for digit, ink, grid in tr_items:
        mx.append(grid)
        my.append(digit)
        for _ in range(mine_aug):
            v = jitter(ink)
            if v is not None:
                mx.append(v)
                my.append(digit)
    X, y = np.vstack([Xs, np.array(mx)]), np.concatenate([ys, np.array(my)])
    print(f"training set: {len(Xs)} fake + {len(mx)} from your drawings = {len(X)}")

    tests = {f"YOUR drawings (n={len(te_items)})": (np.array([g for _, _, g in te_items]),
                                                    np.array([d for d, _, _ in te_items]))}
    tests.update(old_tests(a.quick))

    # old models: trained on fake data only
    base = np.load(MODEL / "softmax_baseline.npy")
    old_nn_path = MODEL / "nn_synthetic.npz"
    if not old_nn_path.exists():
        shutil.copy(MODEL / "nn_weights.npz", old_nn_path)  # keep the fake-data network for comparison
    models = {"softmax (fake only)": ("softmax", {"W": base[:, 1:].T, "b": base[:, 0]}),
              "net (fake only)": ("net", dict(np.load(old_nn_path)))}

    print("training on fake data + your drawings ...")
    sm = train("softmax", X, y, epochs, lr0=1e-2)
    nn = train("net", X, y, epochs, lr0=2e-3)
    models["softmax (+ yours)"] = ("softmax", sm)
    models["net (+ yours)"] = ("net", nn)

    names = list(models)
    lines = ["| test set | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for tname, (Xe, ye) in tests.items():
        row = [f"{accuracy(k, p, Xe, ye):.1%}" for k, p in models.values()]
        lines.append(f"| {tname} | " + " | ".join(row) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    Xe, ye = list(tests.values())[0]
    for name in ("softmax (+ yours)", "net (+ yours)"):
        k, p = models[name]
        wrong = [f"{t}->{q}" for t, q in zip(ye, probs(k, p, Xe / 16.0).argmax(1)) if t != q]
        print(f"{name}: mistakes on your test drawings: {', '.join(wrong) if wrong else 'none'}")

    if a.quick:
        print("\n--quick: nothing saved")
        return
    np.savez(MODEL / "nn_weights.npz", **nn)
    np.save(MODEL / "softmax_mine.npy", np.hstack([sm["b"][:, None], sm["W"].T]))
    (MODEL / "results_mine.md").write_text(
        f"Your drawings: {len(tr_items)} train, {len(te_items)} test (per digit: {per_digit})\n\n{table}\n",
        encoding="utf-8")
    print("\nsaved model/nn_weights.npz (served by the site), model/softmax_mine.npy, model/results_mine.md")


if __name__ == "__main__":
    main()
