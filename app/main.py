"""One free Render service: draw-a-digit website + small neural network API (64 -> 128 -> 10).

  GET  /          the drawing page (static/index.html)
  GET  /health    liveness check
  POST /predict   {"image": "<data URL or base64 PNG>"} ->
                  {"digit", "probs"[10], "grid"[8][8],            (neural net + the 8x8 the models see, 0..16)
                   "models": {"softmax": {...}, "net": {...}}}    (both models, each with digit, probs and
                                                                   "why": the pixels that pushed toward
                                                                   its answer instead of its runner-up)
  GET  /templates the 10 x 64 softmax weights (each digit's "template" as an 8x8 picture)
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .preprocess import DrawingError, image_to_grid

ROOT = Path(__file__).resolve().parent.parent
P = np.load(ROOT / "model" / "nn_weights.npz")
W1, b1, W2, b2 = P["W1"], P["b1"], P["W2"], P["b2"]  # 64->128 (ReLU) ->10
assert W1.shape == (64, 128) and W2.shape == (128, 10)

S = np.load(ROOT / "model" / "softmax_mine.npy")  # (10, 65): column 0 is beta_0
Ws, bs = S[:, 1:], S[:, 0]
assert S.shape == (10, 65)

# Collect mode: only when you start the server locally with COLLECT=1.
# Render never sets it, so the public site cannot save anything.
COLLECT = os.environ.get("COLLECT") == "1"
MINE_DIR = Path(os.environ.get("MINE_DIR", ROOT / "data" / "mine"))

app = FastAPI(title="Neural net digits")


class Drawing(BaseModel):
    image: str


class LabeledDrawing(BaseModel):
    image: str
    label: int


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict_grid(grid: np.ndarray) -> np.ndarray:
    """8x8 grid (0..16) -> 10 probabilities. Same forward pass as in train_nn.py."""
    x = grid.ravel() / 16.0
    h = np.maximum(0.0, x @ W1 + b1)
    return softmax(h @ W2 + b2)


def softmax_probs(grid: np.ndarray) -> np.ndarray:
    return softmax(Ws @ (grid.ravel() / 16.0) + bs)


def _why(x: np.ndarray, probs: np.ndarray, weight_diff) -> dict:
    """Pixels that pushed toward the top digit instead of the runner-up.

    contribution of a pixel = ink there * (how much that pixel raises the top score minus the runner-up's).
    For softmax this is exact; for the net it is gradient x input (a local approximation).
    """
    order = np.argsort(probs)[::-1]
    a, b = int(order[0]), int(order[1])
    return {"top": a, "other": b, "map": np.round((x * weight_diff(a, b)).reshape(8, 8), 4).tolist()}


def explain_softmax(grid: np.ndarray) -> dict:
    x = grid.ravel() / 16.0
    p = softmax(Ws @ x + bs)
    return {"digit": int(p.argmax()), "probs": [round(float(v), 4) for v in p],
            "why": _why(x, p, lambda a, b: Ws[a] - Ws[b])}


def explain_net(grid: np.ndarray) -> dict:
    x = grid.ravel() / 16.0
    pre = x @ W1 + b1
    p = softmax(np.maximum(0.0, pre) @ W2 + b2)
    mask = (pre > 0).astype(np.float64)
    return {"digit": int(p.argmax()), "probs": [round(float(v), 4) for v in p],
            "why": _why(x, p, lambda a, b: W1 @ (mask * (W2[:, a] - W2[:, b])))}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(drawing: Drawing) -> dict:
    try:
        grid = image_to_grid(drawing.image)
    except DrawingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    probs = predict_grid(grid)
    return {
        "digit": int(probs.argmax()),
        "probs": [round(float(p), 4) for p in probs],
        "grid": np.round(grid, 1).tolist(),
        "models": {"softmax": explain_softmax(grid), "net": explain_net(grid)},
    }


@app.get("/templates")
def templates() -> dict:
    """Softmax weights per digit as 8x8 pictures (positive: ink here votes FOR the digit)."""
    return {"templates": np.round(Ws.reshape(10, 8, 8), 3).tolist()}


def counts() -> list[int]:
    return [len(list((MINE_DIR / str(d)).glob("*.png"))) for d in range(10)]


@app.get("/config")
def config() -> dict:
    return {"collect": COLLECT, "counts": counts() if COLLECT else None}


@app.post("/collect")
def collect(item: LabeledDrawing) -> dict:
    """Save the RAW canvas picture (not the 8x8 grid) under data/mine/<digit>/."""
    if not COLLECT:
        raise HTTPException(status_code=404, detail="Collect mode is off.")
    if not 0 <= item.label <= 9:
        raise HTTPException(status_code=422, detail="Label must be 0..9.")
    try:
        image_to_grid(item.image)  # refuse empty / unreadable drawings
    except DrawingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raw = base64.b64decode(item.image.split(",", 1)[-1])
    folder = MINE_DIR / str(item.label)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{time.time_ns() // 1_000_000}.png").write_bytes(raw)
    return {"saved": item.label, "counts": counts()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
