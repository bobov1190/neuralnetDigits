"""Turn a drawing from a canvas into the 64-number vector the model expects.

The digits dataset was made from NIST bitmaps like this (we measured it):
  * every digit is stretched to fill the FULL height of the frame (8 of 8 rows),
  * the width keeps its aspect ratio (about 6 of 8 columns) and is centred,
  * the 8x8 values are "how much of this block is ink", scaled to 0..16,
  * strokes are thin: roughly 2.75 px on a 32 px frame.

A raw canvas drawing looks nothing like that (small, off-centre, thick lines),
so we imitate the same normalisation step by step:

  1. read the picture as "ink" brightness (white strokes on black)
  2. crop to the bounding box of the ink
  3. scale it so the height fills the frame (the width is capped to 3/4)
  4. measure stroke thickness and thin / thicken it to the dataset's thickness
  5. put it on a square frame, centred horizontally by its centre of mass
  6. shrink 64x64 -> 8x8 by averaging blocks (= "ink coverage"), scale to 0..16
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageFilter

FRAME = 64            # working resolution (8x8 blocks of 8x8 pixels)
MAX_W = 56            # digit width cap = 7/8 of the frame (tuned on test drawings)
STROKE = 5.5          # target stroke width in frame pixels (~2.75 px at 32x32)
INK_THRESHOLD = 0.2   # brightness above this counts as ink
MAX_SIDE = 1600       # protect the free tier from huge uploads


class DrawingError(ValueError):
    """The drawing can't be converted (empty, too small, unreadable)."""


def decode_image(data: str | bytes) -> np.ndarray:
    """Data-URL / base64 / raw PNG bytes -> 2D float array 'ink' in 0..1."""
    if isinstance(data, str):
        if "," in data[:100]:
            data = data.split(",", 1)[1]
        try:
            data = base64.b64decode(data, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise DrawingError("Could not read the image data.") from exc
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise DrawingError("The file is not a valid image.") from exc

    img = img.convert("RGBA")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # white strokes on black (opaque) or on transparent: both give ink = brightness * alpha
    return arr[..., :3].mean(axis=-1) * arr[..., 3]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _crop_and_scale(ink: np.ndarray) -> np.ndarray:
    """Crop to the ink and scale: height -> FRAME, but width <= MAX_W."""
    r0, r1, c0, c1 = _bbox(ink > INK_THRESHOLD)
    crop = ink[r0:r1, c0:c1]
    h, w = crop.shape
    scale = min(FRAME / h, MAX_W / w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _stroke_width(g: np.ndarray) -> float:
    """Estimate pen width: for a thick line, width ~ 2 * area / perimeter."""
    mask = (g > 0.5).astype(np.uint8) * 255
    area = float((mask > 0).sum())
    if area == 0:
        return STROKE
    eroded = np.asarray(Image.fromarray(mask).filter(ImageFilter.MinFilter(3)))
    boundary = area - float((eroded > 0).sum())
    return 2.0 * area / boundary if boundary > 0 else STROKE


def _normalise_stroke(g: np.ndarray) -> np.ndarray:
    """Thin fat strokes / thicken hairlines to the dataset's thickness."""
    width = _stroke_width(g)
    img = Image.fromarray((g * 255).astype(np.uint8))
    if width > STROKE + 1.0:
        radius = max(1, round((width - STROKE) / 2))
        img = img.filter(ImageFilter.MinFilter(2 * radius + 1))
    elif width < STROKE - 1.5:
        radius = max(1, round((STROKE - width) / 2))
        img = img.filter(ImageFilter.MaxFilter(2 * radius + 1))
    else:
        return g
    return np.asarray(img, dtype=np.float32) / 255.0


def _place_on_frame(g: np.ndarray) -> np.ndarray:
    """Centre horizontally by centre of mass, vertically by bounding box."""
    h, w = g.shape
    com_x = float((g.sum(axis=0) * np.arange(w)).sum() / max(g.sum(), 1e-6))
    left = int(round(FRAME / 2 - com_x))
    left = min(max(left, 0), FRAME - w)
    top = (FRAME - h) // 2
    frame = np.zeros((FRAME, FRAME), dtype=np.float32)
    frame[top : top + h, left : left + w] = g
    return frame


def to_vector(ink: np.ndarray) -> np.ndarray:
    """Ink picture -> 8x8 grid with values 0..16 (same scale as sklearn digits)."""
    mask = ink > INK_THRESHOLD
    if mask.sum() < 20:
        raise DrawingError("The canvas is empty - draw a digit first.")
    r0, r1, c0, c1 = _bbox(mask)
    if max(r1 - r0, c1 - c0) < 0.1 * max(ink.shape):
        raise DrawingError("The drawing is too small - draw the digit bigger.")

    g = _crop_and_scale(ink)
    g = _normalise_stroke(g)
    # thinning can shave the edges: crop + scale once more so the digit fills the frame
    g = _crop_and_scale(np.pad(g, 2))
    frame = _place_on_frame(g)

    grid = frame.reshape(8, FRAME // 8, 8, FRAME // 8).mean(axis=(1, 3))
    return np.clip(grid * 16.0, 0.0, 16.0)


def image_to_grid(data: str | bytes) -> np.ndarray:
    return to_vector(decode_image(data))
