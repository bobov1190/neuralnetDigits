"""Fake "hand-drawn" training data for the digit recognisers.

Everything here is pushed through the SAME app/preprocess.py as the website, so the network
learns from exactly the kind of 8x8 grids the site will send it:

  1. dataset images blown up to a 280x280 "canvas", randomly rotated / sheared / resized / moved,
     with random brush thickness
  2. digits drawn with many system fonts (random size, thickness, tilt)

Used by train_nn.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.preprocess import DrawingError, to_vector

AUG_PER_IMAGE = 20
FONT_DIR = Path(r"C:\Windows\Fonts")
TRAIN_FONTS = ["arial", "times", "cour", "tahoma", "segoeui", "segoepr", "segoesc", "Inkfree",
               "georgia", "trebuc", "consola", "BRADHITC", "LHANDW", "ariblk", "framd"]
HELD_OUT_FONTS = ["comic", "calibri", "verdana"]  # never used for training

rng = np.random.default_rng(0)


# ---------- making fake "canvas drawings" ----------------------------------------------------
def affine(img: Image.Image, angle: float, shear: float) -> Image.Image:
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
    w, h = img.size
    return img.transform(img.size, Image.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0), Image.BILINEAR)


def paste_random(img: Image.Image) -> np.ndarray:
    canvas = Image.new("L", (280, 280), 0)
    w, h = img.size
    x = int(rng.integers(0, max(1, 280 - w)))
    y = int(rng.integers(0, max(1, 280 - h)))
    canvas.paste(img, (x, y))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def canvas_from_grid(grid: np.ndarray) -> np.ndarray:
    """8x8 dataset image -> a messy 280x280 'drawing' (white ink on black)."""
    im = Image.fromarray((grid / 16 * 255).astype(np.uint8)).resize((200, 200), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255
    a = np.clip((a - rng.uniform(0.25, 0.55)) * rng.uniform(4, 10) + 0.5, 0, 1)  # soft threshold
    im = Image.fromarray((a * 255).astype(np.uint8))
    k = int(rng.choice([1, 3, 3, 5, 7, 9]))  # brush thickness
    if k > 1:
        im = im.filter(ImageFilter.MaxFilter(k))
    side = int(200 * rng.uniform(0.45, 1.0))
    im = im.resize((side, side))
    im = affine(im, rng.uniform(-15, 15), rng.uniform(-0.25, 0.25))
    return paste_random(im)


def canvas_from_font(font_path: Path, digit: int) -> np.ndarray:
    size = int(rng.integers(100, 240))
    font = ImageFont.truetype(str(font_path), size)
    img = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(img).text((size, size), str(digit), font=font, fill=255, anchor="mm",
                             stroke_width=int(rng.integers(2, 9)), stroke_fill=255)
    img = affine(img, rng.uniform(-12, 12), rng.uniform(-0.2, 0.2))
    bbox = img.getbbox()
    img = img.crop(bbox)
    img.thumbnail((270, 270))
    return paste_random(img)


def to_grid(ink: np.ndarray) -> np.ndarray | None:
    try:
        return to_vector(ink).ravel()
    except DrawingError:
        return None





def fonts(names: list[str]) -> list[Path]:
    found = []
    for n in names:
        for ext in (".ttf", ".TTF"):
            if (FONT_DIR / f"{n}{ext}").exists():
                found.append(FONT_DIR / f"{n}{ext}")
                break
    return found


def font_set(paths: list[Path], per_digit: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for p in paths:
        for d in range(10):
            for _ in range(per_digit):
                v = to_grid(canvas_from_font(p, d))
                if v is not None:
                    X.append(v)
                    y.append(d)
    return np.array(X), np.array(y)
