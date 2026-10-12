"""Sanity checks for the drawing -> 8x8 pipeline (not needed on Render).

  python scripts/check_pipeline.py

1. Fake "canvas drawings": digits drawn with a system font, thick white strokes on black,
   at random size / position, sent through the same code the server uses.
2. Dataset images upscaled to canvas size (the easy case).
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.main import predict_grid  # noqa: E402
from app.preprocess import image_to_grid  # noqa: E402

rng = np.random.default_rng(0)
FONTS = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoepr.ttf", r"C:\Windows\Fonts\comic.ttf"]
FONTS = [f for f in FONTS if Path(f).exists()]


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def font_digit(d: int) -> bytes:
    img = Image.new("RGB", (280, 280), "black")
    size = int(rng.integers(110, 230))
    font = ImageFont.truetype(FONTS[int(rng.integers(len(FONTS)))], size)
    x, y = rng.integers(0, 280 - size // 2), rng.integers(0, 280 - size)
    ImageDraw.Draw(img).text((x, y), str(d), font=font, fill="white", stroke_width=int(rng.integers(3, 8)), stroke_fill="white")
    return png_bytes(img)


if FONTS:
    ok = n = 0
    for d in range(10):
        for _ in range(10):
            try:
                pred = predict_grid(image_to_grid(font_digit(d))).argmax()
            except Exception:
                continue
            ok += pred == d
            n += 1
    print(f"font-drawn digits: {ok}/{n} = {ok / n:.0%}")

from sklearn.datasets import load_digits  # noqa: E402

D = load_digits()
idx = rng.choice(len(D.target), 300, replace=False)
ok = 0
for i in idx:
    im = Image.fromarray((D.images[i] / 16 * 255).astype(np.uint8)).resize((280, 280), Image.BICUBIC)
    ok += predict_grid(image_to_grid(png_bytes(im))).argmax() == D.target[i]
print(f"dataset images upscaled: {ok}/300 = {ok / 300:.0%}")
