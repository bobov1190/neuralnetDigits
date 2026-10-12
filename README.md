# Neural Net Digits

Draw a digit in the browser, the server shrinks it to 8x8 and a small neural network guesses it.
One free Render service: FastAPI + numpy (no sklearn at runtime).

This is the "next step" after the softmax regression version
(https://github.com/bobov1190/softmaxregressionDigits). Same drawing page, same
preprocessing, same training data. Only the model changed.

## The model

```
64 pixels -> 128 hidden units (ReLU) -> 10 numbers -> softmax -> 10 probabilities
```

- 64*128 + 128 + 128*10 + 10 = 9610 parameters (softmax regression had 650)
- the last step is still softmax, so the "probabilities sum to 1" idea from the lesson is the same
- the new part is the hidden layer: 128 small "feature detectors" between the pixels and the answer

## How it was trained (`train_nn.py`)

- plain numpy, no deep-learning library
- data, the same as for the softmax v2 model (`data_aug.py`):
  - the original 8x8 digits (80% of the dataset, the other 20% is never trained on)
  - many distorted "canvas" versions of them (rotation, slant, brush thickness)
  - digits written with 15 fonts
- Adam optimizer, mini-batches of 256, 80 epochs, learning rate 2e-3 with cosine decay, small weight decay
- He initialisation (random weights, scaled by the layer size)
- three fonts (comic, calibri, verdana) are held out and used only for testing

Run it yourself: `pip install scikit-learn pillow numpy`, then `python train_nn.py`
(needs Windows fonts, takes a few minutes). It writes `model/nn_weights.npz`.

## Results (`python scripts/compare.py`)

| test set                          | softmax | neural net |
|-----------------------------------|--------:|-----------:|
| dataset test images 8x8           |   93.6% |      98.1% |
| the same images drawn as a canvas |   90.0% |      96.9% |
| 3 fonts never seen in training    |   96.9% |      99.9% |

The softmax column is `model/softmax_baseline.npy`, trained on exactly the same data.
Fonts-as-drawn-with-a-thick-brush (`scripts/check_pipeline.py`) are harder than clean
font digits: about 85% on the server pipeline.

## Things to remember

- The model is still very small and the dataset is only 8x8, so thin 1s and 7s can still be mixed up.
- Softmax has no "I don't know" answer. A filled or random canvas still gets a confident digit,
  because the probabilities have to sum to 1.

## Run locally

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Deploy on Render (free)

- New -> Web Service -> this repo, plan Free
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`
- Environment: `PYTHON_VERSION = 3.12.8`

Or New -> Blueprint, and `render.yaml` sets everything.
A free service sleeps after 15 minutes without requests; the first request after that is slow.

## API

`POST /predict` with `{"image": "<data URL or base64 PNG>"}` returns
`{"digit": 4, "probs": [10 numbers], "grid": [8][8]}` (grid = what the model sees, 0..16).
