# Neural Net Digits

Draw a digit in the browser, the server shrinks it to 8x8 and a small neural network guesses it.
One free Render service: FastAPI + numpy (no sklearn at runtime).

This is the "next step" after the softmax regression version
(https://github.com/bobov1190/softmaxregressionDigits). Same drawing page, same preprocessing.

## The model

```
64 pixels -> 128 hidden units (ReLU) -> 10 numbers -> softmax -> 10 probabilities
```

- 64*128 + 128 + 128*10 + 10 = 9610 parameters (softmax regression has 650)
- the last step is still softmax, so "probabilities sum to 1" is the same idea as in the lesson
- the new part is the hidden layer: 128 small "feature detectors" between the pixels and the answer

## An honest lesson: our first test was fake

The first version of this network was trained only on **fake drawings** (dataset images that were
blown up and distorted, plus digits written with fonts), and it was tested only on fake drawings too.
Those tests said "neural net 98% vs softmax 94%".

Then we drew real digits with a mouse on the real site, and the neural net lost to the simple
softmax regression on several of them (a 9 read as 3, a 4 read as 9, a 1 with a foot read as 6).

Why: a bigger model fits its training data more closely. If the training data does not look like
real life, the bigger model is wrong in a more confident way. And a test made from the same fake
data cannot warn you about it.

So the rule now is: **train on real drawings and test on real drawings that the model has never seen.**

## How we train now (`train_mine.py`)

1. **Collect your own drawings.** Run the site locally in collect mode:

   ```
   pip install -r requirements.txt
   # PowerShell:
   $env:COLLECT = "1"; uvicorn app.main:app
   # bash:
   COLLECT=1 uvicorn app.main:app
   ```

   Open http://127.0.0.1:8000, draw a digit and press its key (0-9) or a "save as" button.
   The raw canvas picture is saved to `data/mine/<digit>/<time>.png`.
   Draw the way you really draw: different sizes, thin and thick, sloppy ones too.
   About 15-20 drawings per digit is a good start.
   Collect mode works only locally. Render never sets `COLLECT`, so the public site saves nothing.

2. **Lock away a test set.** About 30% of your drawings are chosen as the test set. The choice is made
   from a hash of the file name, so when you add more drawings later, an old test drawing never
   moves into the training set. The model never trains on them.

3. **Train.** Training data = fake data (distorted dataset images + 15 fonts, as before) + the other ~70% of your
   drawings, each one also repeated 40 times with a small random tilt, shear and brush thickness
   (real drawings are rare, so they need to be "multiplied"). Everything goes through the same
   `app/preprocess.py` as the website.

   ```
   python train_mine.py            # full run, a few minutes, saves the new weights
   python train_mine.py --quick    # tiny smoke test, saves nothing
   ```

4. **Compare on the same test.** Both models, softmax and the neural net, are trained on exactly the same
   data with the same optimizer (Adam, mini-batches of 256, cosine learning-rate decay). Then both are tested on
   your locked-away drawings. The script prints a table with four columns:
   softmax and net trained on fake data only (before), and softmax and net trained with your drawings (after).
   The table is also saved to `model/results_mine.md`.

Files: `model/nn_weights.npz` (served by the site, replaced by `train_mine.py`),
`model/nn_synthetic.npz` (the old net trained on fake data only), `model/softmax_baseline.npy`
(old softmax, fake data only), `model/softmax_mine.npy` (new softmax, 10 x 65, column 0 is beta_0).

## Results

The numbers of the first version (fake tests only, so do not trust them too much):

| fake test set                     | softmax | neural net |
|-----------------------------------|--------:|-----------:|
| dataset test images 8x8           |   93.6% |      98.1% |
| the same images drawn as a canvas |   90.0% |      96.9% |
| 3 fonts never seen in training    |   96.9% |      99.9% |

The honest table on real drawings (made by `train_mine.py`) goes to `model/results_mine.md`.

## Things to remember

- The dataset is only 8x8 and the models are small, so thin 1s and 7s can still be mixed up.
- Softmax has no "I don't know" answer. A filled or random canvas still gets a confident digit,
  because the probabilities have to sum to 1.
- A test set is only honest if the model never saw it, and if it looks like what the model will meet in real life.

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

- `POST /predict` with `{"image": "<data URL or base64 PNG>"}` returns
  `{"digit": 4, "probs": [10 numbers], "grid": [8][8]}` (grid = what the model sees, 0..16)
- `GET /config` tells the page whether collect mode is on
- `POST /collect` (collect mode only) saves a labeled drawing
