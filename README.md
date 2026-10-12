# Neural Net Digits

Draw a digit in the browser, the server shrinks it to 8x8 and a small neural network guesses it.
One free Render service: FastAPI + numpy (no sklearn at runtime).

The page shows both models side by side (softmax regression and the neural net) and lets you look inside:
see "The page: two models and a look inside" below.

This is the "next step" after the softmax regression version
(https://github.com/bobov1190/softmaxregressionDigits). Same drawing page, same preprocessing, and at first the same training data (only the model changed).
Later both models were retrained on real drawings, see "How we train now" below.

## The model

```
64 pixels -> 128 hidden units (ReLU) -> 10 numbers -> softmax -> 10 probabilities
```

- 64*128 + 128 + 128*10 + 10 = 9610 parameters (softmax regression has 650)
- the last step is still softmax, so "probabilities sum to 1" is the same idea as in the lesson
- the new part is the hidden layer: 128 small "feature detectors" between the pixels and the answer

## First version: how it was trained (`train_nn.py`, fake data only)

- plain numpy, no deep-learning library
- data, the same as for the softmax v2 model (`data_aug.py`):
  - the original 8x8 digits (80% of the dataset, the other 20% is never trained on)
  - many distorted "canvas" versions of them (rotation, slant, brush thickness)
  - digits written with 15 fonts
- Adam optimizer, mini-batches of 256, 80 epochs, learning rate 2e-3 with cosine decay, small weight decay
- He initialisation (random weights, scaled by the layer size)
- three fonts (comic, calibri, verdana) are held out and used only for testing

Run it yourself: `pip install scikit-learn pillow numpy`, then `python train_nn.py`
(needs Windows fonts, takes a few minutes). It writes `model/nn_weights.npz` (careful: that is the file `train_mine.py` writes too).

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

The softmax column is `model/softmax_baseline.npy`, trained on exactly the same data.
Fonts-as-drawn-with-a-thick-brush (`scripts/check_pipeline.py`) are harder than clean
font digits: about 85% on the server pipeline.

### Honest table: real mouse drawings (made by `train_mine.py`)

150 real drawings (15 per digit): 107 used for training, 43 locked away for testing
(per digit: 0:2, 1:6, 2:5, 3:3, 4:5, 5:4, 6:4, 7:6, 8:3, 9:5). Both models were trained
on exactly the same data (fake data + the 107 drawings, each repeated 40 times with small jitter).

| test set | softmax (fake only) | net (fake only) | softmax (+ yours) | net (+ yours) |
|---|---:|---:|---:|---:|
| YOUR drawings (n=43) | 69.8% | 74.4% | 86.0% | 95.3% |
| fake: dataset 8x8 | 93.6% | 98.1% | 92.2% | 98.3% |
| fake: as canvas | 90.3% | 96.4% | 90.0% | 96.9% |
| fake: unseen fonts | 97.6% | 99.8% | 97.2% | 99.9% |

Mistakes on the 43 locked-away drawings:
- softmax (+ yours): 0 read as 5, 1 as 6, 1 as 8, 4 as 9, 5 as 9, 9 as 0 (6 mistakes)
- net (+ yours): 1 read as 7, 9 read as 0 (2 mistakes)

What this shows:
- Training on real drawings helped both models a lot on real drawings: softmax 69.8% -> 86.0%, net 74.4% -> 95.3%.
  The fake-only models were weak on real drawings (about 70-75%), even though they scored 90-99% on fake tests.
- On real drawings the net is better than softmax, but the first impression ("the net is worse") came from only 4 drawings.
  Before training on real drawings the two were close (74.4% vs 69.8%).
- Only 43 test drawings: one drawing is about 2.3 percentage points, so 95.3% vs 86.0% is a difference of 4 drawings.
  It is a good sign, not a proof. More test drawings would make the numbers more reliable.
- The net reaches 99.7% on its own training data but 95.3% on new drawings, so it still memorizes a bit.
  Softmax stays at 90% on its training data: a linear model is too simple for some digits, more data would not fix that.
- The test set is small, and it is drawn by one person (the same hand, mouse and canvas size).
  The numbers are for this kind of drawing, another person may get different results.

The table is also saved in `model/results_mine.md`.

## The page: two models and a look inside

One drawing, two answers. Softmax regression and the neural net get the same 8x8 grid and answer next to each other.
When they disagree, the page says so. That is the best moment of the lesson: *why* do they disagree?

**"Why this answer?" maps.** For each model the page draws an 8x8 map of how much each pixel pushed the model toward its
answer instead of its runner-up (the digit with the second-highest probability):

- red pixel: ink here pushed toward the answer, blue pixel: ink here pushed toward the runner-up, white: did not matter
- softmax: **exact**. The score of a digit is `w . x + b`, so a pixel's contribution is `(w_top - w_other) * x`,
  and all 64 contributions plus the bias difference add up to the gap between the two scores (checked in a test:
  1.643 and 1.643)
- neural net: an **approximation**, gradient x input of the same score gap. It adds up to about 7.0 where the real
  gap grew by 6.5 in a test, because the hidden ReLU layer is not a straight line. It shows the direction, not the exact amount

**The 10 softmax templates.** The 64 weights of each digit drawn as an 8x8 picture (`GET /templates`).
Red = ink here votes for this digit, blue = ink here votes against it. A softmax score is these templates laid over your drawing.
The templates are noisy, not neat digit shapes: they come from training on many different drawings, and each weight is
tuned against all other digits at once, not to draw the digit.

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
- `/predict` also returns `models.softmax` and `models.net`: each has `digit`, `probs` and `why`
  (`top`, `other`, and the 8x8 `map` of contributions); the top-level `digit`/`probs` are the neural net's
- `GET /templates` returns the 10 softmax weight pictures (10 x 8 x 8)
- `GET /config` tells the page whether collect mode is on
- `POST /collect` (collect mode only) saves a labeled drawing

## The softmax site got the same update

The softmax trained here on the same data (`model/softmax_mine.npy`) is also installed in the softmax repo
(https://github.com/bobov1190/softmaxregressionDigits, `model/weights.npy`), so both sites can be compared live.
