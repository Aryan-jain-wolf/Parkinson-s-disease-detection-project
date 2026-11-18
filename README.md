# Parkinson's Disease Detection Using Voice Signal Analysis

Research project: analyse **jitter, shimmer, pitch and harmonic features** of sustained-vowel recordings and
**compare six machine learning algorithms** for separating Parkinson's-related voice patterns from healthy ones,
using a leakage-safe, participant-grouped evaluation.

> **Research prototype only.** Not a medical device and not a diagnostic tool. Scores show similarity to voice
> patterns in a small research dataset. They must never be used for clinical decisions.

## What is inside

```
parkinsons-voice-project/
├── train.py                 # CLI: grouped CV for 6 models, figures, results JSON, saved best model
├── pdvoice/
│   ├── data.py              # UCI loader, subject ids, synthetic demo data
│   ├── features.py          # WAV -> 16 acoustic features (F0, jitter x5, shimmer x6, NHR, HNR), NumPy/SciPy only
│   ├── modeling.py          # pipelines, StratifiedGroupKFold, metrics, permutation importance, effect sizes
│   ├── reporting.py         # figures, JSON, LaTeX table rows
│   ├── service.py           # results + audio scoring (framework-free)
│   └── api.py               # FastAPI endpoints used by the dashboard
├── dashboard/index.html     # your research dashboard (browser UI; still shows simulated numbers)
├── scripts/download_data.py # fetch the UCI dataset
├── tests/                   # 24 unit tests (python -m unittest discover -s tests)
├── data/  results/  models/ # dataset in, results out, saved models out
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) get the real dataset (see data/README.md)
python scripts/download_data.py

# 2) run the study: 6 models x grouped 5-fold CV x 3 repeats, for two feature sets
python train.py

# 3) check everything works
python -m unittest discover -s tests

# 4) start the API + dashboard
uvicorn pdvoice.api:app --reload        # http://127.0.0.1:8000  (API docs: /docs)
```

No dataset yet? `python train.py --demo` runs the whole pipeline on **synthetic** UCI-like data. This only checks
that the code works; every output is labelled DEMO and **must not be reported as a result**.

## How it works

```
Voice recording (.wav, sustained "Ah")
   -> preprocessing (DC removal, <=12 s, resample to 16 kHz, peak-normalise)
   -> feature extraction (F0, jitter, shimmer, NHR/HNR)
   -> [UCI table: 22 features incl. nonlinear RPDE, DFA, spread1/2, D2, PPE]
   -> impute + scale (inside each CV fold)
   -> StratifiedGroupKFold by participant -> LR | SVM | RF | KNN | DT | GB
   -> accuracy, precision, recall, specificity, F1, ROC-AUC (mean +/- SD over folds and repeats)
   -> best model by mean F1 -> permutation importance, saved model
```

**Why grouped CV matters.** The dataset has ~6 recordings per person. With an ordinary random split, the same
person ends up in training and test sets and the model can recognise the speaker instead of the disease.
`train.py` therefore reports the honest grouped numbers and, separately, the leaky ordinary-split numbers
(`results/<set>/leakage_comparison.json`, `figures/leakage_comparison.png`) so you can show the inflation.

**Two feature sets.**
* `all22`: the full UCI table. Used for the research comparison.
* `classical16`: only the measures this project can compute from audio. Used for the audio scoring endpoint.

## Outputs (per feature set, in `results/<set>/`)

| File | Content | Use in the paper |
|---|---|---|
| `model_results.json` | metrics and confusion counts for all six models (dashboard-compatible fields) | Table II, Fig. 3(b),(c) |
| `table_ii_rows.tex` | ready LaTeX rows, best value in bold | paste into Table II |
| `effect_sizes.csv` | Cohen's d per recording and per subject, class means | Table I, Fig. 3(a) |
| `feature_importance.json` | permutation importance of the best model | discussion |
| `leakage_comparison.json` | grouped vs ordinary split | Section VI-B |
| `cv_folds.csv` | every fold of every repeat | supplementary |
| `figures/*.png` | model comparison, ROC, confusion matrix, leakage, effect sizes, importance | figures |

Also `results/dataset_summary.json` and `models/<set>_best.joblib`.

## API

| Method and path | Purpose |
|---|---|
| `GET /api/health` | status |
| `GET /api/dataset-summary` | counts, subjects, missing values |
| `GET /api/model-results?feature_set=all22` | six-model comparison (replaces the dashboard's simulated `MODELS`) |
| `GET /api/feature-importance?feature_set=all22` | permutation importance |
| `POST /api/analyze-voice` (multipart `file`) | features + similarity score for an uploaded `.wav` |
| `POST /api/train-models?demo=false&repeats=3` | re-run training |

## Tools used

| Area | Tool | Used for |
|---|---|---|
| Language | Python 3 (tested on 3.12) | everything in the backend |
| Numerics and data | **NumPy**, **pandas** | arrays, tables, per-subject aggregation |
| Signal processing | **SciPy** (`signal`, `io.wavfile`) | WAV reading, band-pass filtering, peak detection, resampling |
| Machine learning | **scikit-learn** | `Pipeline`, `StandardScaler`, `SimpleImputer`, `StratifiedGroupKFold`, six classifiers, metrics, permutation importance |
| Model storage | **joblib** | saving the fitted best pipeline |
| Plots | **Matplotlib** | result figures |
| Web API | **FastAPI**, **Uvicorn**, python-multipart | REST endpoints and file upload |
| Dashboard (front end) | HTML/CSS/JavaScript, **Chart.js**, Web Audio API, MediaRecorder API | interactive research dashboard |
| Dataset | UCI *Parkinsons* (Oxford PD Detection) dataset, `ucimlrepo` (optional) | training and evaluation data |
| Testing | `unittest` (standard library) | 24 tests, including synthetic vowels with known jitter/shimmer |
| Paper | LaTeX (pdfLaTeX, Overleaf) | the IEEE-style paper |

Not used (possible extensions): Praat/Parselmouth, librosa, SHAP, XGBoost, deep learning.

## Honest limitations

* The audio extractor follows standard definitions but is **not** the proprietary MDVP software that produced the
  UCI numbers. On synthetic vowels it recovers F0 almost exactly, jitter within about 15 % and shimmer within
  about 6 % of ground truth, but real recordings will differ. The scoring endpoint warns when a feature falls
  outside the training range. Validate on your own recordings before trusting any score.
* RPDE, DFA, spread1, spread2, D2 and PPE are **not** computed from audio. They exist only in the UCI table.
* 31 speakers is a very small, narrow sample. High scores on it do not imply clinical performance. External
  validation on independent data is required.
* Models use fixed, sensible hyper-parameters (`class_weight="balanced"` where available) and no tuning, to avoid
  optimistic bias on so little data. Add nested CV if you want to tune.
* Voice is sensitive personal data: get consent, anonymise, store securely.

## Updating the paper with real results

1. `python train.py` on the real dataset.
2. Copy `results/all22/table_ii_rows.tex` into Table II and the numbers from `effect_sizes.csv` into Table I.
3. Replace the "simulated" wording in the abstract, Section VII and the figure captions.
