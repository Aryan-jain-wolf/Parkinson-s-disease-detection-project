#!/usr/bin/env python
"""Train and compare the six classifiers with participant-grouped cross-validation.

Examples
--------
    python train.py                       # real UCI data in data/parkinsons.data
    python train.py --demo                # synthetic demo data (pipeline check only, NOT results)
    python train.py --feature-sets classical16 --repeats 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import joblib
import numpy as np

from pdvoice import __version__
from pdvoice.data import DatasetNotFoundError, FEATURE_SETS, PROJECT_ROOT, load_dataset
from pdvoice.modeling import (best_model_name, cross_validate_models, dashboard_rows, effect_sizes,
                              make_pipeline, model_zoo, permutation_importances)
from pdvoice.reporting import latex_table_rows, make_figures, save_json


def run_feature_set(ds, fs: str, out_root: Path, models_dir: Path, args) -> dict:
    X, y, groups = ds.X(fs), ds.y, ds.groups
    tag = "DEMO - synthetic data, not real results" if ds.is_demo else "UCI Oxford PD dataset"
    out = out_root / fs
    print(f"\n=== feature set: {fs} ({X.shape[1]} features) ===")

    def prog(name, r):
        print(f"  {name:24s} F1={r['f1']:.3f}±{r['f1_std']:.3f}  recall={r['recall']:.3f}  AUC={r['roc_auc']:.3f}  ({r['fit_ms']:.0f} ms/fit)")

    grouped = cross_validate_models(X, y, groups, n_splits=args.folds, repeats=args.repeats, seed=args.seed, grouped=True, progress=prog)
    print("  -- same models with an ordinary (leaky) split, for comparison only --")
    naive = cross_validate_models(X, y, groups, n_splits=args.folds, repeats=args.repeats, seed=args.seed, grouped=False)
    best = best_model_name(grouped)
    print(f"  best by mean F1: {best}")

    importance = permutation_importances(model_zoo(args.seed)[best], X, y, groups, n_splits=args.folds, seed=args.seed)
    effects = effect_sizes(ds.df, FEATURE_SETS[fs])

    rows = dashboard_rows(grouped)
    meta = {
        "feature_set": fs, "features": FEATURE_SETS[fs], "best_model": best, "data_source": ds.source, "is_demo": ds.is_demo,
        "protocol": f"StratifiedGroupKFold({args.folds}), repeats={args.repeats}, seed={args.seed}, scaler fitted inside folds",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "version": __version__,
        "disclaimer": "Research prototype. Not a medical device or diagnostic tool.",
    }
    save_json({**meta, "models": rows}, out / "model_results.json")
    save_json({**meta, "leaky_ordinary_split": {n: {"accuracy": r["accuracy"], "roc_auc": r["roc_auc"]} for n, r in naive.items()},
               "grouped": {n: {"accuracy": r["accuracy"], "roc_auc": r["roc_auc"]} for n, r in grouped.items()}}, out / "leakage_comparison.json")
    save_json({**meta, "method": "permutation importance on held-out grouped folds, ROC-AUC", "importance": importance}, out / "feature_importance.json")
    effects.to_csv(out / "effect_sizes.csv", index=False)
    import pandas as pd
    pd.DataFrame([{**f, "model": n} for n, r in grouped.items() for f in r["folds"]]).to_csv(out / "cv_folds.csv", index=False)
    (out / "table_ii_rows.tex").write_text(latex_table_rows(rows), encoding="utf-8")
    figs = make_figures(out / "figures", grouped, naive, importance, effects, best, tag)

    final = make_pipeline(model_zoo(args.seed)[best].named_steps["model"])
    final.fit(X, y)
    stats = {"features": FEATURE_SETS[fs],
             "min": X.min().to_dict(), "max": X.max().to_dict(),
             "pd_mean": X[y == 1].mean().to_dict(), "pd_std": X[y == 1].std().to_dict(),
             "hc_mean": X[y == 0].mean().to_dict(), "hc_std": X[y == 0].std().to_dict()}
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "meta": meta, "stats": stats}, models_dir / f"{fs}_best.joblib")
    print(f"  saved {out}/ and {models_dir / (fs + '_best.joblib')} ({len(figs)} figures)")
    return {"best": best}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="path to parkinsons.data (default: data/parkinsons.data)")
    ap.add_argument("--demo", action="store_true", help="use synthetic UCI-like data (pipeline check only)")
    ap.add_argument("--feature-sets", nargs="+", default=["all22", "classical16"], choices=list(FEATURE_SETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    args = ap.parse_args(argv)
    try:
        ds = load_dataset(args.data, demo=args.demo)
    except DatasetNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    s = ds.summary()
    print(f"Dataset: {s['source']} | {s['n_recordings']} recordings ({s['n_pd_recordings']} PD / {s['n_healthy_recordings']} healthy), "
          f"{s['n_subjects']} subjects")
    if ds.is_demo:
        print("WARNING: synthetic demo data. These numbers only check that the pipeline runs; they are NOT research results.")
    out = Path(args.out)
    save_json(s, out / "dataset_summary.json")
    for fs in args.feature_sets:
        run_feature_set(ds, fs, out, Path(args.models_dir), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
