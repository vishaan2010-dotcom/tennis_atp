"""
Stability and significance test for the ablation study.

The Ablation Study tab reports a single-run AUC for each feature tier. A single
run cannot tell you whether the gap between tiers is a real effect or just
run-to-run noise, so this script answers two separate questions:

  1. Model variability. The gradient booster uses stochastic subsampling, so its
     AUC moves a little every time it is fitted. Repeating every tier across many
     random seeds on the identical train/test split measures that movement, and a
     paired per-seed delta shows whether the tier ordering survives it.

  2. Sampling uncertainty. The hold-out set is one finite sample of matches. A
     paired bootstrap resamples the test matches with replacement and produces a
     95% confidence interval for the AUC gap. If that interval contains zero, the
     gap is not distinguishable from zero on this data.

Both are reported honestly, including the case where the interval crosses zero.

Usage:
    python src/stability_test.py                      # fetch data as the app does
    python src/stability_test.py --data-dir path/     # read local atp_matches_YYYY.csv
    python src/stability_test.py --seeds 20 --boot 2000

This script trains models only. It does not import or launch the Streamlit UI.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # noqa: E402  (reuses the app's feature engineering, no duplication)


# Tier names match the labels used in the Ablation Study tab.
def build_tiers(feature_cols):
    return [
        ("Ranking only", app.RANKING_FEATURES),
        ("+ Physiological Load", app.RANKING_FEATURES + app.LOAD_FEATURES),
        ("+ Form", app.RANKING_FEATURES + app.LOAD_FEATURES + app.FORM_FEATURES),
        ("+ H2H + Quality", app.RANKING_FEATURES + app.LOAD_FEATURES
         + app.FORM_FEATURES + app.H2H_FEATURES + app.QUALITY_FEATURES),
        ("Full (all features)", feature_cols),
    ]


def load_features(data_dir=None, years=8):
    """Return the engineered feature frame, from local CSVs or the app's loader."""
    if data_dir:
        paths = sorted(glob.glob(os.path.join(data_dir, "atp_matches_[12][0-9][0-9][0-9].csv")))
        if not paths:
            raise SystemExit(f"No atp_matches_YYYY.csv files found in {data_dir}")
        paths = paths[-years:]
        print(f"Local data: {', '.join(os.path.basename(p) for p in paths)}")
        raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    else:
        print(f"Fetching {years} seasons through app.load_raw_data() ...")
        raw = app.load_raw_data(years=years)
        if raw.empty:
            raise SystemExit("No match data returned. Check network access to the data source.")
    return app.engineer_features(raw)


def fit_tier(X_train, y_train, X_test, seed):
    """Fit one tier at one seed and return its test-set probabilities."""
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_split=20, subsample=0.8, random_state=seed
    )
    gbm.fit(X_train, y_train)
    return gbm.predict_proba(X_test)[:, 1]


def paired_bootstrap(y_test, prob_a, prob_b, n_boot, rng):
    """95% percentile CI for AUC(b) - AUC(a), resampling test matches with replacement.

    Both tiers are scored on the identical resample, which is what makes the
    comparison paired and removes the variance common to both models.
    """
    n = len(y_test)
    deltas = np.empty(n_boot)
    kept = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y_test[idx]
        if ys.min() == ys.max():        # degenerate resample, no positives or no negatives
            continue
        deltas[kept] = roc_auc_score(ys, prob_b[idx]) - roc_auc_score(ys, prob_a[idx])
        kept += 1
    deltas = deltas[:kept]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return deltas.mean(), lo, hi, float((deltas > 0).mean()), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="Folder of atp_matches_YYYY.csv files. Omit to fetch as the app does.")
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ref-seed", type=int, default=42,
                    help="Seed whose predictions feed the bootstrap (42 = the app's seed).")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    t0 = time.time()
    df = load_features(args.data_dir, args.years)
    feature_cols = [c for c in df.columns if c != "p1_wins"]
    y = df["p1_wins"].values

    # Identical chronological positional hold-out as app.train_model and the
    # ablation tab. The split never varies, so only model randomness varies.
    split = int(len(df) * 0.8)
    y_train, y_test = y[:split], y[split:]

    tiers = build_tiers(feature_cols)
    seeds = list(range(args.seeds))
    if args.ref_seed not in seeds:
        seeds.append(args.ref_seed)

    print(f"\nMatches: {len(df)}   train: {split}   test: {len(df) - split}")
    print(f"Seeds: {args.seeds} (+ reference seed {args.ref_seed})   bootstrap: {args.boot}\n")

    auc = {}       # tier -> array of AUC, one per seed in `seeds` order
    ref_prob = {}  # tier -> test probabilities at the reference seed

    for name, cols in tiers:
        cols_present = [c for c in cols if c in df.columns]
        X = df[cols_present].values
        Xtr, Xte = X[:split], X[split:]
        probs = Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(fit_tier)(Xtr, y_train, Xte, s) for s in seeds
        )
        auc[name] = np.array([roc_auc_score(y_test, p) for p in probs])
        ref_prob[name] = probs[seeds.index(args.ref_seed)]
        print(f"{name:24s} {len(cols_present):3d} feat   "
              f"AUC {auc[name].mean():.4f} +/- {auc[name].std(ddof=1):.4f}   "
              f"[min {auc[name].min():.4f}, max {auc[name].max():.4f}]")

    base = "Ranking only"
    comparisons = [("+ Physiological Load", "load features vs ranking alone"),
                   ("Full (all features)", "all features vs ranking alone")]

    print("\nPaired per-seed deltas (same seed, same split, tier vs ranking only)")
    for tier, label in comparisons:
        d = auc[tier] - auc[base]
        print(f"  {label:34s} mean {d.mean():+.4f} +/- {d.std(ddof=1):.4f}   "
              f"positive in {int((d > 0).sum())}/{len(d)} seeds")

    rng = np.random.default_rng(0)
    print(f"\nPaired bootstrap on the test set, seed {args.ref_seed} predictions")
    for tier, label in comparisons:
        m, lo, hi, p_pos, kept = paired_bootstrap(
            y_test, ref_prob[base], ref_prob[tier], args.boot, rng)
        crosses = lo < 0 < hi
        verdict = ("CROSSES ZERO: not distinguishable from zero on this hold-out"
                   if crosses else "Does not cross zero on this hold-out")
        print(f"  {label}")
        print(f"    delta AUC {m:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"P(delta > 0) = {p_pos:.3f}   resamples kept {kept}")
        print(f"    {verdict}")

    print(f"\nElapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
