import numpy as np
from sklearn.metrics import roc_auc_score

# your existing code
from src.features import build_features     # rename to match yours
from src.model import train_and_predict     # rename to match yours

TIERS = {
    "ranking_only": ["rank_diff", "elo_diff"],          # your actual ranking cols
    "plus_load":    ["rank_diff", "elo_diff", "matches_14d", "sets_14d"],
    "full":         None,                                # None = all features
}
SEEDS = range(25)

X_tr, y_tr, X_te, y_te = build_features()

results = {}
preds = {}
for name, cols in TIERS.items():
    aucs = []
    for s in SEEDS:
        p = train_and_predict(X_tr, y_tr, X_te, features=cols, seed=s)
        aucs.append(roc_auc_score(y_te, p))
        if s == 0:
            preds[name] = p
    results[name] = np.array(aucs)
    print(f"{name:14s} AUC {results[name].mean():.4f} +/- {results[name].std():.4f}")

# bootstrap the full-vs-ranking delta on the test set
rng = np.random.default_rng(0)
n = len(y_te)
deltas = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    if len(np.unique(y_te[idx])) < 2:
        continue
    d = roc_auc_score(y_te[idx], preds["full"][idx]) - \
        roc_auc_score(y_te[idx], preds["ranking_only"][idx])
    deltas.append(d)
deltas = np.array(deltas)
lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"\ndelta AUC = {deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print("crosses zero:", lo < 0 < hi)
