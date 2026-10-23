#!/usr/bin/env python
"""Properly-tuned Central (A) baseline for the SVM and logistic-regression variants.

Central (A) is fitted, for all sixteen model variants alike, at one fixed configuration
(alpha=1e-3, SGDClassifier, tol=1e-3, max_iter=1000) -- Supplementary Figure 1's whole point is
to compare regularisation choices under a shared, uncontrolled setting, not to find each
variant's best-possible score. That is unchanged here.

But the same fixed config is also what the manuscript quotes as "the ceiling" against which the
federated/pooled arms (D) are discussed (Model Selection, Results, R1.3). For the four SVM and
four logistic-regression variants, that fixed SGD fit turned out not to be a fair ceiling: D's
pooled baseline fits the SAME model class with a converged solver (LogisticRegression / LinearSVC)
and a properly searched regularisation strength, and for log_l2 that pooled, cross-cohort number
(0.806) came in ABOVE the supposed in-distribution ceiling (0.763) -- incoherent for a value the
text calls an upper bound.

This script re-fits Central (A) for all eight svm*/log* variants with the same fit quality D's
pooled baseline gets, on the SAME 25 folds (5 reps x 5-fold, same rep-varying seed) as the
canonical central arm, so the numbers are directly comparable. Trees (rf*, lgbm*) are not touched:
both A and D-pooled already fit LightGBM directly, so there is no solver mismatch to fix.

Two scalings, selectable via --scale: 'zscore' (log1p -> StandardScaler, the SIAMCAT default every
other number in this codebase uses -- isolates the solver/tuning fix as the only variable) and
'l2norm' (log1p -> per-sample L2 unit-norm, the scaling Reviewer 2 named and Supplementary Table 7
found beats z-score, but which had only ever been measured on the untuned central arm via
compare_normalization.py, never combined with a converged solver). Default 'both' runs each variant
twice and tags every row/summary entry with which scaling produced it.

Regularisation strength (l1/l2/elasticnet) is chosen by a 5-fold CV nested INSIDE each outer
training fold -- never touching the outer test fold, so nothing leaks. The unregularised base
variants (svm, log) have no regularisation-strength axis to search; sklearn's LogisticRegression/
LinearSVC cannot express "no penalty" for hinge loss, so those two keep SGD but with a real
convergence budget (tol=1e-5, max_iter=50000) instead of the pipeline's tol=1e-3/max_iter=1000 --
the same fallback the pooled-tuning code already uses for svm_elasticnet, where no converged
hinge+elasticnet solver exists either.

Does NOT touch the federated or pooled simulations, _tuned_estimator, _client_estimator, or
PreFittedVotingClassifier -- nothing about arm (D) changes. Writes only tunedA_*_central.csv;
no canonical table is overwritten.

Run from inside code/:

    uv run python tuned_central.py                  # all four datasets, all 8 svm*/log* variants
    uv run python tuned_central.py --dataset meteor  # one dataset
    uv run python tuned_central.py --dry-run         # 1 repetition, 2 folds -- smoke test only

Supplementary Figure 1 takes SVM with unit-length normalisation ('l2norm') and LogReg with z-score
standardisation ('zscore'), the one setup per model the paper reports. Those cells are run in parallel by
run_tunedA_parallel.py, with the same functions and seeds; see run_full_server.sh.
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler

import run_analysis as ra

PREFIX = 'tunedA_'
VARIANTS = ['svm', 'svm_l1', 'svm_l2', 'svm_elasticnet', 'log', 'log_l1', 'log_l2', 'log_elasticnet']
SCALES = ['zscore', 'l2norm']

# Same grid D's pooled baseline searches (run_analysis._LINEAR_GRID); irrelevant for svm/log (no
# penalty to strengthen), used for the other six variants.
_C_GRID = ra._LINEAR_GRID

# name -> label. Loaded via the same experiment-name + load_experiment() path
# compare_normalization.py uses, one canonical lgbm-suffixed experiment per profiler.
DATASETS = [
    ('count_taxa_meteor_0',     'METEOR2'),
    ('count_taxa_metaphlan4_0', 'MetaPhlAn4'),
    ('count_taxa_sylph_0',      'Sylph'),
]


def _family_penalty(variant):
    family, *suffix = variant.split('_')
    penalty = suffix[0] if suffix else None
    return family, penalty


def _scaled(clf, scale):
    """log1p, then either per-feature z-score (StandardScaler, the SIAMCAT default used
    everywhere else in this codebase) or per-sample L2 unit-norm (Normalizer) -- the same two
    treatments compare_normalization.py contrasts on the untuned central arm, here applied to a
    properly tuned, converged fit instead."""
    scaler = StandardScaler() if scale == 'zscore' else Normalizer(norm='l2')
    return Pipeline([('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                      ('scale', scaler), ('clf', clf)])


def _converged_or_real_budget(variant, C, rep, scale):
    """The best available fit for this exact penalty -- converged solver where one exists for
    this loss+penalty combination, otherwise SGD given a real convergence budget instead of the
    pipeline's loose default. `scale` selects the preprocessing (see _scaled); everything else is
    identical between the two, so any difference isolates the scaling choice."""
    family, penalty = _family_penalty(variant)
    seed = ra.RANDOM_STATE + rep

    if family == 'log':
        if penalty is None:
            clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
        elif penalty == 'l1':
            clf = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=5000, random_state=seed)
        elif penalty == 'elasticnet':
            clf = LogisticRegression(C=C, penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                      max_iter=5000, random_state=seed)
        else:  # l2
            clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=5000)
    elif family == 'svm':
        if penalty is None:
            clf = SGDClassifier(loss='hinge', penalty=None, max_iter=50000, tol=1e-5,
                                 learning_rate='optimal', random_state=seed)
        elif penalty == 'elasticnet':
            # No converged hinge+elasticnet solver exists (same gap run_analysis._converged_linear
            # documents for the pooled arm) -- SGD with a real budget instead of a fixed alpha.
            clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=1.0 / max(C, 1e-12),
                                 max_iter=50000, tol=1e-5, learning_rate='optimal', random_state=seed)
        elif penalty == 'l1':
            clf = LinearSVC(C=C, penalty='l1', loss='squared_hinge', dual=False, max_iter=20000, random_state=seed)
        else:  # l2
            clf = LinearSVC(C=C, penalty='l2', loss='squared_hinge', dual='auto', max_iter=20000, random_state=seed)
    else:
        raise ValueError(f'not a linear variant: {variant!r}')

    return _scaled(clf, scale)


def _continuous_scores(est, X):
    if hasattr(est, 'predict_proba'):
        return est.predict_proba(X)[:, 1]
    return np.asarray(est.decision_function(X)).ravel()


def _inner_tune(variant, has_grid, Xtr, ytr, seed, inner_folds, scale):
    """Regularisation strength chosen on the training fold only. No grid for the unregularised
    base variants (svm, log) -- there is nothing to sweep, so this returns C=None and the single
    real-budget fit is used directly."""
    if not has_grid:
        return None
    inner_kf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_c, best_score = None, -np.inf
    for C in _C_GRID:
        scores = []
        for tr, va in inner_kf.split(Xtr, ytr):
            try:
                m = _converged_or_real_budget(variant, C, rep=0, scale=scale).fit(Xtr[tr], ytr[tr])
                scores.append(roc_auc_score(ytr[va], _continuous_scores(m, Xtr[va])))
            except Exception:
                continue
        if scores and np.mean(scores) > best_score:
            best_score, best_c = np.mean(scores), C
    return best_c


def run_one(variant, X, y, repetition, dry_run, scale):
    has_grid = _family_penalty(variant)[1] is not None   # l1/l2/elasticnet search C; base does not
    reps = 1 if dry_run else repetition
    folds = 2 if dry_run else 5
    inner_folds = 2 if dry_run else 5

    rows = []
    for rep in range(reps):
        seed = ra.get_deterministic_random_state('cv_split', rep=rep)
        outer_kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_i, (tr, te) in enumerate(outer_kf.split(X, y)):
            best_c = _inner_tune(variant, has_grid, X[tr], y[tr], seed=seed + 1000 + fold_i,
                                  inner_folds=inner_folds, scale=scale)
            final = _converged_or_real_budget(variant, best_c, rep=rep, scale=scale).fit(X[tr], y[tr])
            auc = roc_auc_score(y[te], _continuous_scores(final, X[te]))
            rows.append(dict(rep=rep, cv=fold_i, chosen_C=best_c, test_roc_auc=auc))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default=None,
                     help='Only run datasets whose name contains this substring (e.g. meteor).')
    ap.add_argument('--repetition', type=int, default=ra.REPETITION)
    ap.add_argument('--scale', choices=SCALES + ['both'], default='both',
                     help="'zscore' (SIAMCAT default, matches D), 'l2norm', or 'both' (default).")
    ap.add_argument('--dry-run', action='store_true', help='1 repetition, 2 folds -- smoke test only.')
    args = ap.parse_args()

    out_dir = Path(ra.save_dir) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = SCALES if args.scale == 'both' else [args.scale]

    summary_path = out_dir.parent / 'figures' / 'tunedA_central_auroc.csv'
    # Merge into any existing summary (e.g. a prior --scale zscore run) rather than clobber it, so
    # running the two scalings as separate invocations still ends up with one combined table.
    prior = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(
        columns=['dataset', 'model', 'scale', 'mean', 'median', 'std', 'count'])
    summary_rows = []

    for name, label in DATASETS:
        if args.dataset and args.dataset not in name:
            continue
        exp = next(e for e in ra.experiments if e.name == f'{name}_lgbm')
        anno_df, counts_df = ra.load_experiment(exp, ra.data_dir)
        anno_df = ra.apply_cohort_definitions(anno_df, merge_germany=True, split_italy=True)
        central_data, _, _, _, _ = ra._prepare_simulation_data(anno_df, counts_df, 'health_status', ['cohort_name'])
        X = central_data.drop(columns=['health_status', 'cohort_name']).to_numpy()
        y = central_data['health_status'].to_numpy()
        print(f'{label}: X={X.shape} y+={int(y.sum())}/{len(y)}')

        for scale in scales:
            print(f' -- scale={scale} --')
            for variant in VARIANTS:
                df = run_one(variant, X, y, args.repetition, args.dry_run, scale)
                out_path = out_dir / f'{PREFIX}{scale}_{name}_{variant}_central.csv'
                df.to_csv(out_path, index=False)
                mean, std, median, n = df['test_roc_auc'].mean(), df['test_roc_auc'].std(ddof=1), \
                    df['test_roc_auc'].median(), len(df)
                print(f'  {variant:16s} mean={mean:.4f} sd={std:.4f} median={median:.4f} n={n}  -> {out_path.name}')
                summary_rows.append(dict(dataset=label, model=variant, scale=scale, mean=round(mean, 4),
                                          median=round(median, 4), std=round(std, 4), count=n))

    if summary_rows:
        new = pd.DataFrame(summary_rows)
        combined = pd.concat([prior, new]).drop_duplicates(subset=['dataset', 'model', 'scale'], keep='last')
        combined = combined.sort_values(['dataset', 'scale', 'model']).reset_index(drop=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(summary_path, index=False)
        print(f'\nwrote summary -> {summary_path} ({len(combined)} rows)')


if __name__ == '__main__':
    main()
