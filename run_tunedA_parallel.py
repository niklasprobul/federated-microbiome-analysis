"""Parallel driver for tuned_central.py: same functions, same seeds, selected
(dataset, variant, scale) cells only, folds run in parallel. Writes tunedA_{scale}_{name}_{variant}_central.csv
exactly as tuned_central.main() would."""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import tuned_central as tc
ra = tc.ra

def one_fold(variant, scale, X, y, rep, fold_i, tr, te):
    has_grid = tc._family_penalty(variant)[1] is not None
    seed = ra.get_deterministic_random_state('cv_split', rep=rep)
    best_c = tc._inner_tune(variant, has_grid, X[tr], y[tr], seed=seed + 1000 + fold_i, inner_folds=5, scale=scale)
    final = tc._converged_or_real_budget(variant, best_c, rep=rep, scale=scale).fit(X[tr], y[tr])
    return dict(rep=rep, cv=fold_i, chosen_C=best_c, test_roc_auc=roc_auc_score(y[te], tc._continuous_scores(final, X[te])))

ap = argparse.ArgumentParser()
ap.add_argument('--cells', required=True, help='name:variant:scale,...')
ap.add_argument('--out', required=True); ap.add_argument('--n-jobs', type=int, default=12)
ap.add_argument('--only', default=None, help='rep:fold to run a single fold (validation)')
a = ap.parse_args()
cells = [c.split(':') for c in a.cells.split(',')]
data = {}
for name in sorted({c[0] for c in cells}):
    exp = next(e for e in ra.experiments if e.name == f'{name}_lgbm')
    anno_df, counts_df = ra.load_experiment(exp, ra.data_dir)
    anno_df = ra.apply_cohort_definitions(anno_df, merge_germany=True, split_italy=True)
    cd, *_ = ra._prepare_simulation_data(anno_df, counts_df, 'health_status', ['cohort_name'])
    data[name] = (cd.drop(columns=['health_status', 'cohort_name']).to_numpy(), cd['health_status'].to_numpy())
    print(f'{name}: X={data[name][0].shape}', flush=True)
jobs = []
for name, variant, scale in cells:
    X, y = data[name]
    for rep in range(ra.REPETITION):
        seed = ra.get_deterministic_random_state('cv_split', rep=rep)
        for fold_i, (tr, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y)):
            if a.only and a.only != f'{rep}:{fold_i}': continue
            jobs.append((name, variant, scale, rep, fold_i, tr, te))
print(f'{len(jobs)} fold jobs', flush=True)
res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_fold)(v, s, *data[n], r, f, tr, te) for n, v, s, r, f, tr, te in jobs)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for (name, variant, scale) in cells:
    rows = [r for j, r in zip(jobs, res) if j[:3] == (name, variant, scale)]
    df = pd.DataFrame(rows).sort_values(['rep', 'cv'])
    p = out / f'tunedA_{scale}_{name}_{variant}_central.csv'; df.to_csv(p, index=False)
    print(f'{name} {variant} {scale}: mean={df.test_roc_auc.mean():.4f} sd={df.test_roc_auc.std():.4f} n={len(df)} -> {p.name}', flush=True)
