#!/usr/bin/env python
"""Supplementary Figure 1 — central hyperparameter/model-variant screen across profilers.

Runs the `central` arm (pooled, stratified 5-fold CV, `repetition` repeats) over the FULL 16-variant
grid — {rf, lgbm, svm, log} x {none, l1, l2, elasticnet}, i.e. method x regularisation — for each of the
four feature matrices the paper uses:

    METEOR2 (species)   count_taxa_meteor_0     taxa_counts_meteor_0.csv   anno_full_clean.csv
    METEOR2 (strain)    count_strain_meteor_nan_0  strain_meteor_nan.csv   anno_full_nan.csv
    MetaPhlAn4          count_taxa_metaphlan4_0 taxa_meta4.csv             anno_full_clean.csv
    Sylph               count_taxa_sylph_0      taxa_sylph.csv             anno_full_clean.csv

Everything is written under the `hpo_` prefix (`data/simulations/tables/hpo_<experiment>_central.csv`),
so no canonical table is touched — the other evaluations come from the other steps of run_full_server.sh.

Why a driver script and not extra entries in `run_analysis.experiments`: the four datasets other than
METEOR2 species are pinned to a single model there on purpose (strain and MetaPhlAn4 are lgbm-only
arms, Sylph carries four unregularised variants). Fanning them out globally would add 64 experiments to
every no-flag run of every arm. The screen is central-only and cheap, so it lives here.

**Not every cell is fittable.** The strain matrix is ~22% NaN — 21.7% raw, 21.6% over the rows
actually fitted (an absent strain is signal; LGBM splits on it natively) — and its annotation (`anno_full_nan.csv`) leaves ~11% of age/sex/BMI missing. The linear
families go through `log1p -> <scaler> -> SGDClassifier`, which cannot ingest NaN, so those eight cells
have no value to plot. They are reported as missing, never as zero, and the
figure states it.

Run from inside code/:

    uv run python hpo_central.py                 # all four datasets, 16 variants each
    uv run python hpo_central.py --dataset sylph # one dataset
    uv run python hpo_central.py --dry-run       # 1 repetition, 2 folds -- smoke test only
    uv run python hpo_central.py --models svm,svm_l1,svm_l2,svm_elasticnet   # rescreen one family

The two linear families are screened under DIFFERENT scalers -- see LINEAR_SCALING below.
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

import run_analysis as ra
from utility import Experiment, propagate_experiments

PREFIX = 'hpo_'

# Scaling of the two linear families in this screen. The defaults are the scalers run_analysis.py's
# templates already carry (one scaling per family, the same in every evaluation); --svm-scaling /
# --log-scaling override them for this screen only.
#
#   svm* -> 'l2norm'  log1p -> Normalizer(norm='l2'), per SAMPLE. The scaling Reviewer 2 names for the
#                     SVM, and the one that gives it its gain in the centralized screen.
#   log* -> 'zscore'  log1p -> StandardScaler, per FEATURE. The SIAMCAT default the rest of the
#                     pipeline uses.
#
# The scaler REPLACES its counterpart -- it is never stacked on top of it; the template pipeline has a
# single 'scale' slot. Only that slot is swapped: solver, alpha, folds and seeds are identical across all
# 16 variants, so the screen stays comparable on everything except the one step named here.
#
# NOTE FOR THE CAPTION: with these defaults the two linear families run DIFFERENT preprocessing and the
# figure itself says so nowhere (its explanatory text was removed on purpose). State it in the caption.
LINEAR_SCALING = {'svm': 'l2norm', 'log': 'zscore'}

_SCALERS = {'zscore': StandardScaler, 'l2norm': lambda: Normalizer(norm='l2')}


def apply_linear_scaling(scaling):
    """Swap the 'scale' step of each linear template in ra.estimators. In place: get_estimator() clones
    these templates, so every fit in this run picks the new scaler up, and nothing outside this process
    is affected."""
    for name, est in ra.estimators.items():
        want = scaling.get(name.split('_')[0])
        if want is None or not isinstance(est, Pipeline):
            continue
        est.set_params(scale=_SCALERS[want]())

# name -> (annotation, count matrix, label for the figure). The annotation is the one the canonical
# arm for that matrix uses, so a variant screened here is comparable to the arm it feeds.
DATASETS = [
    ('count_taxa_meteor_0',       ra.anno_all, ra.taxa_counts_meteor_0, 'METEOR2'),
    ('count_strain_meteor_nan_0', ra.anno_nan, ra.strain_meteor_nan,    'METEOR2 (strain)'),
    ('count_taxa_metaphlan4_0',   ra.anno_all, ra.taxa_meta4,           'MetaPhlAn4'),
    ('count_taxa_sylph_0',        ra.anno_all, ra.taxa_sylph,           'Sylph'),
]


def build_experiments(dataset_filter=None):
    base = [Experiment(name=n, anno=a, count=c)
            for n, a, c, _ in DATASETS
            if not dataset_filter or dataset_filter in n]
    if not base:
        raise SystemExit(f"No dataset matched --dataset={dataset_filter!r}. "
                         f"Known: {[n for n, *_ in DATASETS]}")
    return propagate_experiments(base, ra.estimators.keys())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default=None,
                   help='substring filter on the dataset name (e.g. sylph, strain, metaphlan4)')
    p.add_argument('--models', default=None,
                   help='comma list restricting the 16-variant grid (default: all 16)')
    p.add_argument('--repetition', type=int, default=ra.REPETITION)
    p.add_argument('--cross-val', type=int, default=ra.CROSS_VAL)
    p.add_argument('--n-jobs', type=int, default=ra.N_JOBS)
    p.add_argument('--threads-per-job', type=int, default=ra.THREADS_PER_JOB)
    p.add_argument('--dry-run', action='store_true', help='1 repetition, 2 folds, prefix dryrun_hpo_')
    p.add_argument('--svm-scaling', choices=sorted(_SCALERS), default=LINEAR_SCALING['svm'],
                   help="scaler for the svm* variants (default: %(default)s)")
    p.add_argument('--log-scaling', choices=sorted(_SCALERS), default=LINEAR_SCALING['log'],
                   help="scaler for the log* variants (default: %(default)s)")
    args = p.parse_args()

    scaling = {'svm': args.svm_scaling, 'log': args.log_scaling}
    apply_linear_scaling(scaling)
    # The screen fits every variant at one fixed configuration; the central arm's nested C search for
    # the penalised linear models stays off here.
    ra.TUNE = False

    exps = build_experiments(args.dataset)
    if args.models:
        keep = {m.strip() for m in args.models.split(',')}
        exps = [e for e in exps if e.model in keep]
        if not exps:
            raise SystemExit(f"No experiment left after --models={args.models!r}")

    prefix = PREFIX
    repetition, cross_val = args.repetition, args.cross_val
    if args.dry_run:
        prefix, repetition, cross_val = 'dryrun_' + PREFIX, 1, 2

    print('=' * 64)
    print(f"  central model-variant screen (Supplementary Figure 1)")
    print(f"  experiments: {len(exps)}  ({len({e.name.rsplit('_', 1)[0] for e in exps})} dataset(s) "
          f"x {len({e.model for e in exps})} variant(s))")
    print(f"  repetition : {repetition} | cross_val: {cross_val} | seed: {ra.RANDOM_STATE} | exec: redo")
    print(f"  scaling    : " + ' | '.join(f'{f}* -> log1p+{s}' for f, s in scaling.items()))
    print(f"  output     : data/simulations/tables/{prefix}<experiment>_central.csv")
    print('=' * 64)

    ra.run_simulation_with_threading(
        exps,
        simulation_type='central',
        scoring=ra.scoring,
        feature_importance='',
        verbose=False,
        EXEC_MODE='redo',
        REPETITION=repetition,
        CROSS_VAL=cross_val,
        prefix=prefix,
        max_threads=args.n_jobs,
        threads_per_job=args.threads_per_job,
    )

    written = sorted(Path(ra.table_dir).glob(f'{prefix}*_central.csv'))
    print(f"\n{len(written)} of {len(exps)} variant tables written to {ra.table_dir}")
    missing = {e.name for e in exps} - {f.name[len(prefix):-len('_central.csv')] for f in written}
    if missing:
        print(f"not fittable / failed ({len(missing)}): {', '.join(sorted(missing))}")


if __name__ == '__main__':
    main()
