#!/usr/bin/env python


# Import libraries
import hashlib
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import traceback
import warnings
import csv

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from itertools import combinations
from os import path
from pathlib import Path
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, FunctionTransformer
from tqdm import tqdm

from lightgbm import LGBMClassifier

# Local imports
from utility import *
from PreFittedVotingClassifier import PreFittedVotingClassifier, FederatedBoostedModel


# === Configuration ===
# ALL run parameters live in config.toml. Running run_analysis.py with no flags reproduces the paper
# experiment from that file; every CLI flag overrides the matching key. The module-level globals below are
# initialised from it (and may still be overridden per-call, e.g. by workers via _pool_init).
import tomllib

_CONFIG_PATH = Path(__file__).resolve().parent / 'config.toml'
with open(_CONFIG_PATH, 'rb') as _cf:
    CONFIG = tomllib.load(_cf)
_C_EXP, _C_FED = CONFIG['experiment'], CONFIG['federated']
_C_COH, _C_RES = CONFIG['cohorts'], CONFIG['resources']

# Simulation settings
REPETITION = _C_EXP['repetition']
EXEC_MODE = _C_EXP['exec_mode']
DEFAULT_SIMULATIONS = list(_C_EXP['simulations'])
# Optional reduced model grid (config: [experiment].models). Empty list = the full 16-model grid.
# --models overrides it. Applied before --check-invariant so the gate is scoped to the same grid.
DEFAULT_MODELS = list(_C_EXP.get('models', []))

# Cross-validation settings
CROSS_VAL = _C_EXP['cross_val']
RANDOM_STATE = _C_EXP['random_state']

# Parallel processing settings
N_JOBS = _C_RES['n_jobs']
THREADS_PER_JOB = _C_RES['threads_per_job']

# Federated aggregation settings
# Federated voting rule (see PreFittedVotingClassifier).
VOTING = _C_FED['voting']                  # auto -> soft when members expose predict_proba, else hard
DEFAULT_WEIGHTING = list(_C_FED['weighting'])
DEFAULT_ROUNDS = _C_FED['rounds']
DEFAULT_MAX_COMBOS = _C_FED['max_combos_per_k'] or None   # 0/absent -> None (exhaustive)

# Groups for grouping data
# currently only supports 1 group at a time
groups = ["cohort_name"]

# Label for prediction
label = ['health_status']


# === Cohort definitions ===
# Canonical cohort definitions, verified against the source publications and ENA run-level provenance:
#   * Germany1 (Wirbel et al. 2019, PRJEB27928; newly-generated ColoCare CRC + PRAEVENT controls, HiSeq
#     4000) and Germany2 (Zeller et al. 2014, PRJEB6070 German subset; ColoCare CRC validation +
#     my.microbes controls, HiSeq 2000) are the same Heidelberg/ColoCare German recruitment, published
#     across two studies. Their union (disjoint samples) reconstructs Wirbel 2019's canonical "DE" cohort
#     -> merged into one ``Germany`` cohort. NOTE this combines two sequencing waves (transparency caveat
#     in the manuscript). Toggle --no-merge-germany for the unmerged view.
#   * Italy1 (Thomas et al. 2019, PRJNA447983) is two Italian cohorts (Vercelli/Alessandria vs Milan/IEO,
#     different IRBs and DNA kits) -> split into ``Italy1a`` / ``Italy1b`` via the ``thomas_alias`` column.
#   * Canada1 + USA2 (both PRJNA389927, Hannigan et al. 2018) are one multi-site North-American study
#     -> Canada1 (n=3) folded into ``USA2``.
#   * NB France1 + Germany2 also share one accession (PRJEB6070, Zeller 2014) but are a French discovery
#     cohort and a German validation cohort -> kept SEPARATE (a shared ENA deposition is not one cohort).
# Applied as a label remap on the loaded annotation table (the shared read-only inputs are never modified).
MERGE_GERMANY = _C_COH['merge_germany']
SPLIT_ITALY = _C_COH['split_italy']
MERGE_HANNIGAN = True
_GERMANY_SOURCES = ["Germany1", "Germany2"]
_ITALY_SPLIT = {"Cohort1": "Italy1a", "Cohort2": "Italy1b"}
_HANNIGAN_MERGE = {"Canada1": "USA2"}   # PRJNA389927 (Hannigan 2018) is one multi-site N.-American study


def apply_cohort_definitions(anno_df, merge_germany=None, split_italy=None, verbose=False):
    """Relabel ``cohort_name`` to the canonical cohort definitions (merge Germany, split Italy1, fold
    the single-study Canada1 into USA2)."""
    merge_germany = MERGE_GERMANY if merge_germany is None else merge_germany
    split_italy = SPLIT_ITALY if split_italy is None else split_italy
    if 'cohort_name' not in anno_df.columns:
        return anno_df
    df = anno_df.copy()
    if merge_germany:
        m = df['cohort_name'].isin(_GERMANY_SOURCES)
        df.loc[m, 'cohort_name'] = 'Germany'
        if verbose:
            print(f"[cohorts] merged {_GERMANY_SOURCES} -> 'Germany' ({int(m.sum())} samples)")
    if split_italy:
        if 'thomas_alias' not in df.columns:
            if verbose:
                print("[cohorts] WARNING: no 'thomas_alias' column; cannot split Italy1.")
        else:
            it = df['cohort_name'] == 'Italy1'
            for alias, new in _ITALY_SPLIT.items():
                sub = it & (df['thomas_alias'] == alias)
                df.loc[sub, 'cohort_name'] = new
                if verbose:
                    print(f"[cohorts] Italy1/{alias} -> '{new}' ({int(sub.sum())} samples)")
    if MERGE_HANNIGAN:
        for src, dst in _HANNIGAN_MERGE.items():
            m = df['cohort_name'] == src
            df.loc[m, 'cohort_name'] = dst
            if verbose and m.any():
                print(f"[cohorts] folded {src} -> '{dst}' ({int(m.sum())} samples, one Hannigan 2018 study)")
    return df


def summarize_cohorts(anno_df, label_col='health_status', min_samples=40):
    """Per-cohort summary: n, class counts, and whether it qualifies as a training cohort
    (both classes present and n > ``min_samples``)."""
    rows = []
    for coh, sub in anno_df.groupby('cohort_name'):
        classes = sub[label_col].dropna().unique()
        rows.append({
            'cohort_name': coh, 'n': len(sub), 'n_classes': len(classes),
            'trainable': bool(len(classes) > 1 and len(sub) > min_samples),
        })
    return pd.DataFrame(rows).sort_values('n', ascending=False).reset_index(drop=True)


# Define scoring metrics
scoring = [
    'accuracy',
    'balanced_accuracy',
    'roc_auc',
    # 'pr_auc',
    'f1',
    'mcc',
]

# List of metric names for reporting
metrics = [f'test_{_}' for _ in scoring]

linear_params = {
    'alpha' : 1e-3, 
    'max_iter' : 1000, 
    'learning_rate' : 'optimal', 
    'tol' : 1e-3, 
    'early_stopping' : False, 
    'n_jobs' : THREADS_PER_JOB
}

forest_params = {
    'n_estimators' : 100,
    'max_depth' : 5,
    'class_weight' : None,
    'num_leaves' : 31,
    'objective' : 'binary',
    'n_jobs' : THREADS_PER_JOB,
    'verbose' : -1,
    'min_data_in_leaf' : 2,
}

# Define models
estimators = {
    # shelved because SLOW
    # 'rf': RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=None,  
    #     criterion='entropy',
    #     min_samples_leaf=5,
    #     max_features='sqrt',
    #     class_weight=None,
    #     max_samples=0.75, 
    #     n_jobs=THREADS_PER_JOB
    # ), 

    'rf': LGBMClassifier( # could try L1 / L2 with this
        boosting_type='rf',
        bagging_fraction=0.7,
        feature_fraction=0.7,
        **forest_params
    ),
    'rf_l1': LGBMClassifier( # could try L1 / L2 with this
        boosting_type='rf',
        bagging_fraction=0.7,
        feature_fraction=0.7,
        reg_alpha=1.0,
        **forest_params
    ),
    'rf_l2': LGBMClassifier( # could try L1 / L2 with this
        boosting_type='rf',
        bagging_fraction=0.7,
        feature_fraction=0.7,
        reg_lambda=1.0,
        **forest_params
    ),
    'rf_elasticnet': LGBMClassifier( # could try L1 / L2 with this
        boosting_type='rf',
        bagging_fraction=0.7,
        feature_fraction=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5, 
        **forest_params
    ),
    'lgbm': LGBMClassifier(
        boosting_type='gbdt',
        **forest_params
    ),
    'lgbm_l1': LGBMClassifier(
        boosting_type='gbdt',
        reg_alpha=1.0,
        **forest_params
    ),
    'lgbm_l2': LGBMClassifier(
        boosting_type='gbdt',
        reg_lambda=1.0,
        **forest_params
    ),
    'lgbm_elasticnet': LGBMClassifier(
        boosting_type='gbdt',
        reg_alpha=0.5,
        reg_lambda=0.5,
        **forest_params
    ),
    'svm': SGDClassifier(
        loss='hinge', 
        penalty=None, 
        **linear_params
    ),
    'svm_l1': SGDClassifier(
        loss='hinge', 
        penalty='l1', 
        **linear_params
    ),
    'svm_l2': SGDClassifier(
        loss='hinge', 
        penalty='l2', 
        **linear_params
    ),
    'svm_elasticnet': SGDClassifier(
        loss='hinge', 
        penalty="elasticnet", 
        **linear_params
    ),
    'log': SGDClassifier(
        loss='log_loss', 
        penalty=None, 
        **linear_params
    ),
    'log_l1': SGDClassifier(
        loss='log_loss', 
        penalty='l1', 
        **linear_params
    ),
    'log_l2': SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        **linear_params
    ),
    'log_elasticnet': SGDClassifier(
        loss='log_loss',
        penalty="elasticnet",
        **linear_params
    ),
}

# Reviewer 2 (SIAMCAT, PMID 33785070): the linear models (SVM, LogReg) are scale-sensitive and were
# previously fitted directly on raw compositional MSP pseudocounts, which unfairly depressed their
# performance. Wrap every linear model in a normalization pipeline — log1p (variance-stabilising for
# compositional counts) followed by one scaler per family, the same in every arm: the SVM
# scales each SAMPLE to unit length (Normalizer, the scaling Reviewer 2 names for it), LogReg scales each
# FEATURE to z-scores (StandardScaler, the SIAMCAT default). The scaler
# is fitted only on training data inside each CV fold / locally per cohort, so no test information leaks
# (in the federated setting each client normalises its own data before sharing its model). Tree models
# (RF, LGBM) are invariant to monotonic feature scaling and are left on the untransformed pseudocount
# matrix, unchanged. Note the wording: both families see the same MSP pseudocounts described in
# Preprocessing; "raw" here means untransformed, never a separate raw-read-count matrix.
def _normalized_linear(model_type, clf):
    fam = str(model_type).split('_')[0]
    return Pipeline([
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', Normalizer(norm='l2') if fam == 'svm' else StandardScaler()),
        ('clf', clf),
    ])

estimators = {
    name: (_normalized_linear(name, est) if name.startswith(('svm', 'log')) else est)
    for name, est in estimators.items()
}


def _pin_random_state(est, random_state=RANDOM_STATE):
    """Pin the template's seed so a direct clone is reproducible.

    These are TEMPLATES: production always goes through get_estimator(), which overrides random_state per
    (model, rep, cohort), so pinning here changes no published number. It exists to close a footgun --
    left unset, random_state is None, and SGDClassifier then shuffles from numpy's *global* RNG, so any
    code that clones a template directly is silently nondeterministic. That cost us a bogus erratum: an
    audit script cloned a template and its linear numbers swung 0.065 AUROC between runs, which was
    briefly mistaken for a defect in the pipeline itself.
    """
    target = est.steps[-1][1] if isinstance(est, Pipeline) else est
    if hasattr(target, 'random_state'):
        target.random_state = random_state
    return est


estimators = {name: _pin_random_state(est) for name, est in estimators.items()}

def get_deterministic_random_state(model_type, rep=0, group=None, n_clients=None):
    """
    Generate a deterministic random state based on training context.
    
    Args:
        model_type: Type of model being trained
        rep: Current repetition number
        group: Group name (if applicable)
        n_clients: Number of clients (for federated learning)
        
    Returns:
        An integer to use as random_state
    """
    # Create a string representation of the context
    context = f"{model_type}_{rep}"
    if group is not None:
        context += f"_{group}"
    if n_clients is not None:
        context += f"_{n_clients}"
        
    # Stable hash -> deterministic seed. The builtin hash() is SALTED per interpreter (PYTHONHASHSEED),
    # so it gave a different "deterministic" seed on every invocation -- the whole run was only
    # reproducible if the caller happened to export PYTHONHASHSEED=0. hashlib is stable
    # across processes and runs, so no-flag reproducibility no longer depends on the environment.
    hash_value = int(hashlib.sha256(context.encode()).hexdigest(), 16) % 2147483647
    return (hash_value + RANDOM_STATE) % 2147483647

def get_estimator(type="rf", rep=0, group=None, n_clients=None, estimators=estimators):
    """
    Get an estimator by type with a deterministic random state.
    
    Args:
        type: The type of estimator to get
        rep: Current repetition number
        group: Group name (if applicable)
        n_clients: Number of clients (for federated learning)
        estimators: Dictionary of base estimators
        
    Returns:
        The estimator with a context-specific random state
    """
    # Get a copy of the base estimator
    estimator = clone(estimators[type])
    
    # Set a deterministic random state based on context
    random_state = get_deterministic_random_state(type, rep, group, n_clients)
    
    # Set the random_state parameter if the estimator supports it
    if hasattr(estimator, 'random_state'):
        estimator.random_state = random_state
    elif isinstance(estimator, Pipeline):
        # For normalized linear models, random_state lives on the final estimator in the pipeline.
        final_est = estimator.steps[-1][1]
        if hasattr(final_est, 'random_state'):
            final_est.random_state = random_state

    return estimator


# File paths
data_dir = './prep_data'
data_dir = Path(data_dir)

# Input data files
taxa_meteor = 'taxa_meteor.csv'
strain_meteor_nan = 'strain_meteor_nan.csv'
strain_meteor_imputed = 'strain_meteor_imputed.csv'
strain_taxa_meteor = 'strain_taxa_meteor.csv'
taxa_meta4 = 'taxa_meta4.csv'
taxa_sylph = 'taxa_sylph.csv'
taxa_counts_meteor_0 = 'taxa_counts_meteor_0.csv'

anno_all = 'anno_full_clean.csv'  # + remove NAN in metadata
anno_nan = 'anno_full_nan.csv'  # + NANs allowed in metadata


# Output directories
default_out = './data'
output_dir = default_out
save_dir = Path(os.path.join(output_dir, 'simulations'))
save_dir.mkdir(exist_ok=True, parents=True)

figure_dir = Path(os.path.join(save_dir, 'figures'))
figure_dir.mkdir(exist_ok=True, parents=True)

table_dir = Path(os.path.join(save_dir, 'tables'))
table_dir.mkdir(exist_ok=True, parents=True)

raw_results_dir = Path(os.path.join(save_dir, 'raw_results'))
raw_results_dir.mkdir(exist_ok=True, parents=True)


# Extract features for analysis
features = ['age', 'bmi', 'gender', 'health_status', 'country']


# Define experiments
experiments = [
    Experiment(name='count_strain_meteor_nan_0', anno=anno_nan, count=strain_meteor_nan, model="lgbm"),
    # Strain preprocessing selection (central AUC, 10 reps x 5 folds, same lgbm model):
    #   raw/NaN 0.847 > strain+taxa 0.831 > imputed 0.746.
    # LGBM splits on NaN natively, so an absent strain is signal; imputing it destroys that.
    # Both alternatives dropped from the sweep — re-enable to reproduce the comparison.
    # Experiment(name='count_strain_taxa_meteor_0', anno=anno_nan, count=strain_taxa_meteor, model="lgbm"),
    # Experiment(name='count_strain_imputed_meteor_0', anno=anno_nan, count=strain_meteor_imputed, model='lgbm'),
    Experiment(name='count_taxa_meteor_0', anno=anno_all, count=taxa_counts_meteor_0),
    # Experiment(name='count_taxa_metaphlan4_0', anno=anno_all, count=taxa_meta4),
    # Experiment(name='count_taxa_sylph_0', anno=anno_all, count=taxa_sylph),
    
        
    # Experiment(name= 'count_taxa_func_meteor_0',           anno= anno_all,        count= taxa_func_counts_0),
    # Experiment(name= 'count_func_meteor_0',                anno= anno_all,        count= func_counts_0),
    
    # All four sylph variants MUST share one annotation, or the 4-way model comparison is invalid.
    # rf/lgbm previously used anno_nan (1953 samples / 12 cohorts, incl. China4) while svm/log used
    # anno_all (1722 / 11) -- so the tree models were scored on +231 samples and an extra cohort.
    # anno_all is the annotation the headline taxa_meteor grid uses, so sylph is now comparable to it too.
    Experiment(name='count_taxa_sylph_0_rf', anno=anno_all, count=taxa_sylph, model='rf'),
    Experiment(name='count_taxa_sylph_0_lgbm', anno=anno_all, count=taxa_sylph, model='lgbm'),
    # MetaPhlAn4 profiles, same annotation as the headline taxa_meteor grid, so the profiler
    # sensitivity analysis (Data Selection) is comparable to it. lgbm only: it is the model the
    # main text reports.
    Experiment(name='count_taxa_metaphlan4_0_lgbm', anno=anno_all, count=taxa_meta4, model='lgbm'),
    # Supplementary sensitivity arm: the SAME taxa/METEOR2 counts on the permissive annotation
    # (anno_nan), which keeps the 231 samples with incomplete age/sex/BMI and therefore restores
    # China4 as a 12th training cohort (12 cohorts / 1820 training samples vs 11 / 1607). Only lgbm
    # can run it: the demographic columns are model features and LightGBM splits on NaN natively,
    # while the SGD linear models cannot ingest NaN. The name deliberately contains neither 'meteor'
    # nor 'count_taxa_meteor_0', so the canonical `--dataset meteor` run does not pick it up.
    Experiment(name='sensitivity_permissive_lgbm', anno=anno_nan, count=taxa_counts_meteor_0, model='lgbm'),
    Experiment(name='count_taxa_sylph_0_svm', anno=anno_all, count=taxa_sylph, model='svm'),
    Experiment(name='count_taxa_sylph_0_log', anno=anno_all, count=taxa_sylph, model='log'),
]
# if experiments have no explicit model, they will be executed with all models
experiments = propagate_experiments(experiments, estimators.keys())


def _prepare_simulation_data(anno_df, counts_df, label, groups):
    # Common preparations
    meta_features = [label, 'gender', 'age', 'bmi'] + groups
    

    # Merge counts and selected metadata
    anno_df = anno_df.loc[:, meta_features]
    central_data = pd.merge(anno_df, counts_df, left_index=True, right_index=True)
    
    # 'QC' and spliting dataset for reuse
    datasets = {}
    train_cohorts = []
    val_cohorts = []
    
    for cohort, _df in central_data.groupby(groups):
        cohort = cohort[0] if isinstance(cohort, tuple) else cohort
        _df.drop(columns=groups, inplace=True)
        
        # Only allow data for training with both classes and at least 40 (2*min_samples_leaf (lgbm)) samples
        if len(_df[label].unique()) > 1 and len(_df) > 40:
            train_cohorts.append(cohort)
        else:
            val_cohorts.append(cohort)  
            
        datasets[cohort] = (_df.drop(columns=[label]).to_numpy(), _df[label].to_numpy())
        
    feature_names = central_data.drop(columns=[label] + groups).columns.tolist()

    return central_data, train_cohorts, val_cohorts, feature_names, datasets
    
def _run_central_simulation(train_data, feature_names, label, groups, model, CROSS_VAL, scoring, feature_importance, random_state=RANDOM_STATE):
    """Run centralized simulation."""

    X = train_data.drop(columns=[label] + groups).to_numpy()
    y = train_data[label].to_numpy()
    
    # Create cross-validation folds
    kf = StratifiedKFold(n_splits=CROSS_VAL, shuffle=True, random_state=random_state)
    
    # Run cross-validation
    scores, importances = custom_cross_validate(
        model, X, y,
        features=feature_names,
        cv=kf, metrics=scoring,
        feature_importance=feature_importance
    )
    
    meta_inf = {
    "train_dataset": 'all',
    "val_dataset": 'all',
    'n_clients':1,
    }

    # update metadata if results are not empty (i.e. if importances were saved)
    for u in (scores, importances):
        for entry in u:
            if not entry:
                continue
            entry.update(meta_inf)

    # The C each outer fold chose, recorded like the pooled arm records its tuned rows.
    if isinstance(model, _NestedTunedCentralModel):
        for entry, hyperparams in zip(scores, model.hyperparams_):
            entry.update(tuned=True, hyperparams=hyperparams)

    return scores, importances

def _run_local_simulation(datasets, feature_names, train_cohorts, model, CROSS_VAL, scoring, feature_importance, rep=0):
    """Run local simulation.

    `model` is either a fitted-estimator template shared by every cohort, or a callable `grp -> estimator`
    so each centre can be given its own model (this is how --tune-local hands each cohort a
    `_SelfTunedLocalModel`).
    """

    # Run cross-validation for each group
    all_results = []
    all_importances = []

    for grp in train_cohorts:
        # Get precomputed X, y for this group
        X, y = datasets[grp]
        est = model(grp) if callable(model) else model

        # Per-rep, per-cohort seeded split so repetitions genuinely re-partition the data (real
        # error bars). Previously the split was seed-fixed (and passed as an int, so unshuffled),
        # making every rep identical for deterministic models. Pass the splitter, not an int.
        kf = StratifiedKFold(n_splits=CROSS_VAL, shuffle=True,
                             random_state=get_deterministic_random_state('cv_split', rep=rep, group=grp))

        # Run cross-validation
        scores, importances = custom_cross_validate(
            est, X, y,
            features=feature_names,
            cv=kf, metrics=scoring,
            feature_importance=feature_importance
        )

        meta_inf = {
            "train_dataset": grp,
            "val_dataset": grp,
            'n_clients':1,
            'tuned_local': isinstance(est, _SelfTunedLocalModel),
        }
        
        # update metadata if results are not empty (i.e. if importances were saved)
        for u in (scores, importances):
            for entry in u:
                if not entry:
                    continue
                entry.update(meta_inf)
                       
        all_results.extend(scores)
        all_importances.extend(importances)
    
    return all_results, all_importances

def _is_tree_model(model_type):
    """rf/lgbm *and* their regularised variants (rf_l1, lgbm_elasticnet, ...).

    The old check was a literal `model_type in ['rf','lgbm']`, so it matched 2 of the 8 tree models.
    """
    return str(model_type).split('_')[0] in ('rf', 'lgbm')


def _estimator_cache_key(rep, model_type, grp):
    return (rep, model_type, grp)


def _precompute_estimators(datasets, train_cohorts, model_type, repetition, cache=None,
                           client_tune_cache=None):
    """
    Precompute and cache estimators for reuse in federated learning simulations.
    
    Args:
        datasets: Dictionary mapping groups to datasets
        train_cohorts: List of valid groups
        model_type: Type of model to use
        repetition: Current repetition number
        cache: Dictionary to cache trained models (created if None)
        
    Returns:
        Dictionary of cached estimators
    """
    if cache is None:
        cache = {}

    for grp in train_cohorts:
        X, y = datasets[grp]

        # One model per (rep, model, cohort), reused at every k.
        cache_key = _estimator_cache_key(repetition, model_type, grp)

        if cache_key not in cache:
            estimator = _client_estimator(model_type, grp, datasets, repetition, client_tune_cache)
            estimator.fit(X, y)
            cache[cache_key] = estimator

    return cache


# ---------------------------------------------------------------------------------------------------
# Tuned baselines
#
# The centralized/pooled arms used the SAME fixed hyperparameters as the per-client models. A model
# trained on 10x more data needs different regularisation, and the linear pipeline's SGDClassifier is not
# a defensible baseline at pooled scale: it stops on tolerance at n_iter=148/1000, its regularisation
# strength (fixed alpha=1e-3) is worth 0.048 AUROC when tuned, and its solution swings 0.094 AUROC with
# the shuffle seed. Comparing a k-model *average* against a single high-variance fit is biased in the
# ensemble's favour before any federation happens.
#
# Every federated-vs-centralized number is untrustworthy until the baseline is fit properly: a federated
# method cannot beat centralized training on the same data, so a federation that wins points to a broken
# baseline.
# ---------------------------------------------------------------------------------------------------

TUNE = _C_FED['tune']   # config: [federated].tune ; CLI: --tune / --no-tune


def _penalty_of(model_type):
    """'log_l1' -> 'l1'; 'lgbm' -> None. The model name IS the penalty spec."""
    parts = str(model_type).split('_')
    return parts[1] if len(parts) > 1 and parts[1] in ('l1', 'l2', 'elasticnet') else None


def _converged_linear(model_type, C, random_state=None):
    """The same log1p -> scaler pipeline, but a solver that actually converges.

    Keeps the model's penalty (the name declares it) and varies only the regularisation STRENGTH, which
    is the axis that matters: C=0.001 -> 0.8058 vs C=1.0 -> 0.7583 on pooled LOCO.

    'elasticnet' has no converged hinge/liblinear solver, so those variants stay on SGD -- but with a
    real convergence budget (tol 1e-5, max_iter 50k) rather than the pipeline's tol=1e-3, and with alpha
    swept. That is still an honest baseline: fitted to convergence, regularisation tuned.

    `random_state` seeds the solvers that shuffle (liblinear, saga, SGD). The centralized arm passes the
    seeds of the tuned centralized script; the pooled arm and the federation clients pass the seed
    get_estimator() would give them.
    """
    fam = str(model_type).split('_')[0]
    penalty = _penalty_of(model_type) or 'l2'

    if fam == 'log':
        if penalty == 'l1':
            clf = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=5000,
                                     random_state=random_state)
        elif penalty == 'elasticnet':
            clf = LogisticRegression(C=C, penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                     max_iter=5000, random_state=random_state)
        else:
            clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=5000)
    elif fam == 'svm':
        if penalty == 'elasticnet':
            # No converged elasticnet hinge solver exists; give SGD a real budget instead of SGD's default.
            clf = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=1.0 / max(C, 1e-12),
                                max_iter=50000, tol=1e-5, learning_rate='optimal',
                                random_state=RANDOM_STATE if random_state is None else random_state,
                                n_jobs=THREADS_PER_JOB)
        elif penalty == 'l1':
            clf = LinearSVC(C=C, penalty='l1', loss='squared_hinge', dual=False, max_iter=20000,
                            random_state=random_state)
        else:
            clf = LinearSVC(C=C, penalty='l2', loss='squared_hinge', dual='auto', max_iter=20000,
                            random_state=random_state)
    else:
        raise ValueError(f"_converged_linear called with non-linear model {model_type!r}")

    return _normalized_linear(model_type, clf)


# Regularisation strength dominates; 4 decades is the range over which the pooled fit actually moves.
_LINEAR_GRID = [1e-3, 1e-2, 1e-1, 1.0]

# Trees. THE GRID MUST BE ABLE TO EXPRESS THE MODEL THAT WINS, or tuning is just a slower way of
# handicapping the baseline. Twice now it could not:
#
#   - it capped n_estimators at 300 while the federation holds k x 100 = 1000 trees;
#   - it never varied max_depth, and the pipeline pins max_depth=5 -- which for a BAGGING forest
#     (boosting_type='rf') is crippling. At depth 5 pooled rf scores 0.7875 and LOSES to the federation
#     (0.8004); freed to depth >= 8 it scores 0.8027-0.8050 and wins. The entire "federated rf beats
#     centralized" invariant violation was this one missing grid axis.
#
# Capacity itself is genuinely saturated (a capacity check, since removed, found pooled flat from 100 to
# 4000 trees), so
# the budget entries stay modest -- it is DEPTH that the grid was missing, not trees.
_TREE_GRID = [
    {'n_estimators': 100, 'learning_rate': 0.1},
    {'n_estimators': 300, 'learning_rate': 0.05},
    {'n_estimators': 100, 'num_leaves': 7},
    {'n_estimators': 300, 'num_leaves': 7, 'learning_rate': 0.05},
    {'n_estimators': 300, 'max_depth': 8},
    {'n_estimators': 300, 'max_depth': 12},
    {'n_estimators': 300, 'max_depth': -1},          # -1 = unbounded, LightGBM's own default
    {'n_estimators': 1000, 'max_depth': -1, 'learning_rate': 0.05},
]


def _candidate_factories(model_type, rep):
    """[(label, () -> unfitted estimator)] for the inner search. Empty list => nothing to tune."""
    if _is_tree_model(model_type):
        out = []
        for params in _TREE_GRID:
            usable = {k: v for k, v in params.items()
                      if k != 'learning_rate' or getattr(estimators[model_type], 'boosting_type', '') != 'rf'}

            def make(p=usable):
                est = get_estimator(model_type, rep=rep)
                est.set_params(**p)
                return est

            out.append((str(usable), make))
        return out

    if str(model_type).split('_')[0] in ('log', 'svm'):
        seed = get_deterministic_random_state(model_type, rep)
        return [(f'C={C}', partial(_converged_linear, model_type, C, random_state=seed))
                for C in _LINEAR_GRID]

    return []


def _intra_cohort_cv_auc(factory, X, y):
    """Mean AUROC of stratified CV *within one cohort* -- exactly what a client can compute alone.

    Zero disclosure: no data, no scores and no model leave the client. This is the honest instrument for
    a client choosing its OWN hyperparameters, and it is enough: local CV rejects the pipeline's
    min_data_in_leaf=2 in all 11 cohorts, and selecting on it recovers +0.062 AUROC for the federation --
    matching a peer protocol that would have cost an extra communication round.
    """
    if len(np.unique(y)) < 2 or len(y) < 10:
        return float('-inf')
    # n_splits cannot exceed the rarest class's count, and StratifiedKFold needs at least 2.
    min_class = int(np.bincount(y.astype(int)).min())
    if min_class < 2:
        return float('-inf')

    skf = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        try:
            m = factory().fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[va], _continuous_scores(m, X[va])))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else float('-inf')


def _inner_loco_auc(factory, comb, datasets):
    """Mean AUROC of leave-one-COHORT-out *inside the training combination*.

    LEAKAGE BOUNDARY: the score is computed only from cohorts in `comb`. No held-out cohort is read, so
    the selected hyperparameter carries no information about the data it will be evaluated on. Selecting
    on the held-out cohort instead is worth ~+0.036 AUROC of pure fiction.

    At k=1 there is no peer cohort to validate against, so it falls back to CV within the single cohort --
    still strictly inside `comb`.
    """
    scores = []

    if len(comb) < 2:
        X, y = datasets[comb[0]]
        return _intra_cohort_cv_auc(factory, X, y)

    for held in comb:
        inner = [c for c in comb if c != held]
        X = np.vstack([datasets[c][0] for c in inner])
        y = np.concatenate([datasets[c][1] for c in inner])
        X_v, y_v = datasets[held]
        if len(np.unique(y)) < 2 or len(np.unique(y_v)) < 2:
            continue
        try:
            m = factory().fit(X, y)
            scores.append(roc_auc_score(y_v, _continuous_scores(m, X_v)))
        except Exception:
            continue

    return float(np.mean(scores)) if scores else float('-inf')


# ---- Persistent tuning cache -----------------------------------------------------------------
# Persists the WINNING hyperparameter per tuning key so a repeated run skips the (expensive) grid
# search. Only serialisable primitives are stored -- the pooled/client label string, or the boosting
# leaf int -- and the unfitted estimator is rebuilt from that primitive, so a cache HIT yields the
# byte-for-byte same model a fresh search would have crowned (the search only SELECTS a candidate; it
# never alters it). Results are therefore identical with the cache on or off; the cache only removes
# the search work. Everything is BEST-EFFORT: any I/O, parse or lookup failure silently falls back to
# a full search, so the cache can never change a result or break a run.
#
# ON BY DEFAULT, writing to data/simulations/tune_cache. Control via env FEMAI_TUNE_CACHE:
#   unset / '1' / 'on'            -> the default dir data/simulations/tune_cache
#   '0' / 'off' / 'false' / 'no'  -> DISABLED (the exact pre-cache code path; results unchanged either way)
#   any other value               -> use it verbatim as the cache directory path
_env_tc = (os.environ.get('FEMAI_TUNE_CACHE', '') or '').strip()
if _env_tc.lower() in ('0', 'off', 'false', 'no'):
    _TUNE_CACHE_DIR = None
elif _env_tc == '' or _env_tc.lower() in ('1', 'true', 'on', 'yes'):
    _TUNE_CACHE_DIR = os.path.join('data', 'simulations', 'tune_cache')
else:
    _TUNE_CACHE_DIR = _env_tc


def _tune_fingerprint(train_cohorts, datasets):
    """A short hash of everything a cached hyperparameter depends on. If any of it changes, the stored
    labels are stale and the cache for this experiment is ignored (re-searched from scratch)."""
    parts = [f'{c}:{len(datasets[c][1])}x{datasets[c][0].shape[1]}' for c in sorted(map(str, train_cohorts))]
    parts += [f'TREE={_TREE_GRID}', f'LIN={_LINEAR_GRID}', f'LEAF={_CLIENT_LEAF_GRID}',
              f'TUNE={TUNE}', f'TUNE_CLIENTS={TUNE_CLIENTS}',
              f'MG={MERGE_GERMANY}', f'SI={SPLIT_ITALY}']
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]


def _tune_cache_load(kind, analysis_name, fingerprint):
    """Best-effort load of {str_key: primitive} for one experiment+kind; {} on any miss/mismatch."""
    if not _TUNE_CACHE_DIR or not fingerprint:
        return {}
    try:
        p = os.path.join(_TUNE_CACHE_DIR, f'{analysis_name}__{kind}.json')
        if not os.path.exists(p):
            return {}
        with open(p) as fh:
            obj = json.load(fh)
        if obj.get('__fp__') != fingerprint:
            return {}
        return dict(obj.get('entries', {}))
    except Exception:
        return {}


def _tune_cache_save(kind, analysis_name, fingerprint, entries):
    """Best-effort atomic write of the populated cache; silent on any failure."""
    if not _TUNE_CACHE_DIR or not fingerprint or not entries:
        return
    tmp = None
    try:
        os.makedirs(_TUNE_CACHE_DIR, exist_ok=True)
        p = os.path.join(_TUNE_CACHE_DIR, f'{analysis_name}__{kind}.json')
        tmp = f'{p}.tmp{os.getpid()}'
        with open(tmp, 'w') as fh:
            json.dump({'__fp__': fingerprint, 'entries': dict(entries)}, fh, sort_keys=True, indent=1)
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _pooled_factory_for_label(model_type, rep, label):
    """Rebuild the pooled candidate whose label matches; None if the grid no longer has it (stale)."""
    for lbl, f in _candidate_factories(model_type, rep):
        if lbl == label:
            return f
    return None


def _client_factory_for_label(model_type, rep, grp, label):
    """Rebuild the client candidate whose label matches; None if stale."""
    for lbl, f in _client_candidates(model_type, rep, grp):
        if lbl == label:
            return f
    return None


def _tuned_estimator(model_type, comb, datasets, rep, tune_cache=None):
    """Pick pooled hyperparameters by nested LOCO, then return the UNFITTED winner.

    Returns the plain get_estimator() model when TUNE is off, so the untuned path is untouched.

    COST: the search is cached per (rep, model, **k**) -- tuned ONCE per client-count on the first
    combination of that size, and reused for every other combination of the same size. Per-combination
    tuning (the strict version) is ~C(11,k)x more expensive and does NOT finish in practice: it timed out
    after 9 h on one model. The approximation is sound because the pooled hyperparameter that matters --
    regularisation strength / depth -- depends on the training-set *size* (i.e. k), not on *which* k
    cohorts (the fixed alpha was wrong because the pool was big, not because of its membership).
    Mild leakage note: the config for size k is chosen on the first k-combination, so a later combination
    may reuse a config selected with one of its held-out cohorts in the inner-LOCO. This is a single
    coarse model-class choice (which depth), not per-sample information, and is how a centralized baseline
    is normally tuned -- state it in the Methods as "selected by LOCO CV over the training cohorts".

    Linear models instead mirror the federation clients, so the two differ only in being federated or
    pooled: C is chosen by the clients' own
    search, _intra_cohort_cv_auc, run on the pooled training data of each combination. The search is
    deterministic up to the solver seed, so it runs once per combination and every rep reuses its C.
    """
    if not TUNE:
        return get_estimator(model_type, rep=rep, group='+'.join(map(str, comb)))

    linear = str(model_type).split('_')[0] in ('log', 'svm')
    if linear:
        key = f'{model_type}|{"+".join(map(str, sorted(comb)))}'   # once per combination, shared by reps
    else:
        key = f'{rep}|{model_type}|{len(comb)}'     # once per client-count, not per combination
    if tune_cache is not None and key in tune_cache:
        factory = _pooled_factory_for_label(model_type, rep, tune_cache[key])
        if factory is not None:
            return factory()
        # Stored label is no longer in the current grid -> fall through and re-search.

    cands = _candidate_factories(model_type, rep)
    if not cands:
        return get_estimator(model_type, rep=rep, group='+'.join(map(str, comb)))

    if linear:
        X = np.vstack([datasets[g][0] for g in comb])
        y = np.concatenate([datasets[g][1] for g in comb])
        scored = [(_intra_cohort_cv_auc(f, X, y), label, f) for label, f in cands]
    else:
        scored = [(_inner_loco_auc(f, comb, datasets), label, f) for label, f in cands]
    best_score, best_label, best_factory = max(scored, key=lambda t: t[0])

    if best_score == float('-inf'):
        # Nothing was scorable (e.g. a single-cohort combination too small or too class-skewed to split).
        # Fall back to the untuned default rather than silently crowning an arbitrary grid point.
        return get_estimator(model_type, rep=rep, group='+'.join(map(str, comb)))

    if tune_cache is not None:
        tune_cache[key] = best_label     # store only the label; the factory is rebuilt on a hit
    return best_factory()


TUNE_CLIENTS = _C_FED['tune_clients']   # config: [federated].tune_clients ; CLI: --tune-clients / --no-tune-clients

# Leaf size is THE client hyperparameter. The pipeline pins min_data_in_leaf=2: on a
# 44-sample cohort that is a memorised pair, and it costs the FEDERATION ~0.065 AUROC while costing the
# pooled model nothing (pooled has ~1500 samples, so leaves of 2 still generalise). The clients were never
# tuned at all.
_CLIENT_LEAF_GRID = [2, 5, 10, 20, 40]


def _client_candidates(model_type, rep, group):
    """[(label, () -> unfitted estimator)] a client can choose between using ONLY its own data."""
    if _is_tree_model(model_type):
        def make_tree(L):
            def f():
                e = get_estimator(model_type, rep=rep, group=group)
                # Set both names: min_data_in_leaf is LightGBM's alias, min_child_samples the sklearn
                # param. Setting one leaves the other in place and LightGBM silently honours the alias.
                e.set_params(min_data_in_leaf=L, min_child_samples=L)
                return e
            return f
        return [(f'min_leaf={L}', make_tree(L)) for L in _CLIENT_LEAF_GRID]

    if str(model_type).split('_')[0] in ('log', 'svm'):
        # The same model as the pooled baseline: a converged solver, C from _LINEAR_GRID.
        seed = get_deterministic_random_state(model_type, rep, group)
        return [(f'C={C}', partial(_converged_linear, model_type, C, random_state=seed))
                for C in _LINEAR_GRID]

    return []


def _client_estimator(model_type, grp, datasets, rep, cache=None):
    """The model a client contributes to the federation.

    With --tune-clients, the client selects its own hyperparameters by cross-validation ON ITS OWN COHORT.
    This is a METHOD CHANGE to the federated arm, not a bug fix, and must be declared as one -- but it is
    the *cheapest* possible one: zero disclosure (nothing leaves the client), no extra communication round
    (the protocol stays strictly one-shot), and worth +0.062 AUROC.

    A peer protocol (leave-one-client-out) was tested and does NOT do better (-0.0016), so it is not worth
    the extra round (a check run during the revision, since removed).
    """
    if not TUNE_CLIENTS:
        return get_estimator(model_type, rep=rep, group=grp)

    key = f'{rep}|{model_type}|{grp}'
    if cache is not None and key in cache:
        factory = _client_factory_for_label(model_type, rep, grp, cache[key])
        if factory is not None:
            return factory()
        # Stored label is no longer in the current grid -> fall through and re-search.

    cands = _client_candidates(model_type, rep, grp)
    if not cands:
        return get_estimator(model_type, rep=rep, group=grp)

    X, y = datasets[grp]
    scored = [(_intra_cohort_cv_auc(f, X, y), label, f) for label, f in cands]
    best_score, best_label, best_factory = max(scored, key=lambda t: t[0])
    if best_score == float('-inf'):
        return get_estimator(model_type, rep=rep, group=grp)

    if cache is not None:
        cache[key] = best_label     # store only the label; the factory is rebuilt on a hit
    return best_factory()


TUNE_LOCAL = _C_EXP.get('tune_local', True)   # config: [experiment].tune_local ; CLI: --tune-local / --no-tune-local


class _SelfTunedLocalModel:
    """A centre going it alone that picks its own hyperparameters -- the like-for-like counterpart of a
    tuned federation client.

    Without this, arm (B) is the only arm still carrying the pipeline default `min_data_in_leaf=2` while
    every federated client selects over `_CLIENT_LEAF_GRID` and rejects 2 in all 55 cached selections. A
    federated-versus-local comparison built on that is a comparison against a configuration the method
    itself discarded, not against a centre doing its best alone.

    The selection runs inside `fit()`, over the training rows handed to it, using the same candidate grid
    (`_client_candidates`) and the same instrument (`_intra_cohort_cv_auc`) a client uses. Because the
    outer CV split is applied first, the evaluation fold never enters the choice -- so this is a nested
    estimate, not a tuned-on-the-test-fold one. That distinction is the whole reason the wrapper exists
    rather than a call to `_client_estimator`, which selects on the full cohort.
    """

    def __init__(self, model_type, rep, group):
        self.model_type = model_type
        self.rep = rep
        self.group = group
        self.estimator_ = None
        self.best_label_ = None

    def fit(self, X, y):
        factory, label = None, None
        cands = _client_candidates(self.model_type, self.rep, self.group)
        if cands:
            score, label, factory = max(
                ((_intra_cohort_cv_auc(f, X, y), lab, f) for lab, f in cands), key=lambda t: t[0])
            if score == float('-inf'):
                factory, label = None, None
        if factory is None:
            # Too small or too class-skewed to split the training part -- fall back to the pipeline
            # default rather than crowning an arbitrary grid point.
            factory = lambda: get_estimator(self.model_type, rep=self.rep, group=self.group)
            label = 'default'
        self.best_label_ = label
        self.estimator_ = factory().fit(X, y)
        return self

    def __getattr__(self, name):
        # Reached only for attributes this wrapper does not define. Delegating to the fitted model means
        # the wrapper advertises predict_proba / decision_function exactly when the underlying model
        # does, which is what custom_cross_validate branches on.
        if name.startswith('__'):
            raise AttributeError(name)
        est = self.__dict__.get('estimator_')
        if est is None:
            raise AttributeError(name)
        return getattr(est, name)


class _NestedTunedCentralModel:
    """Centralized evaluation of a penalised linear model, as in every other evaluation: C is chosen from
    `_LINEAR_GRID` by a CV nested inside each outer training fold, so the outer test fold never enters the
    choice.

    Ported from tuned_central.py with its folds, seeds and grid, so it reproduces the
    published tunedA tables fold by fold. custom_cross_validate fits one instance once per outer fold, in
    fold order, so the fit count is the fold index: the inner split is seeded `outer_seed + 1000 + fold`.
    The inner candidates are seeded RANDOM_STATE and the final fit RANDOM_STATE + rep. A C whose inner
    fits all fail is skipped, and a tie keeps the smaller C.
    """

    def __init__(self, model_type, rep, outer_seed, inner_folds):
        self.model_type = model_type
        self.rep = rep
        self.outer_seed = outer_seed
        self.inner_folds = inner_folds
        self.fold_ = 0
        self.estimator_ = None
        self.hyperparams_ = []    # one _describe_estimator() record per outer fold

    def fit(self, X, y):
        inner = StratifiedKFold(n_splits=self.inner_folds, shuffle=True,
                                random_state=self.outer_seed + 1000 + self.fold_)
        best_c, best_score = None, -np.inf
        for C in _LINEAR_GRID:
            scores = []
            for tr, va in inner.split(X, y):
                try:
                    m = _converged_linear(self.model_type, C, random_state=RANDOM_STATE).fit(X[tr], y[tr])
                    scores.append(roc_auc_score(y[va], _continuous_scores(m, X[va])))
                except Exception:
                    continue
            if scores and np.mean(scores) > best_score:
                best_score, best_c = np.mean(scores), C
        self.estimator_ = _converged_linear(self.model_type, best_c,
                                            random_state=RANDOM_STATE + self.rep).fit(X, y)
        self.hyperparams_.append(_describe_estimator(self.estimator_))
        self.fold_ += 1
        return self

    def __getattr__(self, name):
        # As in _SelfTunedLocalModel: expose predict_proba / decision_function exactly when the fitted
        # model does, which is what custom_cross_validate branches on.
        if name.startswith('__'):
            raise AttributeError(name)
        est = self.__dict__.get('estimator_')
        if est is None:
            raise AttributeError(name)
        return getattr(est, name)


def _member_transfer_loco(factory, comb, datasets):
    """Mean cross-cohort transfer of a per-cohort member at the candidate hyperparameter.

    Trains one member on ONE cohort in `comb` and scores it on every OTHER cohort in `comb`, averaged.
    This is exactly what a boosting member does -- it is fitted on a single cohort's rows -- so it
    selects the leaf size at which members STOP memorising their own cohort. Cheap: each
    member is fitted once, not per inner fold.

    LEAKAGE BOUNDARY: only cohorts in `comb` are ever read. No held-out test cohort touches the choice.
    """
    if len(comb) < 2:
        X, y = datasets[comb[0]]
        return _intra_cohort_cv_auc(factory, X, y)

    scores = []
    for train_c in comb:
        Xt, yt = datasets[train_c]
        if len(np.unique(yt)) < 2:
            continue
        try:
            m = factory().fit(Xt, yt)
        except Exception:
            continue
        for eval_c in comb:
            if eval_c == train_c:
                continue
            Xe, ye = datasets[eval_c]
            if len(np.unique(ye)) < 2:
                continue
            try:
                scores.append(roc_auc_score(ye, _continuous_scores(m, Xe)))
            except Exception:
                continue
    return float(np.mean(scores)) if scores else float('-inf')


def _tuned_multishot_template(model, comb, datasets, rep, cache=None):
    """ONE global hyperparameter config for the shared-model (boosting) arms.

    The one-shot voting arm tunes each client independently -- fine, because it combines *independent*
    complete models by averaging their outputs. The boosting arms (cyclic, fedboost) grow ONE shared
    model, so per-client leaf sizes would be an incoherent patchwork and per-client learning rates would
    break boosting convergence. A shared model gets ONE config, chosen server-side.

    Still leakage-free: selected by leave-one-cohort-out INSIDE `comb`, never touching a held-out test
    cohort. Only `min_data_in_leaf` is tuned -- the axis on which a member memorises its cohort -- because
    each contributing model, in cyclic and fedboost alike, is grown on a single cohort's rows and can
    memorise it. Learning rate is deliberately held fixed and global: it must be, for the aggregated
    boosting to be coherent.

    Returns the (unfitted) template with the chosen leaf set. Reduces to the plain template when tuning
    is off, so the untuned path is byte-for-byte unchanged.
    """
    if not TUNE_CLIENTS or not _is_tree_model(model):
        return get_estimator(model, rep=rep)

    key = f'{rep}|{model}|{"+".join(map(str, sorted(comb)))}'
    if cache is not None and key in cache:
        best_leaf = int(cache[key])     # JSON round-trips ints as ints; coerce defensively
    else:
        best_leaf, best_score = None, float('-inf')
        for L in _CLIENT_LEAF_GRID:
            def make(L=L):
                e = get_estimator(model, rep=rep)
                e.set_params(min_data_in_leaf=L, min_child_samples=L)
                return e
            s = _member_transfer_loco(make, comb, datasets)
            if s > best_score:
                best_leaf, best_score = L, s
        if best_leaf is None:
            best_leaf = get_estimator(model, rep=rep).get_params().get('min_data_in_leaf', 2)
        if cache is not None:
            cache[key] = best_leaf

    tmpl = get_estimator(model, rep=rep)
    tmpl.set_params(min_data_in_leaf=best_leaf, min_child_samples=best_leaf)
    return tmpl


def _describe_estimator(est):
    """Compact record of what was actually fitted, so a tuned row can be audited after the fact.

    A tuned run that does not say WHICH hyperparameters won is not reproducible, and the selection is
    itself a result: if the inner LOCO picks C=0.001 everywhere, that is worth knowing.
    """
    clf = est.steps[-1][1] if isinstance(est, Pipeline) else est
    keep = ('C', 'alpha', 'penalty', 'solver', 'l1_ratio', 'n_estimators', 'num_leaves',
            'learning_rate', 'max_depth', 'boosting_type')
    params = clf.get_params()
    return f"{clf.__class__.__name__}(" + ", ".join(
        f"{k}={params[k]}" for k in keep if k in params and params[k] is not None) + ")"


def _continuous_scores(estimator, X):
    """Continuous positive-class score for AUROC: predict_proba if available, else the margin."""
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(X)[:, 1]
    return np.asarray(estimator.decision_function(X)).ravel()


def _build_weights(w, comb, datasets):
    """Client weights for one combination; None means uniform.

    LEAKAGE RULE: a weight may only depend on the clients *in `comb`*. Anything indexed by a held-out
    cohort fabricates performance -- weighting by the evaluated cohort's own result is worth ~+0.036
    AUROC, roughly 9x the size of the real size-weighting effect.
    """
    if w in ('none', None):
        return None

    if w == 'size':
        total = sum(len(datasets[g][1]) for g in comb)
        return [len(datasets[g][1]) / total for g in comb]

    raise ValueError(f"Unknown weighting: {w!r}")

def _run_federated_simulation(datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                             scoring, feature_importance, weighting, rep, analysis_name, dataset_name, estimator_cache,
                             max_combos_per_k=None, voting=None, pred_cache=None):
    """
    Run federated simulation where multiple pre-trained models are combined.
    
    Args:
        datasets: Dictionary mapping groups to datasets
        feature_names: Array containing all feature names
        train_cohorts: List of training groups
        val_cohorts: List of validation groups
        n_clients: Number of clients to use
        label: Column name for label
        model: Model type to use
        scoring: Scoring metrics to use
        feature_importance: Method for calculating feature importance
        weighting: List of weighting methods to try
        rep: Current repetition number
        estimator_cache: Cache of pre-trained estimators
        
    Returns:
        list: List of (results, feature importances) tuples
    """
    results = []
    voting = voting or VOTING

    # Hinge-loss SGD models have no predict_proba. svm_l1 was missing from this list and was rescued
    # only by the hasattr() fallback below; had that been removed, PFVC.fit would have raised and the
    # blanket `except Exception` around this call would have silently dropped svm_l1 from the results.
    supports_predict_proba = model not in ('svm', 'svm_l1', 'svm_l2', 'svm_elasticnet')

    # For each combination of client groups
    all_combs = list(combinations(train_cohorts, n_clients))
    # Optional cost control: when the number of k-cohort combinations is very large, evaluate a
    # deterministic random subsample. Panel D estimates the *mean* performance over all k-combinations,
    # which a random subsample estimates without bias. Set max_combos_per_k=None to use every combination.
    if max_combos_per_k is not None and len(all_combs) > max_combos_per_k:
        rng = np.random.default_rng(RANDOM_STATE + rep * 1000 + n_clients)
        sel = sorted(rng.choice(len(all_combs), size=max_combos_per_k, replace=False))
        all_combs = [all_combs[i] for i in sel]
    for comb in all_combs:
        test_data = [_ for _ in train_cohorts if _ not in comb]
        test_data.extend(val_cohorts)
        
        # Get pre-trained models from cache
        prefitted_estimators = {}
        for grp in comb:
            cache_key = _estimator_cache_key(rep, model, grp)

            if estimator_cache and cache_key in estimator_cache:
                prefitted_estimators[grp] = estimator_cache[cache_key]
            else:
                # Fallback if not in cache (shouldn't happen with proper precomputing)
                print(f"Warning: Model for {grp} not found in cache. Training on-demand.")
                X, y = datasets[grp]
                prefitted_estimators[grp] = get_estimator(model, rep=rep, group=grp).fit(X, y)
            
        # Identity of each member's *fitted model*, reused as the identity of its predictions.
        member_keys = [_estimator_cache_key(rep, model, grp) for grp in comb]

        for w in weighting:
            # Build the federated model with the current weighting method. Estimator order and the
            # weight vector must stay aligned -- both follow `comb`.
            combined_estimators = [
                (f"{grp}_{prefitted_estimators[grp].__class__.__name__}", prefitted_estimators[grp])
                for grp in comb
            ]
            combined_weights = _build_weights(w, comb, datasets)

            # Determine if all estimators in the combination support predict_proba
            can_use_soft_voting = supports_predict_proba
            if can_use_soft_voting:
                # Double-check that all estimators have predict_proba
                for _, est in combined_estimators:
                    if not hasattr(est, 'predict_proba'):
                        can_use_soft_voting = False
                        break

            # 'auto' preserves the historical behaviour (soft iff members expose predict_proba).
            # Note this makes voting perfectly confounded with model family (soft <-> trees,
            # hard <-> hinge SVM), so an explicit setting is required to compare voting rules.
            if voting == 'auto':
                active_voting = 'soft' if can_use_soft_voting else 'hard'
            elif voting == 'soft' and not can_use_soft_voting:
                active_voting = 'hard'   # members have no predict_proba; cannot average probabilities
            else:
                active_voting = voting

            # Create and fit the voting classifier
            combined_estimator = PreFittedVotingClassifier(
                estimators=combined_estimators,
                weights=combined_weights,
                voting=active_voting,
                pred_cache=pred_cache,
                member_keys=member_keys,
            )

            # Does not refit. Simply sets all necessary variables
            combined_estimator.fit()
            
            if combined_estimator is None:
                print(f"Skipping combination {comb} with weighting {w}: Failed to build federated model.")
                continue
            
            # Evaluate on test data
            for grp in test_data:
                X_test, y_test = datasets[grp]
                # Identifies the evaluation batch for the prediction memo. Every member's output on
                # this cohort is computed once for the whole sweep and reused by every later
                # (combination, weighting) that contains that member.
                combined_estimator.batch_key = grp
                res, f_imp = custom_comb_validate(
                    combined_estimator, X_test, y_test, 
                    features=feature_names, 
                    metrics=scoring, 
                    feature_importance=feature_importance
                )
                
                # Add metadata
                meta_inf = {
                    "rep": rep, 
                    "train_dataset": ', '.join(map(str, comb)),
                    "val_dataset": grp, 
                    'analysis': analysis_name,
                    "cv": 0, 
                    'n_clients': n_clients, 
                    'model': model,
                    'weighting': w,
                    'voting': active_voting,
                    'score_transform': 'none',
                    'dataset': dataset_name,
                    'simulation_type' : 'federated',
                    # A tuned-client and an untuned-client federation are different METHODS. Never pool them.
                    'tuned_clients': bool(TUNE_CLIENTS),
                }
                
                res.update(meta_inf)
                if f_imp:
                    f_imp.update(meta_inf)
                results.append((res, f_imp))
    
    return results

def _run_pooled_simulation(datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                           scoring, feature_importance, rep, analysis_name, dataset_name,
                           max_combos_per_k=None, tune_cache=None):
    """Centralised baseline at MATCHED k -- the like-for-like control for the federated arm.

    Pools the same k cohorts into one training set and evaluates on the same held-out cohorts, over the
    same combinations. federated(k) vs pooled(k) is then a paired, apples-to-apples contrast.

    Why this is needed: the existing `central` arm is an in-distribution StratifiedKFold over ALL pooled
    samples with no cohort awareness -- same-cohort samples sit in train and test, there is no domain
    shift, and it even includes cohorts the federated arm never trains on. Comparing it to the federated
    arm (strict cross-cohort external validation) confounds *aggregation* with *evaluation protocol*;
    the protocol gap alone is worth ~0.09 AUC. So "federated matches centralized" is not measurable
    against `central`, only against this.

    Unlike the federated arm, this DOES retrain: one model per combination.
    """
    results = []

    all_combs = list(combinations(train_cohorts, n_clients))
    # Same seed as _run_federated_simulation -> the identical subsample of combinations, so the two arms
    # are compared on the same cells and the subsampling noise cancels.
    if max_combos_per_k is not None and len(all_combs) > max_combos_per_k:
        rng = np.random.default_rng(RANDOM_STATE + rep * 1000 + n_clients)
        sel = sorted(rng.choice(len(all_combs), size=max_combos_per_k, replace=False))
        all_combs = [all_combs[i] for i in sel]

    for comb in all_combs:
        test_data = [c for c in train_cohorts if c not in comb]
        test_data.extend(val_cohorts)

        X_pool = np.vstack([datasets[g][0] for g in comb])
        y_pool = np.concatenate([datasets[g][1] for g in comb])
        if len(np.unique(y_pool)) < 2:
            continue

        # With --tune, hyperparameters are selected by leave-one-cohort-out INSIDE `comb` -- never using
        # a held-out cohort. Without it, this is the original fixed-hyperparameter estimator.
        estimator = _tuned_estimator(model, comb, datasets, rep, tune_cache)
        try:
            estimator.fit(X_pool, y_pool)
        except Exception as e:
            print(f"pooled: fit failed for {comb}: {e}")
            continue

        for grp in test_data:
            X_test, y_test = datasets[grp]
            res, f_imp = custom_comb_validate(
                estimator, X_test, y_test,
                features=feature_names, metrics=scoring, feature_importance=feature_importance
            )
            meta_inf = {
                "rep": rep,
                "train_dataset": ', '.join(map(str, comb)),
                "val_dataset": grp,
                'analysis': analysis_name,
                "cv": 0,
                'n_clients': n_clients,
                'model': model,
                'weighting': 'pooled',
                'voting': 'pooled',
                'score_transform': 'none',
                'dataset': dataset_name,
                'simulation_type': 'pooled',
                # Which baseline this row actually is. A tuned and an untuned pooled row are different
                # experiments and must never be pooled into one mean.
                'tuned': bool(TUNE),
                'hyperparams': _describe_estimator(estimator),
            }
            res.update(meta_inf)
            if f_imp:
                f_imp.update(meta_inf)
            results.append((res, f_imp))

    return results


def _run_cyclic_simulation(datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                           scoring, feature_importance, rep, analysis_name, dataset_name,
                           max_combos_per_k=None, rounds=3, multishot_tune_cache=None):
    """Cyclic (sequential) federated boosting -- a genuine MULTI-ROUND federation.

    The k clients pass one growing booster around the ring: each client continues boosting the *shared*
    global model on its own rows (LightGBM `init_model`), for R rounds. Unlike the one-shot arm this is
    iterative, and unlike exact histogram-based federated GBDT it still exchanges ONLY fitted models --
    no gradients, no per-node histograms -- so the paper's disclosure story is unchanged.
    It is the documented "cyclic" baseline in both NVFlare and Flower.

    Tree budget: each visit appends `n_estimators` trees, so a naive loop would give the federation
    R*k*100 trees against pooled's 100 -- measuring ensemble size, not federation strategy. We therefore
    split the SAME total budget across all visits.

    gbdt only: boosting_type='rf' raises `Check failed: init_score == nullptr` with init_model, and the
    caller's broad `except` would swallow it and silently emit zero rows.
    """
    results = []

    if str(model).split('_')[0] != 'lgbm':
        return results   # gbdt continuation only

    all_combs = list(combinations(train_cohorts, n_clients))
    # Same seed as the federated/pooled arms -> identical combinations -> paired comparison.
    if max_combos_per_k is not None and len(all_combs) > max_combos_per_k:
        rng = np.random.default_rng(RANDOM_STATE + rep * 1000 + n_clients)
        sel = sorted(rng.choice(len(all_combs), size=max_combos_per_k, replace=False))
        all_combs = [all_combs[i] for i in sel]

    total_trees = getattr(get_estimator(model, rep=rep), 'n_estimators', 100) or 100

    for comb in all_combs:
        test_data = [c for c in train_cohorts if c not in comb]
        test_data.extend(val_cohorts)

        k = len(comb)
        trees_per_visit = max(1, total_trees // (rounds * k))

        # One global leaf size for the shared booster, tuned server-side by LOCO inside comb when
        # --tune-clients is on (else the untuned template). See _tuned_multishot_template.
        template = _tuned_multishot_template(model, comb, datasets, rep, multishot_tune_cache)
        tuned_leaf = template.get_params().get('min_data_in_leaf')

        # Cyclic boosting is order-dependent (the last cohort visited fits the final residuals).
        # Permute per (rep, comb) so the reported spread includes order sensitivity rather than
        # baking in one arbitrary ring order.
        # Stable per-combination seed (builtin hash() is salted; see get_deterministic_random_state).
        comb_seed = int(hashlib.sha256(str(comb).encode()).hexdigest(), 16) % 100000
        order_rng = np.random.default_rng(RANDOM_STATE + rep * 7919 + comb_seed)
        ring = list(comb)
        order_rng.shuffle(ring)

        booster = None
        try:
            for _r in range(rounds):
                for grp in ring:
                    X, y = datasets[grp]
                    est = clone(template)
                    est.set_params(n_estimators=trees_per_visit)
                    est.fit(X, y, init_model=booster)
                    booster = est.booster_          # carries the accumulated trees to the next client
            final = est
        except Exception as e:
            print(f"cyclic: failed for {comb} (r={rounds}): {type(e).__name__}: {e}")
            continue

        for grp in test_data:
            X_test, y_test = datasets[grp]
            res, f_imp = custom_comb_validate(
                final, X_test, y_test,
                features=feature_names, metrics=scoring, feature_importance=feature_importance
            )
            meta_inf = {
                "rep": rep,
                "train_dataset": ', '.join(map(str, comb)),
                "val_dataset": grp,
                'analysis': analysis_name,
                "cv": 0,
                'n_clients': n_clients,
                'model': model,
                'weighting': 'cyclic',
                'voting': 'cyclic',
                'score_transform': 'none',
                'rounds': rounds,
                'trees_per_visit': trees_per_visit,
                'total_trees': trees_per_visit * rounds * k,
                'dataset': dataset_name,
                'simulation_type': 'cyclic_boost',
                'tuned_clients': bool(TUNE_CLIENTS),
                'min_data_in_leaf': tuned_leaf,
            }
            res.update(meta_inf)
            if f_imp:
                f_imp.update(meta_inf)
            results.append((res, f_imp))

    return results


def _run_fedboost_simulation(datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                             scoring, feature_importance, weighting, rep, analysis_name, dataset_name,
                             max_combos_per_k=None, rounds=5, multishot_tune_cache=None):
    """PARALLEL federated gradient boosting -- the sophisticated multi-round method.

    Each round every client boosts `trees_per_round` trees from the SAME current global margin
    (LightGBM init_score), and the server aggregates the k increments as a WEIGHTED AVERAGE:

        F_r(x) = F_{r-1}(x) + sum_c w_c * dF_c(x),     sum_c w_c = 1

    Contrast with cyclic boosting, which passes one booster sequentially around the ring: there the model
    drifts toward whichever cohort went last, and it collapses as k grows (measured: 0.68 at k=3 ->
    0.57 at k=10). Here every client sees the same starting point each round, so there is no drift and no
    order dependence.

    `weighting` reuses the existing axis: 'size' is NVFlare's scaled shrinkage (w_c proportional to n_c),
    'none' is a uniform average.

    Budget: the per-round increments are AVERAGED (weights sum to 1), so one round adds ~trees_per_round
    of effective capacity. trees_per_round = total // rounds keeps the total budget comparable to pooled.
    """
    results = []

    if str(model).split('_')[0] != 'lgbm':
        return results   # gbdt continuation only (init_score is not supported by boosting_type='rf')

    all_combs = list(combinations(train_cohorts, n_clients))
    if max_combos_per_k is not None and len(all_combs) > max_combos_per_k:
        rng = np.random.default_rng(RANDOM_STATE + rep * 1000 + n_clients)
        sel = sorted(rng.choice(len(all_combs), size=max_combos_per_k, replace=False))
        all_combs = [all_combs[i] for i in sel]

    total_trees = getattr(get_estimator(model, rep=rep), 'n_estimators', 100) or 100
    trees_per_round = max(1, total_trees // rounds)

    for comb in all_combs:
        test_data = [c for c in train_cohorts if c not in comb]
        test_data.extend(val_cohorts)

        # One global leaf size for the shared boosting model, tuned server-side by LOCO inside comb when
        # --tune-clients is on (else the untuned template). See _tuned_multishot_template.
        template = _tuned_multishot_template(model, comb, datasets, rep, multishot_tune_cache)
        tuned_leaf = template.get_params().get('min_data_in_leaf')

        for w_name in weighting:
            raw = _build_weights(w_name, comb, datasets)
            if raw is None:
                w_vec = np.full(len(comb), 1.0 / len(comb))
            else:
                w_vec = np.asarray(raw, dtype=float)
                s = w_vec.sum()
                w_vec = w_vec / s if s > 0 else np.full(len(comb), 1.0 / len(comb))

            glob = FederatedBoostedModel(boosters=[], classes=[0, 1]).fit()
            try:
                for _r in range(rounds):
                    increments = []
                    for ci, grp in enumerate(comb):
                        X, y = datasets[grp]
                        # init_score is computed LOCALLY from the global model the client already holds.
                        init = glob.decision_function(X) if glob.boosters_ else None
                        est = clone(template)
                        est.set_params(n_estimators=trees_per_round)
                        est.fit(X, y, init_score=init)
                        increments.append((float(w_vec[ci]), est.booster_))
                    glob.boosters_.extend(increments)   # server aggregates the weighted average
            except Exception as e:
                print(f"fedboost: failed for {comb} w={w_name}: {type(e).__name__}: {e}")
                continue

            for grp in test_data:
                X_test, y_test = datasets[grp]
                res, f_imp = custom_comb_validate(
                    glob, X_test, y_test,
                    features=feature_names, metrics=scoring, feature_importance=feature_importance
                )
                meta_inf = {
                    "rep": rep,
                    "train_dataset": ', '.join(map(str, comb)),
                    "val_dataset": grp,
                    'analysis': analysis_name,
                    "cv": 0,
                    'n_clients': n_clients,
                    'model': model,
                    'weighting': w_name,
                    'voting': 'fedboost',
                    'score_transform': 'none',
                    'rounds': rounds,
                    'trees_per_round': trees_per_round,
                    'dataset': dataset_name,
                    'simulation_type': 'fedboost',
                    'tuned_clients': bool(TUNE_CLIENTS),
                    'min_data_in_leaf': tuned_leaf,
                }
                res.update(meta_inf)
                if f_imp:
                    f_imp.update(meta_inf)
                results.append((res, f_imp))

    return results


def simulate(
    anno_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    simulation_type: str,
    dataset_name: str,
    analysis_name: str,
    groups: list,
    clients: list | None = None,
    label: str = "health_status",
    REPETITION: int = 10,
    N_ESTIMATOR: int = 100,
    N_JOBS: int = -1,
    CROSS_VAL: int = 5,
    feature_importance: str = 'gini',
    weighting: str = 'none',
    data_dir: str = './',
    scoring: list = ['accuracy'],
    exec_mode: str = 'add',
    filename: str = 'simulation_results.pkl',
    save_path: str = './',
    verbose: bool = False,
    log_level: int = logging.ERROR,
    log_file: str = "execution_time.json",
    model: str = 'rf',
    max_combos_per_k: int | None = None,
    voting: str | None = None,
    rounds: int = 3,
):
    """
    Unified function to run different types of simulations.
    
    Args:
        anno_df: Annotation numpy array
        counts: Count numpy array
        anno_headers: Numpy array of annotation headers
        count_headers: Numpy array of count headers
        simulation_type: Type of simulation ('central', 'local', 'federated')
        dataset_name: Name of the dataset
        analysis_name: Name of the analysis
        groups: List of group columns
        clients: List of client numbers to use for federated simulations
        label: Column name of the target variable
        REPETITION: Number of repetitions to run
        N_ESTIMATOR: Number of estimators for ensemble methods
        N_JOBS: Number of jobs to run in parallel
        CROSS_VAL: Number of cross-validation folds
        feature_importance: Method for calculating feature importance
        weighting: Weighting method for federated learning
        data_dir: Directory containing data files
        scoring: Scoring metrics to use
        exec_mode: Execution mode ('redo', 'load', 'add')
        filename: Output filename
        save_path: Path to save results
        verbose: Whether to print verbose output
        log_level: Logging level
        log_file: Log file path
        plot_confusion: Whether to plot confusion matrices
        model: Model type to use
        
    Returns:
        None. Results are saved to disk.
    """
    # Convert to Path object for better path handling if it's a string
    if isinstance(save_path, str):
        save_path = Path(save_path) / filename
    else:
        save_path = save_path / filename
        
    save_path_unfinished = save_path.with_suffix('.unfinished')

    # Check execution mode
    if exec_mode not in ['redo', 'load', 'add']:
        raise ValueError(f'Invalid exec_mode: {exec_mode}. Must be "redo", "load", or "add".')

    # Initialize results
    res_out, f_imp_out = [], []

    # Load previous results if necessary
    if exec_mode == 'load':
        if save_path.exists():
            if log_level <= logging.INFO:
                print(f"Loading previous results for {analysis_name}.")
            return 
        else:
            print(f"No previous results found for {analysis_name}. Starting from scratch.")

    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Prepare data for simulation
        central_data, train_cohorts, val_cohorts, feature_names, datasets = _prepare_simulation_data(
            anno_df, counts_df, label, groups
        )
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error preparing data for simulation. Please check your input data: {e}")

    n_train = len(train_cohorts)
    if clients is None:
        clients = [_ for _ in range(1, n_train + 1)]
    else:
        # Restrict the k-sweep to an explicit client-count list. Accepts ints and the tokens
        # 'max' (== number of trainable cohorts) and 'max-N' (N below max). 'max-1' is the largest
        # k that still HOLDS OUT a training cohort: at k == n_train every training cohort is used,
        # so the only evaluation cohorts left are the validation-only ones (often single-class ->
        # undefined AUROC). n_train-1 is therefore the meaningful curve endpoint for cross-cohort
        # arms (this is where the manuscript reports fedboost/cyclic_boost). Values are clamped to
        # [1, n_train], deduped and sorted; out-of-range entries are dropped. Only the
        # federated-family arms sweep k, so central/local are unaffected.
        def _resolve_k(tok):
            s = str(tok).strip().lower()
            if s == 'max':
                return n_train
            if s.startswith('max-'):
                return n_train - int(s[4:])   # 'max-1' -> n_train-1
            return int(s)
        _resolved = {_resolve_k(c) for c in clients}
        clients = sorted(k for k in _resolved if 1 <= k <= n_train)
        if not clients:
            raise ValueError(f"clients restriction resolved to no valid k for {analysis_name} "
                             f"(trainable cohorts = {n_train}).")

    if central_data.size == 0:
        raise ValueError("Merged data is empty after dropping NA values. Please check your input data.")

    if verbose:
        print(f"Merged data contains {central_data.shape[0]} samples and {central_data.shape[1]} features.")
        
    # The combination sweep is subsampled to max_combos_per_k per k (identically in the pooled and
    # federated arms), so progress must be counted CAPPED. Counting the full C(n,k) overstated the
    # total ~4x and inflated every ETA. _ncomb is the true per-k work; use it for the total AND every
    # pbar.update below.
    def _ncomb(k):
        c = len(list(combinations(train_cohorts, k)))
        return min(c, max_combos_per_k) if max_combos_per_k else c

    # Calculate total steps for progress bar
    if simulation_type == 'central':
        pbar_total = REPETITION * CROSS_VAL
    elif simulation_type == 'local':
        pbar_total = len(datasets) * REPETITION
    else:  # 'federated'
        if not datasets:
            raise ValueError("No valid groups found for the simulation. Please check your group criteria and input data.")

        pbar_total = sum(_ncomb(n) for n in clients) * REPETITION
        
        if simulation_type == 'federated':
            pbar_total += len(clients) * len(train_cohorts) * REPETITION

    # Make weighting a list if it's not already
    if not isinstance(weighting, list):
        weighting = [weighting]

    # Run simulations with progress tracking
    with tqdm(total=pbar_total, desc=f'{simulation_type} - {analysis_name}') as pbar:
        
        # For federated learning, precompute all models at the beginning
        estimator_cache = None
        # Memo for member predictions, shared across every combination, weighting and k of this
        # experiment. The aggregation sweep does ZERO training -- it is entirely inference, and almost
        # all of it is redundant: a member's prediction on a cohort is the same no matter which other
        # members it is voted with. Roughly 150 distinct (member, cohort) pairs back ~20,000 inferences
        # per weighting.
        pred_cache = {}
        # Persistent tuning cache: seed each in-memory cache from disk (best-effort; {} when disabled or
        # on any mismatch), so a repeated run reuses the winning hyperparameters instead of re-searching.
        # Keyed per experiment+kind and guarded by a data/grid fingerprint. Saved after the rep loop.
        # The fingerprint is computed ONLY when the cache is enabled, and best-effort, so the disabled
        # path (env unset) is byte-for-byte the original behaviour and can never raise here.
        try:
            _tc_fp = _tune_fingerprint(train_cohorts, datasets) if _TUNE_CACHE_DIR else None
        except Exception:
            _tc_fp = None
        # Nested-LOCO hyperparameter selection is deterministic in (rep, model, combination), so the same
        # combination reached at a different k reuses its winner instead of re-searching the grid.
        tune_cache = _tune_cache_load('pooled', analysis_name, _tc_fp)
        # A client's own-CV selection depends only on (rep, model, cohort) -- never on k or on the
        # combination -- so it is chosen once per client and reused everywhere.
        client_tune_cache = _tune_cache_load('client', analysis_name, _tc_fp)
        # The boosting arms tune ONE shared config per combination (leakage-free LOCO inside comb), so
        # this cache is keyed by (rep, model, sorted(comb)).
        multishot_tune_cache = _tune_cache_load('multishot', analysis_name, _tc_fp)
        if simulation_type == 'federated':
            pbar.desc = f'Precomputing estimators for {simulation_type} - {analysis_name}'
            if verbose:
                print(f"Precomputing estimators for all repetitions and clients...")

            estimator_cache = {}
            for rep in range(REPETITION):
                estimator_cache = _precompute_estimators(
                    datasets,
                    train_cohorts,
                    model,
                    rep,
                    estimator_cache,
                    client_tune_cache=client_tune_cache,
                )
                pbar.update(len(train_cohorts) * len(clients))

            pbar.desc = f'{simulation_type} - {analysis_name}'

        for rep in range(REPETITION):
            # Central simulation: one model trained on all data
            if simulation_type == 'central':
                if verbose:
                    print(f'Starting repetition {rep + 1}/{REPETITION}')
                
                try:
                    cv_seed = get_deterministic_random_state('cv_split', rep=rep)
                    # With --tune, a penalised linear model chooses its C inside each outer fold; trees and
                    # the unpenalised linear models keep their fixed hyperparameters.
                    nested = TUNE and str(model).split('_')[0] in ('log', 'svm') and _penalty_of(model)
                    central_model = (_NestedTunedCentralModel(model, rep, cv_seed, CROSS_VAL) if nested
                                     else get_estimator(model, rep=rep))
                    results, importances = _run_central_simulation(
                        central_data, feature_names, label, groups, central_model, CROSS_VAL,
                        scoring, feature_importance,
                        random_state=cv_seed,
                    )
                    
                    # Add rep info to each result
                    for result in results:
                        result["rep"] = rep
                        result["model"] = model
                        result["analysis"] = analysis_name
                        result["dataset"] = dataset_name
                        result['simulation_type'] = simulation_type
                        res_out.append(result)
                    
                    # For feature importances
                    if importances:
                        f_imp_out.append(importances)
                except Exception as e:
                    print(f"Error in repetition {rep + 1}/{REPETITION}: {e}")
                    traceback.print_exc()
                    
                pbar.update(CROSS_VAL)

            # Local simulation: train separate models for each cohorts
            elif simulation_type == "local":
                if verbose:
                    print(f'Starting local cross-validation {rep + 1}/{REPETITION}')
                
                try:
                    local_model = ((lambda grp: _SelfTunedLocalModel(model, rep, grp)) if TUNE_LOCAL
                                   else get_estimator(model, rep=rep))
                    results, importances = _run_local_simulation(
                        datasets, feature_names, train_cohorts, local_model,
                        CROSS_VAL, scoring, feature_importance, rep=rep,
                    )

                    # Add rep info to each result
                    for result in results:
                        result["rep"] = rep
                        result["model"] = model
                        result["analysis"] = analysis_name
                        result["dataset"] = dataset_name
                        result['simulation_type'] = simulation_type
                        res_out.append(result)
                    
                    # For feature importances
                    for imp in importances:
                        f_imp_out.append(imp)
                except Exception as e:
                    print(f"Error in local simulation: {e}")
                    traceback.print_exc()
                    
                pbar.update(len(datasets))

            # Federated simulation: use pre-trained models to combine models in a one-shot federated approach + weighting
            elif simulation_type == 'federated':
                for n_clients in clients:
                    try:
                        results = _run_federated_simulation(
                            datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                            scoring, feature_importance, weighting, rep, analysis_name, dataset_name, estimator_cache,
                            max_combos_per_k=max_combos_per_k,
                            voting=voting,
                            pred_cache=pred_cache,
                        )
                        
                        # Add metadata to each result and append to outputs
                        for (res, f_imp) in results:                            
                            # Append to outputs
                            res_out.append(res)
                            f_imp_out.append(f_imp)
                    except Exception as e:
                        print(f"Error in federated simulation with {n_clients} clients: {e}")
                        traceback.print_exc()

                    pbar.update(_ncomb(n_clients))

            # Parallel federated boosting: the drift-free multi-round method (see _run_fedboost_simulation).
            elif simulation_type == 'fedboost':
                for n_clients in clients:
                    try:
                        results = _run_fedboost_simulation(
                            datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                            scoring, feature_importance, weighting, rep, analysis_name, dataset_name,
                            max_combos_per_k=max_combos_per_k, rounds=rounds,
                            multishot_tune_cache=multishot_tune_cache,
                        )
                        for (res, f_imp) in results:
                            res_out.append(res)
                            f_imp_out.append(f_imp)
                    except Exception as e:
                        print(f"Error in fedboost simulation with {n_clients} clients: {e}")
                        traceback.print_exc()

                    pbar.update(_ncomb(n_clients))

            # Cyclic boosting: a genuine multi-round federation (see _run_cyclic_simulation).
            elif simulation_type == 'cyclic_boost':
                for n_clients in clients:
                    try:
                        results = _run_cyclic_simulation(
                            datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                            scoring, feature_importance, rep, analysis_name, dataset_name,
                            max_combos_per_k=max_combos_per_k, rounds=rounds,
                            multishot_tune_cache=multishot_tune_cache,
                        )
                        for (res, f_imp) in results:
                            res_out.append(res)
                            f_imp_out.append(f_imp)
                    except Exception as e:
                        print(f"Error in cyclic simulation with {n_clients} clients: {e}")
                        traceback.print_exc()

                    pbar.update(_ncomb(n_clients))

            # Pooled: the like-for-like centralised control at matched k (see _run_pooled_simulation).
            elif simulation_type == 'pooled':
                for n_clients in clients:
                    try:
                        results = _run_pooled_simulation(
                            datasets, feature_names, train_cohorts, val_cohorts, n_clients, model,
                            scoring, feature_importance, rep, analysis_name, dataset_name,
                            max_combos_per_k=max_combos_per_k, tune_cache=tune_cache,
                        )
                        for (res, f_imp) in results:
                            res_out.append(res)
                            f_imp_out.append(f_imp)
                    except Exception as e:
                        print(f"Error in pooled simulation with {n_clients} clients: {e}")
                        traceback.print_exc()

                    pbar.update(_ncomb(n_clients))

    # Save all results at once
    with open(save_path, 'wb') as f:
        pickle.dump((res_out, f_imp_out), f)

    # Persist the tuning winners so a later run skips the grid search (best-effort; no-op when the
    # cache is disabled or a cache is empty).
    _tune_cache_save('pooled', analysis_name, _tc_fp, tune_cache)
    _tune_cache_save('client', analysis_name, _tc_fp, client_tune_cache)
    _tune_cache_save('multishot', analysis_name, _tc_fp, multishot_tune_cache)


def format_pickled_results(experiment, simulation_type, input_dir=raw_results_dir, output_dir=table_dir, prefix='', verbose=False, **kwargs):
    """
    Format and save pickled simulation results to CSV.
    
    Args:
        experiment: The experiment object
        simulation_type: Type of simulation
        raw_results_dir: Directory for saving pickled results
        table_dir: Directory for saving formatted CSV results
        prefix: Prefix for filenames
        **kwargs: Additional arguments
        
    Returns:
        None
    """
    save_path = path.join(input_dir, f'{prefix}{experiment.name}_{simulation_type}_results.pkl')
    all_results, all_f_imp = [], []
    
    try:
        with open(save_path, 'rb') as f:
            try:
                while True:
                    res_out_tmp, f_imp_out_tmp = pickle.load(f)
                    all_results.extend(res_out_tmp)
                    all_f_imp.extend(f_imp_out_tmp)
            except EOFError:   
                pass
    except Exception as e:
        print(f"Error loading pickled results: {e}")
        return
        
    if not all_results:
        print(f"No results found for {experiment.name} - {simulation_type} at {save_path}")
        return
    
    # Process results data
    csv_rows = []
    headers = set()
    
    # First pass: get all possible headers
    for result in all_results:
        headers.update(result.keys())
    
    headers = sorted(list(headers))
    
    # Second pass: build rows with consistent columns
    for result in all_results:
        # Check if any values are lists
        has_lists = any(isinstance(result.get(header, None), list) for header in headers)
        
        if has_lists:
            # Get the length of lists (assuming all lists have the same length)
            list_len = max(len(result.get(header, [])) 
                          for header in headers 
                          if isinstance(result.get(header, None), list))
            
            # Create multiple rows for list elements
            for i in range(list_len):
                row = {}
                for header in headers:
                    value = result.get(header, None)
                    if isinstance(value, list) and i < len(value):
                        row[header] = value[i]
                    elif isinstance(value, list):
                        row[header] = None  # Pad shorter lists with None
                    else:
                        row[header] = value  # Non-list values repeated for each row
                csv_rows.append(row)
        else:
            # Regular row with no lists
            row = {header: result.get(header, None) for header in headers}
            csv_rows.append(row)
    
    # Write results to human-readable CSV file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Must carry the same prefix as the pickle it was read from: without it a prefixed run
    # (dry-run, weighting grid) silently overwrites the main results table.
    output_file = path.join(output_dir, f'{prefix}{experiment.name}_{simulation_type}.csv')
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    if verbose:
        print(f"Formatted results saved to {output_file}")
    
def check_baseline_invariant(analysis_name, table_dir=table_dir, weighting='none',
                             tolerance=0.0, verbose=True, arm='federated'):
    """SANITY INVARIANT: centralized (pooled) >= a federated arm, on the same data.

    Pooling is a superset of the information available to any partition of it, so NO federated method --
    one-shot voting OR multi-round boosting (fedboost/cyclic_boost) -- can beat centralized training on
    the same cohorts. When it appears to, the baseline is broken -- that is exactly how the 'federation
    beats pooling on linear models' artefact survived for a week during the revision: the pooled SGD
    never got a fair fit, and the win was read as a finding instead of a bug.

    `arm` selects which federated table to check: 'federated', 'fedboost' or 'cyclic_boost'. Use it
    to gate the boosting arms too, since fedboost is the one most likely to *look* like it beats
    pooling (it averages k boosters and is easy to over-budget).

    Compares only PAIRED cells -- the same (rep, k, training combination, held-out cohort) -- because the
    arms are evaluated on identical cells by construction. Judges the paired MEAN, not individual cells:
    a single cell going the other way is noise; the mean going the other way is a bug.

    Returns (ok, report). Does not raise -- the caller decides -- but says so loudly.
    """
    fed_p = Path(table_dir) / f'{analysis_name}_{arm}.csv'
    pool_p = Path(table_dir) / f'{analysis_name}_pooled.csv'
    if not fed_p.exists() or not pool_p.exists():
        return True, f"{analysis_name}: skipped (need both _{arm}.csv and _pooled.csv)"

    fed = pd.read_csv(fed_p)
    pool = pd.read_csv(pool_p)

    # Select the published federated configuration -- but ONLY when the arm actually sweeps that axis.
    # The boosting arms label their single configuration weighting='cyclic'/voting='cyclic', so filtering
    # to weighting=='none' would empty the table and the gate would silently skip (return "no paired
    # cells") instead of checking. Only filter when the requested value is present.
    if 'weighting' in fed.columns and weighting is not None and (fed['weighting'] == weighting).any():
        fed = fed[fed['weighting'] == weighting]

    keys = ['rep', 'n_clients', 'train_dataset', 'val_dataset']
    m = fed.merge(pool, on=keys, suffixes=('_fed', '_pool'))
    m = m.dropna(subset=['test_roc_auc_fed', 'test_roc_auc_pool'])
    if m.empty:
        return True, f"{analysis_name}: skipped (no paired cells)"

    d = m['test_roc_auc_fed'] - m['test_roc_auc_pool']
    mean_fed, mean_pool = m['test_roc_auc_fed'].mean(), m['test_roc_auc_pool'].mean()
    win = float((d > 0).mean())
    ok = (mean_fed - mean_pool) <= tolerance

    tuned = bool(pool['tuned'].iloc[0]) if 'tuned' in pool.columns and len(pool) else False
    report = (f"{analysis_name}: federated {mean_fed:.4f} vs pooled {mean_pool:.4f} "
              f"({mean_fed - mean_pool:+.4f}, federation wins {win:.1%} of {len(m)} paired cells) "
              f"[pooled tuned={tuned}]")

    if not ok:
        report += ("\n    *** INVARIANT VIOLATED: federation beats centralized on the same data. ***"
                   "\n    Treat as a BUG until proven otherwise -- audit the BASELINE first"
                   "\n    (solver convergence, regularisation at pooled scale, per-client "
                   "hyperparameters applied to a pooled fit)."
                   + ("\n    The pooled arm here is UNTUNED -- re-run it with --tune before "
                      "theorising about a mechanism." if not tuned else ""))
    if verbose:
        print(report)
    return ok, report


def load_experiment(experiment, data_dir=data_dir):
    # Load data with index columns
    anno_df = pd.read_csv(Path(data_dir) / experiment.anno, index_col=0) if isinstance(experiment.anno, (str, Path)) else experiment.anno
    counts_df = pd.read_csv(Path(data_dir) / experiment.count, index_col=0) if isinstance(experiment.count, (str, Path)) else experiment.counts
    # Apply canonical cohort definitions (merge Germany, split Italy1) to every simulation path.
    anno_df = apply_cohort_definitions(anno_df)
    return anno_df, counts_df
    


def simulate_threaded(experiment, simulation_type, save_dir=save_dir, EXEC_MODE=EXEC_MODE, 
                     REPETITION=REPETITION, N_JOBS=N_JOBS, prefix='', **kwargs):
    """
    Run a simulation in a separate thread.
    
    Args:
        experiment: The experiment object
        simulation_type: Type of simulation
        save_dir: Directory for saving results
        EXEC_MODE: Execution mode
        REPETITION: Number of repetitions
        N_JOBS: Number of jobs
        prefix: Prefix for filenames
        **kwargs: Additional arguments
        
    Returns:
        None
    """
    
    try:
        # Load data with index columns
        anno_df, counts_df = load_experiment(experiment, data_dir)
        # Run simulation
        simulate(
            anno_df=anno_df, 
            counts_df=counts_df, 
            simulation_type=simulation_type, 
            dataset_name=experiment.clean_dataset, 
            analysis_name=experiment.name, 
            groups=groups, 
            save_path=raw_results_dir, 
            filename=f'{prefix}{experiment.name}_{simulation_type}_results.pkl',
            label=experiment.label, 
            exec_mode=EXEC_MODE, 
            REPETITION=REPETITION, 
            N_JOBS=N_JOBS, 
            model=experiment.model, 
            **kwargs
        )

        # Format results
        format_pickled_results(experiment, simulation_type, prefix=prefix, **kwargs)
        
    except Exception as e:
        print(f"ERROR: Skipping {experiment.name} due to error: {e}")
        traceback.print_exc()

def _pool_init(merge_germany, split_italy, est_n_jobs, tune=None, tune_clients=None,
               tune_local=None):
    """ProcessPoolExecutor initializer: set worker-local cohort flags and cap each estimator's threading.
    Runs once per worker, so cohort-definition flags and thread caps take effect regardless of the
    multiprocessing start method (fork / forkserver / spawn) — globals reassigned in __main__ do not
    otherwise cross the process boundary under forkserver/spawn.

    TUNE must come through here too, and its failure mode is bad: an untuned worker still produces
    plausible pooled numbers, just against the handicapped baseline the whole exercise exists to remove.
    It would look like a result, not like a bug."""
    global MERGE_GERMANY, SPLIT_ITALY, TUNE, TUNE_CLIENTS, TUNE_LOCAL
    MERGE_GERMANY = merge_germany
    SPLIT_ITALY = split_italy
    if tune is not None:
        TUNE = tune
    if tune_clients is not None:
        TUNE_CLIENTS = tune_clients
    if tune_local is not None:
        TUNE_LOCAL = tune_local
    for _est in estimators.values():
        try:
            _est.set_params(clf__n_jobs=est_n_jobs) if hasattr(_est, 'steps') else _est.set_params(n_jobs=est_n_jobs)
        except Exception:
            pass


def run_simulation_with_threading(experiments, simulation_type, max_threads=N_JOBS, threads_per_job=THREADS_PER_JOB,
                                  merge_germany=None, split_italy=None, **kwargs):
    """
    Run simulations in parallel using multithreading.
    
    Args:
        experiments: List of experiment objects
        simulation_type: Type of simulation
        max_threads: Maximum number of threads to use
        threads_per_job: Number of threads per job
        **kwargs: Additional arguments
        
    Returns:
        None
    """
    # Determine available resources
    num_cores = os.cpu_count() or 1

    # Set threading parameters
    if max_threads is None or max_threads == -1:
        max_threads = num_cores - 2  # Use all available threads
    elif max_threads == -2:
        max_threads = num_cores // 2  # Use half of the available threads
    else:
        max_threads = min(max_threads, num_cores)  # Limit to the available threads
        
    if threads_per_job > max_threads:
        threads_per_job = max_threads
        
    max_concurrent_jobs = min(max_threads // threads_per_job, len(experiments) * threads_per_job)
    
    print(f"Running {simulation_type} with {max_concurrent_jobs} concurrent jobs on {max_threads} threads")
    
    # Cohort-definition flags + estimator thread cap are pushed into every worker via the initializer,
    # so they are honoured under fork, forkserver and spawn alike.
    init_args = (
        MERGE_GERMANY if merge_germany is None else merge_germany,
        SPLIT_ITALY if split_italy is None else split_italy,
        threads_per_job,
        TUNE,
        TUNE_CLIENTS,
        TUNE_LOCAL,
    )

    # Run jobs in parallel
    with ProcessPoolExecutor(max_workers=max_concurrent_jobs,
                             initializer=_pool_init, initargs=init_args) as executor:
        future_to_exp = {
            executor.submit(
                partial(simulate_threaded, exp, simulation_type, N_JOBS=threads_per_job, **kwargs)
            ): exp for exp in experiments
        }
        
        # Progress tracking
        with tqdm(total=len(future_to_exp), desc=f'running {simulation_type}') as pbar:
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    pbar.update(1)
                except Exception as e:
                    print(f"Experiment {exp.name} generated an exception: {e}")
                    traceback.print_exc()


DEBUG = False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run federated CRC microbiome simulations (central/local/federated).')
    parser.add_argument('--dry-run', action='store_true',
        help='Minimal end-to-end smoke test: one light experiment, 1 rep, 2-fold CV, federated k=1,2.')
    parser.add_argument('--repetition', type=int, default=None,
        help=f'Repetitions (config default: {REPETITION}).')
    parser.add_argument('--n-jobs', type=int, default=None, help='Max concurrent experiment processes.')
    parser.add_argument('--threads-per-job', type=int, default=None,
        help='Threads per model; also caps each estimator n_jobs so the machine is not saturated.')
    parser.add_argument('--cohort-table', action='store_true',
        help='Write Supplementary Table 1 (per-cohort n / class counts / trainable) and exit.')
    parser.add_argument('--models', type=str, default=None,
        help="Comma-separated model names to keep (e.g. lgbm,rf,log). Filters on Experiment.model.")
    parser.add_argument('--simulations', type=str, default=None,
        help="Comma-separated simulation types: central,local,federated,pooled,fedboost,cyclic_boost. "
             f"Config default: {','.join(DEFAULT_SIMULATIONS)}")
    parser.add_argument('--rounds', type=int, default=None,
        help=f"Rounds R for fedboost/cyclic_boost (config default: {DEFAULT_ROUNDS}). The total tree budget "
             "is split across all R*k client visits, so R does not change ensemble size.")
    parser.add_argument('--prefix', type=str, default='',
        help="Filename prefix for outputs — write to separate files instead of overwriting the main results.")
    parser.add_argument('--weighting', type=str, default=None,
        help="Comma-separated client weightings for the federated vote. "
             f"none | size. Config default: {','.join(DEFAULT_WEIGHTING)}")
    parser.add_argument('--voting', type=str, default=None,
        choices=['auto', 'soft', 'hard'],
        help="Federated voting rule. 'auto' (default) picks soft iff members expose predict_proba, "
             "which confounds voting with model family; set explicitly to compare voting rules.")
    parser.add_argument('--check-invariant', action='store_true',
                        help='Check that centralized (pooled) >= federated on the same data, on paired '
                             'cells, and exit non-zero if not. A federated method cannot beat centralized '
                             'training on the same cohorts, so a violation is a bug in the baseline -- not '
                             'a finding. Run this before quoting any federated-vs-centralized number.')
    parser.add_argument('--tune-clients', action=argparse.BooleanOptionalAction, default=TUNE_CLIENTS,
                        help='Fit the federation members properly instead of the hand-set min_data_in_leaf=2, '
                             'which overfits a 44-sample cohort and costs ~0.065 AUROC. One-shot: '
                             'each client tunes on its own cohort. Boosting: one shared config tuned '
                             'server-side by LOCO. METHOD CHANGE. Use --no-tune-clients to disable. '
                             f'Config default: {TUNE_CLIENTS}.')
    parser.add_argument('--tune-local', action=argparse.BooleanOptionalAction, default=TUNE_LOCAL,
                        help='Let each centre in the local arm (B) select its own hyperparameters, by the '
                             'same intra-cohort search a federation client uses, nested inside each CV fold. '
                             'Without it (B) is the only arm left on the pipeline default min_data_in_leaf=2, '
                             'so federated-vs-local compares a tuned federation against a configuration every '
                             f'client rejected. Config default: {TUNE_LOCAL}.')
    parser.add_argument('--tune', action=argparse.BooleanOptionalAction, default=TUNE,
                        help='Fit the centralized/pooled BASELINE properly (converged solver + LOCO-selected '
                             'hyperparameters). Without it the baseline reuses per-client hyperparameters and '
                             'every federated-vs-centralized gap is biased toward federation. Use '
                             f'--no-tune to disable. Config default: {TUNE}.')
    parser.add_argument('--max-combos-per-k', type=int, default=None,
        help='Subsample the federated k-cohort combination sweep (0 = exhaustive). '
             f'Config default: {DEFAULT_MAX_COMBOS if DEFAULT_MAX_COMBOS else 0}.')
    parser.add_argument('--clients', type=str, default=None,
        help="Restrict the k-sweep (number of participating cohorts) to this comma list; "
             "'max' = all trainable cohorts, 'max-1' = one below (the largest k that still holds "
             "out a training cohort -- the meaningful endpoint for cross-cohort arms; e.g. "
             "'1,2,3,5,max-1'). Only the federated-family arms (federated/pooled/fedboost/"
             "cyclic_boost) sweep k; central/local ignore it. Default: every k = 1..N.")
    parser.add_argument('--dataset', type=str, default=None,
        help='Only run experiments whose name contains this substring (e.g. count_taxa_meteor_0).')
    parser.add_argument('--redo', action='store_true',
        help='Recompute even if result files exist (default: load/skip existing).')
    parser.add_argument('--no-merge-germany', action='store_true',
        help='Sensitivity analysis: keep Germany1/Germany2 separate instead of merging.')
    parser.add_argument('--no-split-italy', action='store_true', help='Do not split Italy1 into Italy1a/Italy1b.')
    args = parser.parse_args()

    # Every setting below falls back to config.toml (loaded into the module globals at import); a CLI flag
    # only overrides when explicitly given. So `run_analysis.py` with no flags == the paper experiment.

    # Cohort-definition flags (config default; the --no-* flags force off)
    if args.no_merge_germany:
        MERGE_GERMANY = False
    if args.no_split_italy:
        SPLIT_ITALY = False

    # Resource settings
    if args.repetition is not None:
        REPETITION = args.repetition
    if args.n_jobs is not None:
        N_JOBS = args.n_jobs
    if args.threads_per_job is not None:
        THREADS_PER_JOB = args.threads_per_job   # applied to each estimator per-worker via _pool_init

    EXEC_MODE = 'redo' if args.redo else EXEC_MODE
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"execution_time_{timestamp}.json"
    verbose = False
    weighting = [w.strip() for w in args.weighting.split(',')] if args.weighting else list(DEFAULT_WEIGHTING)
    rounds = args.rounds if args.rounds is not None else DEFAULT_ROUNDS
    max_combos_per_k = args.max_combos_per_k if args.max_combos_per_k is not None else DEFAULT_MAX_COMBOS
    max_combos_per_k = max_combos_per_k or None   # 0 -> exhaustive

    TUNE = args.tune                 # BooleanOptionalAction: defaults to config, --no-tune overrides
    TUNE_CLIENTS = args.tune_clients
    TUNE_LOCAL = args.tune_local
    if args.voting is not None:
        VOTING = args.voting

    _valid_w = {'none', 'size'}
    if not set(weighting) <= _valid_w:
        raise SystemExit(f"Unknown weighting(s): {sorted(set(weighting) - _valid_w)}. Valid: {sorted(_valid_w)}")

    run_experiments = experiments
    if args.dataset:
        run_experiments = [e for e in experiments if args.dataset in e.name]
    models_filter = args.models.split(',') if args.models else DEFAULT_MODELS
    if models_filter:
        # Filter by model BEFORE the gate so --check-invariant honours it. --models overrides the
        # config `models` grid; an empty config `models` means the full 16-model grid. Historically
        # this block sat AFTER the check-invariant branch, so the gate scanned every experiment with a
        # table on disk (including stale/out-of-grid ones) regardless of the model selection.
        keep = {m.strip() for m in models_filter}
        run_experiments = [e for e in run_experiments if e.model in keep]

    if args.check_invariant:
        # A gate, not a report: exits non-zero so it can sit in front of anything that publishes a
        # federated-vs-centralized number. Checks every federated arm that has a results table.
        # Linear models are reported, not gated: matched pooled tunes C exactly as the clients
        # do, so the two sit within ~0.005 AUROC of each other and either may lead at a given k.
        print("Sanity invariant: centralized (pooled) >= EVERY federated arm on the same data.\n")
        violations = []
        for e in run_experiments:
            n, gated = e.name, _is_tree_model(e.model)
            for arm in ('federated', 'fedboost', 'cyclic_boost'):
                if not (Path(table_dir) / f'{n}_{arm}.csv').exists():
                    continue
                ok, report = check_baseline_invariant(n, arm=arm, verbose=gated)
                if not gated:
                    print(report.splitlines()[0] + '  [linear: reported, not gated]')
                elif not ok:
                    violations.append(f'{n} [{arm}]')
        print()
        if violations:
            raise SystemExit(f"INVARIANT VIOLATED in {len(violations)} arm(s): "
                             f"{', '.join(violations)}\nDo not publish these numbers. Audit the baseline.")
        print("All checked arms satisfy the invariant.")
        raise SystemExit(0)

    extra_kwargs = dict(REPETITION=REPETITION)
    if max_combos_per_k is not None:
        extra_kwargs['max_combos_per_k'] = max_combos_per_k
    extra_kwargs['voting'] = VOTING
    extra_kwargs['rounds'] = rounds
    if args.clients:
        extra_kwargs['clients'] = [c.strip() for c in args.clients.split(',') if c.strip()]
    if args.prefix:
        extra_kwargs['prefix'] = args.prefix

    if DEBUG:
        REPETITION = 1
        run_experiments = experiments[:2]
        extra_kwargs['REPETITION'] = 1

    if args.dry_run:
        # one light taxa experiment, minimal reps/folds, federated sweep limited to k=1,2
        _light = [e for e in experiments if 'taxa' in e.name]
        run_experiments = (_light or experiments)[:1]
        EXEC_MODE = 'redo'
        # Always prefix dry-run output. Without this the smoke test writes toy results (1 rep, 2-fold,
        # k<=2) over the real ones under the SAME filenames — they then look like completed results.
        extra_kwargs = dict(REPETITION=1, CROSS_VAL=2, clients=[1, 2],
                            prefix=args.prefix or 'dryrun_',
                            voting=VOTING)
        if args.max_combos_per_k is not None:
            extra_kwargs['max_combos_per_k'] = args.max_combos_per_k
        if args.weighting:
            pass  # keep the parsed `weighting` list
        print("=" * 60)
        print("DRY RUN — integrated pipeline (minimal settings)")
        print(f"  experiment : {run_experiments[0].name}")
        print(f"  cohorts    : merge_germany={MERGE_GERMANY} split_italy={SPLIT_ITALY}")
        print(f"  repetitions: 1 | CV folds: 2 | federated clients: [1, 2]")
        print(f"  weighting  : {weighting} | voting={VOTING}")
        print(f"  output     : prefix={extra_kwargs['prefix']!r} (real results are never overwritten)")
        print("=" * 60)

    if args.cohort_table:
        # Supplementary Table 1. summarize_cohorts() computed exactly this content but was never
        # called anywhere -- the table the response letter says "was updated" did not exist.
        _exp = next((e for e in experiments if e.name == 'count_taxa_meteor_0_lgbm'), experiments[0])
        _anno, _ = load_experiment(_exp, data_dir)
        _tbl = summarize_cohorts(_anno, label_col=_exp.label if isinstance(_exp.label, str) else 'health_status')
        table_dir.mkdir(parents=True, exist_ok=True)
        _out = table_dir / 'supplementary_table1_cohorts.csv'
        _tbl.to_csv(_out, index=False)
        print(f"Supplementary Table 1 -> {_out}  ({len(_tbl)} cohorts)")
        print(_tbl.to_string(index=False))
        print(f"\ntrainable cohorts: {int(_tbl['trainable'].sum())} | "
              f"validation-only: {int((~_tbl['trainable']).sum())} | total samples: {int(_tbl['n'].sum())}")
        raise SystemExit(0)

    if not run_experiments:
        raise SystemExit(f"No experiments matched --dataset={args.dataset!r}.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        simulations = ([s.strip() for s in args.simulations.split(',')]
                       if args.simulations else list(DEFAULT_SIMULATIONS))

        # Effective-config banner: everything that determines the result, printed so a run is
        # self-documenting. With no CLI flags these are exactly the config.toml (paper) values.
        if not args.dry_run:
            print("=" * 64)
            print(f"  config     : {_CONFIG_PATH.name}  (CLI flags override it)")
            _eff_models = args.models if args.models else (','.join(DEFAULT_MODELS) if DEFAULT_MODELS else None)
            print(f"  experiments: {len(run_experiments)}"
                  + (f"  (--dataset={args.dataset})" if args.dataset else "  (all)")
                  + (f"  (models={_eff_models})" if _eff_models else "  (full grid)"))
            print(f"  simulations: {', '.join(simulations)}")
            print(f"  repetition : {REPETITION} | cross_val: {CROSS_VAL} | seed: {RANDOM_STATE} | exec: {EXEC_MODE}")
            print(f"  tune baseline: {TUNE} | tune clients: {TUNE_CLIENTS}")
            print(f"  weighting  : {weighting} | voting: {VOTING} | rounds: {rounds}")
            print(f"  max_combos_per_k: {max_combos_per_k if max_combos_per_k else 'exhaustive'}"
                  f" | cohorts: merge_germany={MERGE_GERMANY} split_italy={SPLIT_ITALY}")
            if args.clients:
                print(f"  clients (k-sweep): {args.clients}  (federated-family arms only)")
            if args.prefix:
                print(f"  output prefix: {args.prefix!r}")
            print("=" * 64)

        for simulation_type in simulations:
            print(f"\n-----------------------------\n")
            print(f"Running {simulation_type} simulations...")
            run_simulation_with_threading(
                run_experiments,
                scoring=scoring,
                simulation_type=simulation_type,
                feature_importance='',
                verbose=verbose,
                EXEC_MODE=EXEC_MODE,
                log_file=log_file,
                weighting=weighting,
                max_threads=N_JOBS,
                threads_per_job=THREADS_PER_JOB,
                merge_germany=MERGE_GERMANY,
                split_italy=SPLIT_ITALY,
                **extra_kwargs,
            )

        print("\n-----------------------------\n")
        if args.dry_run:
            print("Dry run complete.")
