#!/usr/bin/env python
"""Every figure in the paper, drawn from the published tables.

    figure1     Figure 1 (bottom): A centralized, B intra-cohort, C inter-cohort and D federated evaluation
                of LGBM on METEOR2, training cohorts only. AUROC (top row) and MCC (bottom row); the paper
                shows the AUROC row, cropped by hand and placed under the hand-drawn schematic.
    supp-fig1   Supplementary Figure 1: the 16 model variants under centralized evaluation, on the four
                feature matrices.
    supp-fig2   Supplementary Figure 2: as Figure 1, with training and validation cohorts, both rows.
    supp-fig3   Supplementary Figure 3: the federated aggregation strategies against matched pooled, by k.

Each subcommand writes its figure and, next to it, a CSV of the plotted values. Nothing here fits a
model: run the simulations first (run_full_server.sh). Run from inside code/, the default paths are
relative:

    uv run python plot.py figure1
    uv run python plot.py supp-fig1
    uv run python plot.py supp-fig2
    uv run python plot.py supp-fig3

Every subcommand takes --table-dir and --fig-dir (defaults data/simulations/{tables,figures}).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import seaborn as sns

METRIC = 'test_roc_auc'

# --- colour ---------------------------------------------------------------------------------------
# seaborn tab20 with the red pair removed — red is kept out for viewers with a red-green deficiency,
# and reserved for the central model (Figure 1 draws the central box and its reference line in 'r').
COHORT_PALETTE = [
    '#1f77b4', '#aec7e8',   # blue
    '#ff7f0e', '#ffbb78',   # orange
    '#2ca02c', '#98df8a',   # green
    '#9467bd', '#c5b0d5',   # purple
    '#8c564b', '#c49c94',   # brown
    '#e377c2', '#f7b6d2',   # pink/magenta
    '#7f7f7f', '#c7c7c7',   # gray
    '#bcbd22', '#dbdb8d',   # olive/lime
    '#17becf', '#9edae5',   # teal/cyan
]

# --- cohort identity ------------------------------------------------------------------------------
# Canonical names (Germany merged; Italy1 split into Italy1a/Italy1b). China2 and China4 are absent:
# every sample of both lacks the demographic metadata the models take as features (age, sex, BMI).
TRAIN_COHORTS = [
    'Austria1', 'Brazil1',
    'China1',
    'China3',
    'China5', 'France1',
    'Germany', 'Italy1a', 'Italy1b',
    'Japan1',
    'USA1',
]

VAL_COHORTS = [
    # validation only: single-phenotype (India1, India2) or under the 40-sample threshold (USA2, n=30).
    'India1', 'India2', 'USA2',
]


# --- tables ---------------------------------------------------------------------------------------
def stem(dataset, model, arm=None):
    """Join the non-empty name parts with '_'. Handles datasets whose experiment name already carries
    the model (e.g. strain: 'count_strain_meteor_nan_0' with model='' -> no dangling '__')."""
    return '_'.join(p for p in (dataset, model, arm) if p)


def add_io_args(ap):
    """The --table-dir/--fig-dir pair every subcommand takes."""
    ap.add_argument('--table-dir', default='data/simulations/tables')
    ap.add_argument('--fig-dir', default='data/simulations/figures')


# --- the published scope --------------------------------------------------------------------------
def training_cohorts(df):
    """Union of every cohort that appears as a training client — the held-out scope the paper reports.

    Validation-only cohorts are excluded from every k-curve because at large k the held-out set would
    be dominated by the tiny single-class India cohorts.
    """
    return set(df['train_dataset'].astype(str).str.split(', ').explode().unique())


def k_curve(df, col=METRIC):
    """Per-k mean with SEM across repetitions, over held-out TRAINING cohorts only, plus the rep count.
    Within a repetition, the mean runs over (combination x held-out cohort). None if nothing is left.

    Reps differ because the combination subsample is seeded per rep, so the spread is real sampling
    variation — for lgbm the model itself is deterministic across reps, so any width here is the
    subsample, not run-to-run noise.
    """
    d = df[df['val_dataset'].isin(training_cohorts(df))].dropna(subset=[col])
    if d.empty:
        return None
    g = d.groupby(['rep', 'n_clients'])[col].mean().reset_index().groupby('n_clients')[col]
    return pd.DataFrame({'mean': g.mean(),
                         'sem': g.std(ddof=0) / np.sqrt(g.count()),
                         'n_reps': g.count()}).sort_index()


# --- Figure 1 and Supplementary Figure 2 ----------------------------------------------------------
EVAL_EXPERIMENT = 'count_taxa_meteor_0_lgbm'
EVAL_METRICS = ('test_roc_auc', 'test_mcc')
EVAL_TITLE = f'{EVAL_EXPERIMENT} test_roc_auc test_mcc federated analysis - line by eval'


def load_eval_table(table_dir, simulation_type, cohort_order):
    """One arm's table, with the cohorts as ordered categories; cohorts outside the order become NaN."""
    df = pd.read_csv(Path(table_dir) / f'{EVAL_EXPERIMENT}_{simulation_type}.csv')
    df['val_dataset'] = pd.Categorical(df['val_dataset'], categories=cohort_order, ordered=True)
    df['train_dataset_single'] = pd.Categorical(df['train_dataset'], categories=cohort_order, ordered=True)
    return df


def eval_values(fed, central, local, cohort_sizes):
    """The plotted values of the four panels, one row per box or point, for both metrics."""
    def describe(panel, metric, values, cohort='', k=None):
        v = values.dropna()
        return {'panel': panel, 'metric': metric, 'cohort': cohort, 'n_clients': k,
                'n_samples': cohort_sizes.get(cohort), 'n': len(v), 'mean': v.mean(), 'sd': v.std(),
                'median': v.median(), 'q1': v.quantile(0.25), 'q3': v.quantile(0.75)}

    rows = []
    for m in EVAL_METRICS:
        rows.append(describe('A', m, central[m], 'All Cohorts'))
        for c, g in local.loc[local['n_clients'] == 1].groupby('val_dataset', observed=True):
            rows.append(describe('B', m, g[m], c))
        for c, g in fed.loc[fed['n_clients'] == 1].groupby('train_dataset_single', observed=True):
            rows.append(describe('C', m, g[m], c))
        for (k, c), g in fed.loc[fed['n_clients'] > 0].groupby(['n_clients', 'val_dataset'], observed=True):
            rows.append(describe('D', m, g[m], c, k))
    out = pd.DataFrame(rows)
    out['n_samples'] = out['n_samples'].astype('Int64')
    out['n_clients'] = out['n_clients'].astype('Int64')
    return out.round(4)


def plot_double_linebox(_df, _central_df, _local_df, cohort_sizes, palette, save_path):
    """Draw the four panels for AUROC (top) and MCC (bottom); write the PNG and the plotted values."""
    metric1, metric2 = EVAL_METRICS
    target_path = Path(save_path) / 'double_linebox' / "_".join([metric1, metric2])
    target_path.mkdir(exist_ok=True, parents=True)
    vmin1, vmax1 = (0.0, 1.0)
    vmin2, vmax2 = (-0.5, 1.0)

    # remove highest number of clients
    _df =  _df.loc[_df['n_clients'] < _df['n_clients'].max()]

    fig = plt.figure(figsize=(12, 8))

    row, col = 2, 4
    gs = GridSpec(row, col, width_ratios=[1, 3, 3, 4])

    axes = [fig.add_subplot(gs[i, j]) for i in range(row) for j in range(col)]

    flierprops = dict(marker='o', markersize=3, markeredgewidth=0.2)

    # Plot for metric1
    sns.boxplot(data=_central_df, y=metric1, ax=axes[0], color='r')
    axes[0].set_title('Central\nModel')
    axes[0].set_ylabel('AUROC', size='large')
    axes[0].set_xlabel("All Cohorts")
    axes[0].set_xticks([])
    axes[0].set_ylim((vmin1, vmax1))

    sns.boxplot(data=_local_df.loc[_local_df['n_clients'] == 1], x='n_clients', y=metric1, hue='val_dataset', palette=palette, ax=axes[1],
                flierprops=flierprops)

    axes[1].axhline(_central_df[metric1].mean(), color='r', linestyle='--')
    axes[1].set_xticks([])
    axes[1].set_ylim((vmin1, vmax1))
    axes[1].set_xlabel("Individual Cohorts\n(intra-cohort)")
    axes[1].set_ylabel("")
    axes[1].set_title('Local Models\n')
    axes[1].legend().remove()

    sns.boxplot(data=_df.loc[_df['n_clients'] == 1], x='n_clients', y=metric1, hue='train_dataset_single', palette=palette, ax=axes[2],
                flierprops=flierprops)

    axes[2].axhline(_central_df[metric1].mean(), color='r', linestyle='--')
    axes[2].set_xticks([])
    axes[2].set_ylim((vmin1, vmax1))
    axes[2].set_xlabel("Individual Cohorts\n(inter-cohort)")
    axes[2].set_ylabel("")
    # Panel C is a single locally trained model applied to the other cohorts, NOT the federated model
    # of panel D. It used to be titled "FEDERATED Models n=1", which reviewers read as the proposed
    # method performing badly across cohorts.
    axes[2].set_title('Inter-Cohort Evaluation\n(by training cohort)')
    axes[2].legend().remove()


    sns.pointplot(data=_df.loc[_df['n_clients'] > 0], x='n_clients', y=metric1, hue='val_dataset',
                  palette=palette, dodge=True, ax=axes[3], marker=".", markersize=10, markeredgewidth=3,
                  err_kws={'linewidth': 0.5}, errorbar='sd')

    axes[3].axhline(_central_df[metric1].mean(), color='r', linestyle='--')
    axes[3].set_ylim((vmin1, vmax1))
    axes[3].set_ylabel("")
    axes[3].set_xlabel("Number of Cohorts")
    axes[3].set_title("Federated Evaluation\n(by validation cohort)")
    axes[3].legend().remove()

    handles, labels = axes[2].get_legend_handles_labels()

    # Plot for metric2
    sns.boxplot(data=_central_df, y=metric2, ax=axes[4], color='r')

    axes[4].set_ylabel('MCC', size='large')
    axes[4].set_xlabel("All Cohorts")
    axes[4].set_ylim((vmin2, vmax2))
    axes[4].set_xticks([])

    sns.boxplot(data=_local_df.loc[_local_df['n_clients'] == 1], x='n_clients', y=metric2, hue='val_dataset', palette=palette, ax=axes[5],
                flierprops=flierprops)

    axes[5].axhline(_central_df[metric2].mean(), color='r', linestyle='--')
    axes[5].set_xticks([])
    axes[5].set_ylim((vmin2, vmax2))
    axes[5].set_xlabel("Individual Cohorts\n(intra-cohort)")
    axes[5].set_ylabel("")
    axes[5].legend().remove()

    sns.boxplot(data=_df.loc[_df['n_clients'] == 1], x='n_clients', y=metric2, hue='train_dataset_single', palette=palette, ax=axes[6],
                flierprops=flierprops)

    axes[6].axhline(_central_df[metric2].mean(), color='r', linestyle='--')
    axes[6].set_xticks([])
    axes[6].set_ylim((vmin2, vmax2))
    axes[6].set_xlabel("Individual Cohorts\n(inter-cohort)")
    axes[6].set_ylabel("")
    axes[6].legend().remove()

    sns.pointplot(data=_df.loc[_df['n_clients'] > 0], x='n_clients', y=metric2, hue='val_dataset',
                  palette=palette, dodge=True, ax=axes[7], marker=".", markersize=10, markeredgewidth=3,
                  err_kws={'linewidth': 0.5}, errorbar='sd')

    axes[7].axhline(_central_df[metric2].mean(), color='r', linestyle='--')
    axes[7].set_ylim((vmin2, vmax2))
    axes[7].set_ylabel("")
    axes[7].set_xlabel("Number of Cohorts")
    axes[7].legend().remove()


    # Reviewer 2: mark AUROC = 0.5 (random guessing). The full 0-1 axis is kept on purpose so that
    # below-0.5 values (systematic inverse prediction, panel C) remain visible rather than clipped.
    for ax in axes[:4]:
        ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, zorder=0)

    fig.suptitle(EVAL_TITLE)
    # Fixed at 2 rows regardless of entry count: 11 training cohorts -> 6 wide, 14 (training +
    # validation-only) -> 7 wide. ceil(n/2) keeps it 2-tall for any legend size.
    ncol = (len(labels) + 1) // 2

    for ax, letter in zip([axes[0], axes[1], axes[2], axes[3]], ['A', 'B', 'C', 'D']):
        ax.text(-0.1, 1.15, letter, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='center')

    labels = [f"{label} ({cohort_sizes.get(label, 0)})" for label in labels]

    fig.legend(handles, labels, title='(B) (D) Validation Cohorts / (C) Training Cohorts', bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=ncol)

    fig.tight_layout()
    png = Path(target_path, f'{EVAL_TITLE.replace(" ", "_")}.png')
    fig.savefig(png, bbox_inches='tight')
    plt.close(fig)

    csv = png.with_suffix('.csv')
    eval_values(_df, _central_df, _local_df, cohort_sizes).to_csv(csv, index=False)
    print(f'wrote {png}\nwrote {csv}')


def evaluation_figure(table_dir, fig_dir, cohort_order, scope):
    """Figure 1 (scope 'training_cohorts') or Supplementary Figure 2 (scope 'all_cohorts')."""
    # Silenced, as the old driver did: pandas warns about the cohorts outside the order, seaborn about
    # the palette being longer than the cohort list. Both are intended.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        central = load_eval_table(table_dir, 'central', cohort_order)
        local = load_eval_table(table_dir, 'local', cohort_order)
        fed = load_eval_table(table_dir, 'federated', cohort_order)
        # The legend's sample counts: the cohort table `run_analysis.py --cohort-table` writes from the
        # annotation of this experiment, with the canonical cohort names.
        cohorts = pd.read_csv(Path(table_dir) / 'supplementary_table1_cohorts.csv')
        cohort_sizes = dict(zip(cohorts['cohort_name'], cohorts['n']))
        plot_double_linebox(fed, central, local, cohort_sizes, sns.color_palette(COHORT_PALETTE),
                            save_path=Path(fig_dir, scope, 'lgbm'))


def figure1(args):
    evaluation_figure(args.table_dir, args.fig_dir, TRAIN_COHORTS, 'training_cohorts')


def supp_fig2(args):
    evaluation_figure(args.table_dir, args.fig_dir, TRAIN_COHORTS + VAL_COHORTS, 'all_cohorts')


# --- Supplementary Figure 1 -----------------------------------------------------------------------
# A seaborn boxplot of the per-fold AUROC, x = feature matrix, hue = model variant, styled after the
# model screen of the sibling publication ("The bigger the better?",
# `combinations_publication/code/central_hyperparameter_opt.py::plot_hyperparameter_summary`). The figure
# carries no interpretation: which variant wins is left to the caption and to the CSV.
#
# The tree cells come from the `hpo_` tables (hpo_central.py), the SVM and LogReg cells from the tuned
# centralized tables (`tunedA_<scaling>_<experiment>_<model>_central.csv`, run_tunedA_parallel.py), each
# with the scaling hpo_central.LINEAR_SCALING names: unit-length normalisation for the SVM, z-score
# standardisation for LogReg, the one setup per model the paper reports. The profiles, the scalings and
# the prefix are those of hpo_central.DATASETS, LINEAR_SCALING and PREFIX. Not imported: hpo_central
# loads the whole pipeline through run_analysis.
#
# Not every cell exists. The METEOR2 strain matrix is ~22% NaN (an absent strain is signal, and LGBM
# splits on NaN natively), and its annotation leaves ~11% of age/sex/BMI missing. The linear models
# cannot ingest NaN, and the strain matrix has no tuned tables, so those eight cells are drawn as "n/a",
# never as a gap the eye reads as zero.
SCREEN_STEM = 'model_comparison_central_auroc_by_dataset_tuned'
SCREEN_PROFILES = [('count_taxa_meteor_0', 'METEOR2'), ('count_strain_meteor_nan_0', 'METEOR2 (strain)'),
                   ('count_taxa_metaphlan4_0', 'MetaPhlAn4'), ('count_taxa_sylph_0', 'Sylph')]
LINEAR_SCALING = {'svm': 'l2norm', 'log': 'zscore'}
HPO_PREFIX = 'hpo_'
FAMILIES = [('rf', 'RF'), ('lgbm', 'LGBM'), ('svm', 'SVM'), ('log', 'LogReg')]
REGS = [('', 'none'), ('_l1', 'L1'), ('_l2', 'L2'), ('_elasticnet', 'ElasticNet')]

MODEL_ORDER = [f'{fam}{reg}' for fam, _ in FAMILIES for reg, _ in REGS]
MODEL_LABEL = {f'{fam}{reg}': (flab if rlab == 'none' else f'{flab} + {rlab}')
               for fam, flab in FAMILIES for reg, rlab in REGS}

# 16 model hues = 4 family hues x 4 shades, one shade per regularisation. Same construction as the
# sibling paper's plot_production_params_central.py, which is what makes 16 levels legible at all.
_SHADES = [1.0, 0.78, 0.56, 0.34]


def model_palette():
    base = dict(zip([f for f, _ in FAMILIES], ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']))
    pal = {}
    for fam, _ in FAMILIES:
        rgb = np.array(matplotlib.colors.to_rgb(base[fam]))
        for (reg, _), f in zip(REGS, _SHADES):
            pal[f'{fam}{reg}'] = tuple(rgb + (1 - rgb) * (1 - f))
    return pal


def screen_table_path(table_dir, name, model):
    """The tuned centralized table of a linear cell, or the `hpo_` table of a tree cell."""
    family = model.split('_')[0]
    if family in LINEAR_SCALING:
        return Path(table_dir) / f'tunedA_{LINEAR_SCALING[family]}_{name}_{model}_central.csv'
    return Path(table_dir) / f'{HPO_PREFIX}{name}_{model}_central.csv'


def load_screen(table_dir):
    """Long frame of per-fold scores: dataset x model variant, plus the cells that have no value."""
    rows, missing = [], []
    for name, label in SCREEN_PROFILES:
        for model in MODEL_ORDER:
            p = screen_table_path(table_dir, name, model)
            if not p.exists():
                missing.append((label, model))
                continue
            df = pd.read_csv(p)
            vals = df[METRIC].dropna() if METRIC in df else pd.Series(dtype=float)
            if vals.empty:
                missing.append((label, model))
                continue
            for v in vals:
                rows.append({'dataset': label, 'model': model, METRIC: float(v)})
    if not rows:
        raise SystemExit(f'No {HPO_PREFIX}*_central.csv tables in {table_dir}. Run hpo_central.py first.')
    return pd.DataFrame(rows), missing


def annotate_missing(ax, missing, x_levels, hue_levels):
    """Mark the cells that have no value, so a gap never reads as a low score."""
    n = len(hue_levels)
    width = 0.8 / n
    for label, model in missing:
        if label not in x_levels or model not in hue_levels:
            continue
        x = x_levels.index(label) - 0.4 + width * (hue_levels.index(model) + 0.5)
        ax.text(x, ax.get_ylim()[0] + 0.012, 'n/a', ha='center', va='bottom', fontsize=7,
                color='#999999', rotation=90)


def supp_fig1(args):
    d, missing = load_screen(args.table_dir)
    datasets = [lab for _, lab in SCREEN_PROFILES if lab in set(d['dataset'])]
    models = [m for m in MODEL_ORDER if m in set(d['model'])]
    d['dataset'] = pd.Categorical(d['dataset'], categories=datasets, ordered=True)
    d['model'] = pd.Categorical(d['model'], categories=models, ordered=True)
    palette = model_palette()

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sns.boxplot(data=d, x='dataset', y=METRIC, hue='model', order=datasets, hue_order=models,
                palette=palette, ax=ax, linewidth=0.8, fliersize=2,
                flierprops=dict(marker='o', markersize=2, markeredgewidth=0.2, alpha=0.5))

    ax.axhline(0.5, color='#444444', linestyle=':', linewidth=1.6, zorder=1, label='_nolegend_')

    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('AUROC (central, pooled 5-fold CV)', fontsize=11)
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_title('Central hyperparameter optimization', fontsize=12.5, pad=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#ececec', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    annotate_missing(ax, missing, datasets, models)

    handles = [Patch(facecolor=palette[h], edgecolor='#666666', linewidth=0.6, label=MODEL_LABEL[h])
               for h in models]
    ax.legend(handles=handles, title='Models Variations', bbox_to_anchor=(0.5, -0.16),
              loc='upper center', ncol=4, frameon=False, fontsize=9.5)

    fig.tight_layout()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(fig_dir / f'{SCREEN_STEM}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    summary = (d.groupby(['dataset', 'model'], observed=True)[METRIC]
               .agg(['mean', 'median', 'std', 'count']).round(4).reset_index())
    summary.to_csv(fig_dir / f'{SCREEN_STEM}.csv', index=False)

    print(f'wrote {fig_dir / f"{SCREEN_STEM}.png"}')
    print(f'wrote {fig_dir / f"{SCREEN_STEM}.csv"}  ({len(summary)} of '
          f'{len(datasets) * len(MODEL_ORDER)} cells)')
    if missing:
        print(f'not fittable ({len(missing)}): '
              + ', '.join(f'{lab}/{m}' for lab, m in missing))


# --- Supplementary Figure 3 -----------------------------------------------------------------------
# The aggregation strategies against matched pooled, as a function of k, for LGBM on METEOR2. Scope:
# held-out TRAINING cohorts only (k_curve). The dotted line is centralized evaluation.
ARMS_DATASET, ARMS_MODEL = 'count_taxa_meteor_0', 'lgbm'

# (table suffix, label, colour, marker, linestyle, is_reference). Colours are Okabe-Ito (colour-blind
# safe). `pooled` is the reference: the honest ceiling, same k cohorts pooled into one training set.
ARMS = [
    ('pooled',       'Centralized (pooled, matched k)', '#000000', 'o', '--', True),
    ('federated',    'Federated (one-shot voting)',     '#0072B2', 's', '-',  False),
    ('fedboost',     'Federated boosting (parallel)',   '#009E73', '^', '-',  False),
    ('cyclic_boost', 'Federated boosting (cyclic)',     '#D55E00', 'v', '-',  False),
]


def supp_fig3(args):
    def table(arm):
        return pd.read_csv(Path(args.table_dir) / f'{stem(ARMS_DATASET, ARMS_MODEL, arm)}.csv')

    curves, rows = {}, []
    for arm, label, *_ in ARMS:
        c = k_curve(table(arm))
        curves[arm] = c
        for k, r in c.iterrows():
            rows.append({'arm': arm, 'label': label, 'k': int(k),
                         'auroc': round(r['mean'], 4), 'sem': round(r['sem'], 4), 'n_reps': int(r['n_reps'])})

    central = float(table('central')[METRIC].dropna().mean())

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for arm, label, colour, marker, ls, is_ref in ARMS:
        c = curves[arm]
        ax.plot(c.index, c['mean'], marker=marker, linestyle=ls, color=colour, label=label,
                linewidth=2 if not is_ref else 1.8, markersize=5, zorder=3)
        ax.fill_between(c.index, c['mean'] - c['sem'], c['mean'] + c['sem'], color=colour, alpha=0.15, zorder=1)

    ax.axhline(central, color='#555555', ls=':', lw=1.3, zorder=2)
    ax.text(ax.get_xlim()[1], central, f'  central CV {central:.3f}', va='center', ha='left',
            fontsize=8, color='#555555')

    ax.set_xlabel('Number of participating cohorts $k$')
    ax.set_ylabel('AUROC (held-out training cohorts)')
    ax.set_title(f'Federated strategies vs centralized — {ARMS_DATASET} ({ARMS_MODEL})', fontsize=11)
    xs = sorted({int(k) for c in curves.values() for k in c.index})
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.margins(x=0.02)
    fig.tight_layout()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    name = stem(ARMS_DATASET, ARMS_MODEL, 'federated_comparison')
    # Default bbox, not tight: the published PNG was cropped that way.
    png = fig_dir / f'{name}.png'
    fig.savefig(png, dpi=300)
    plt.close(fig)
    csv = fig_dir / f'{name}.csv'
    summary = pd.DataFrame(rows).sort_values(['arm', 'k'])
    summary.to_csv(csv, index=False)

    print(f"\nHeld-out-training-cohort AUROC by k  ({ARMS_DATASET}_{ARMS_MODEL}):\n")
    wide = summary.pivot(index='k', columns='arm', values='auroc')
    print(wide[[a for a, *_ in ARMS]].to_string())
    print(f"\n  central (in-distribution 5-fold CV): {central:.4f}")
    print(f"\nwrote {png}\nwrote {csv}")


# --- CLI ------------------------------------------------------------------------------------------
COMMANDS = {
    'figure1':   (figure1,   'Figure 1: the four evaluation strategies, training cohorts'),
    'supp-fig1': (supp_fig1, 'Supplementary Figure 1: the 16 model variants, centralized evaluation'),
    'supp-fig2': (supp_fig2, 'Supplementary Figure 2: as Figure 1, training and validation cohorts'),
    'supp-fig3': (supp_fig3, 'Supplementary Figure 3: the aggregation strategies against matched pooled'),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='figure', required=True, metavar='figure')
    for name, (_, help_) in COMMANDS.items():
        add_io_args(sub.add_parser(name, help=help_))
    args = ap.parse_args(argv)
    COMMANDS[args.figure][0](args)


if __name__ == '__main__':
    sys.exit(main())
