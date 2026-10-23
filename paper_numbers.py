#!/usr/bin/env python
"""Every number the paper quotes, recomputed from the published tables.

Reads the result tables under data/simulations/tables, plus the two annotation files the pipeline starts
from (for the sample counts before the metadata filter and the size of Germany2 before the merge). No
model is fitted. Writes, into the table directory:

    paper_numbers.csv                      every in-text number of the manuscript, its supplement and
                                           the response to reviewers: id, where it appears, value, and
                                           the table and filter it comes from
    supplementary_table3_aggregation.csv   Supplementary Table 3: the aggregation strategies against
                                           matched pooled, by k
    supplementary_table4_full_results.csv  the full results, deposited but not printed in the
                                           supplement: centralized evaluation of the 16 variants on the
                                           four profiles; federated and matched pooled evaluation at
                                           every k for the four models on METEOR2, and for LGBM on the
                                           other profiles
    supplementary_table5_sensitivity.csv   the sensitivity analysis that keeps the samples without age,
                                           sex or BMI (LGBM), deposited but not printed in the supplement
    inversion_rates.csv                    the fraction of (training combination, held-out cohort) pairs
                                           below AUROC 0.5, by k, over held-out training cohorts (the
                                           paper's scope) and over every held-out cohort

Conventions, the same in every output:
    * federated and matched pooled means are the held-out-training-cohort k-curve that Supplementary
      Figure 3 plots (plot.k_curve);
    * every SD of a federated or matched pooled value is the SD across held-out cohorts: the SD of the
      per-cohort means over every combination and repetition;
    * centralized evaluation has no held-out cohort, so its SD is across the 25 test folds (5 folds x
      5 repetitions); section (B) quotes the fold-to-fold SD within each cohort, and the intra-cohort
      row of the sensitivity analysis the SD across cohorts.

The two deposit-only tables keep the file names they had while they were Supplementary Tables 4 and 5.

Run from inside code/, after the simulations (run_full_server.sh):

    uv run python paper_numbers.py
"""
import argparse
from pathlib import Path

import pandas as pd

from plot import METRIC, k_curve, stem, training_cohorts

DATASET = 'count_taxa_meteor_0'
# (code name, id, paper name, regularisation): the four models the paper reports
MODELS = [('lgbm', 'lgbm', 'LGBM', 'none'), ('rf', 'rf', 'RF', 'none'),
          ('svm_l2', 'svm', 'SVM', 'L2'), ('log_l2', 'logreg', 'LogReg', 'L2')]
# The four profiles and 16 variants of Supplementary Figure 1, as hpo_central.DATASETS,
# hpo_central.LINEAR_SCALING and plot.py name them. Not imported: hpo_central loads the whole pipeline
# through run_analysis.
PROFILES = [('count_taxa_meteor_0', 'METEOR2'), ('count_strain_meteor_nan_0', 'METEOR2 (strain)'),
            ('count_taxa_metaphlan4_0', 'MetaPhlAn4'), ('count_taxa_sylph_0', 'Sylph')]
LINEAR_SCALING = {'svm': 'l2norm', 'log': 'zscore'}
FAMILIES = [('rf', 'RF'), ('lgbm', 'LGBM'), ('svm', 'SVM'), ('log', 'LogReg')]
REGS = [('', 'none'), ('_l1', 'L1'), ('_l2', 'L2'), ('_elasticnet', 'ElasticNet')]
# LGBM federated and matched pooled on the other profiles. The strain experiment name carries no model.
OTHER_PROFILE_RUNS = [('count_taxa_metaphlan4_0', 'lgbm', 'MetaPhlAn4'), ('count_taxa_sylph_0', 'lgbm', 'Sylph'),
                      ('count_strain_meteor_nan_0', '', 'METEOR2 (strain)')]
STRATEGIES = [('federated', 'One-shot voting'), ('fedboost', 'Parallel boosting'),
              ('cyclic_boost', 'Sequential boosting'), ('pooled', 'Matched pooled')]
HELD_OUT = 'held-out training cohorts'


class Tables:
    """Reads each published table once; a missing table stops the run with the file name."""

    def __init__(self, table_dir):
        self.dir = Path(table_dir)
        self._cache = {}

    def __call__(self, name):
        if name not in self._cache:
            if not self.exists(name):
                raise SystemExit(f'missing published table {self.dir / name}.csv. '
                                 'Run the simulations first (run_full_server.sh).')
            self._cache[name] = pd.read_csv(self.dir / f'{name}.csv')
        return self._cache[name]

    def exists(self, name):
        return (self.dir / f'{name}.csv').exists()


def held_out_curve(df):
    """Mean AUROC by k over held-out training cohorts (plot.k_curve), plus the SD across them."""
    curve = k_curve(df)
    d = df[df['val_dataset'].isin(training_cohorts(df))].dropna(subset=[METRIC])
    per_cohort = d.groupby(['n_clients', 'val_dataset'])[METRIC].mean()
    curve['sd'] = per_cohort.groupby('n_clients').std()
    return curve


def inversions(df, cohorts):
    """Per k: (training combination, held-out cohort) pairs whose mean AUROC over repetitions is below
    0.5, among the scorable pairs (a single-class held-out cohort has no AUROC)."""
    d = df[df['val_dataset'].isin(cohorts)]
    pairs = d.groupby(['n_clients', 'train_dataset', 'val_dataset'])[METRIC].mean().dropna()
    by_k = pairs.groupby('n_clients')
    out = pd.DataFrame({'inverted': by_k.apply(lambda s: int((s < 0.5).sum())), 'scorable': by_k.size()})
    out['percent'] = 100 * out['inverted'] / out['scorable']
    return out.rename_axis('k').reset_index()


def central_tables(profile):
    """The 16 centralized cells of one profile, as Supplementary Figure 1 reads them: trees from the
    central screen (hpo_), SVM and LogReg from the tuned centralized runs."""
    for fam, flab in FAMILIES:
        for reg, rlab in REGS:
            if fam in LINEAR_SCALING:
                yield flab, rlab, f'tunedA_{LINEAR_SCALING[fam]}_{profile}_{fam}{reg}_central'
            else:
                yield flab, rlab, f'hpo_{profile}_{fam}{reg}_central'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table-dir', default='data/simulations/tables')
    ap.add_argument('--anno-dir', default='prep_data', help='where anno_full_clean.csv and anno_full_nan.csv are')
    args = ap.parse_args(argv)
    table = Tables(args.table_dir)
    numbers = []

    def put(id_, where, value, source):
        value = round(float(value), 6) if isinstance(value, float) else value
        numbers.append({'id': id_, 'where': where, 'value': value, 'source': source})

    # --- centralized evaluation (A) ---------------------------------------------------------------
    centralized = {}
    for code, key, _, _ in MODELS:
        t = stem(DATASET, code, 'central')
        scores = table(t)[METRIC]
        centralized[code] = scores.mean()
        put(f'centralized_{key}', 'Results overview and (A); response R1.3, R2.4', scores.mean(), f'{t}.csv')
        put(f'centralized_sd_{key}', 'Response R1.3, R2.4 (SD across the 25 test folds)', scores.std(), f'{t}.csv')

    # --- federated evaluation (D) and matched pooled, the four models ------------------------------
    for code, key, _, _ in MODELS:
        fed_t, pool_t = stem(DATASET, code, 'federated'), stem(DATASET, code, 'pooled')
        fed, pool = held_out_curve(table(fed_t)), held_out_curve(table(pool_t))
        src = f'{fed_t}.csv; k = 10, {HELD_OUT}'
        put(f'federated_k10_{key}', 'Results overview; response R1.3, R2.4', fed.loc[10, 'mean'], src)
        put(f'federated_sd_k10_{key}', 'Response R1.3, R2.4 (SD across held-out cohorts)', fed.loc[10, 'sd'], src)
        put(f'matched_pooled_k10_{key}', 'Results overview and (D); response R2.5', pool.loc[10, 'mean'],
            f'{pool_t}.csv; k = 10, {HELD_OUT}')
        gap = (pool['mean'] - fed['mean']).dropna()
        src = f'{pool_t}.csv minus {fed_t}.csv; k-curves over {HELD_OUT}'
        put(f'gap_k10_{key}', 'Results overview; response R2.1 (matched pooled minus federated, k = 10)',
            gap.loc[10], src)
        put(f'gap_mean_{key}', 'Abstract and (D) (mean over k = 1 - 10)', gap.mean(), src)
        put(f'gap_max_{key}', '(D); response R2.1 (maximum over k)', gap.max(), src)
        put(f'gap_max_k_{key}', '(D) (k of the maximum gap)', int(gap.idxmax()), src)
        put(f'gap_absmax_{key}', 'Results overview (largest |gap| over k)', gap.abs().max(), src)
        put(f'centralized_minus_federated_{key}', 'Response R1.3, R2.4 (centralized minus federated at k = 10)',
            centralized[code] - fed.loc[10, 'mean'],
            f'{stem(DATASET, code, "central")}.csv minus {fed_t}.csv at k = 10')

    # --- Supplementary Table 3: aggregation strategies, LGBM -----------------------------------------
    st3, strategy_curves = [], {}
    for arm, _ in STRATEGIES:
        strategy_curves[arm] = held_out_curve(table(stem(DATASET, 'lgbm', arm)))
    boost_ks = strategy_curves['fedboost'].index
    for k in boost_ks:
        for arm, label in STRATEGIES:
            c = strategy_curves[arm]
            st3.append({'k': int(k), 'strategy': label, 'mean_auroc': c.loc[k, 'mean'], 'sd_auroc': c.loc[k, 'sd']})
    for arm, key in (('fedboost', 'parallel_boosting'), ('cyclic_boost', 'sequential_boosting')):
        put(f'{key}_k10_lgbm', '(D); Supplementary Figure 3', strategy_curves[arm].loc[10, 'mean'],
            f'{stem(DATASET, "lgbm", arm)}.csv; k = 10, {HELD_OUT}')
    # The boosting arms ran on a subsample of the combinations. Caption: one-shot voting and matched
    # pooled on that subsample against their values over every combination.
    combo = ['rep', 'n_clients', 'train_dataset']
    subsample = table(stem(DATASET, 'lgbm', 'fedboost'))[combo].drop_duplicates()
    deviation = 0.0
    for arm in ('federated', 'pooled'):
        on_subsample = held_out_curve(table(stem(DATASET, 'lgbm', arm)).merge(subsample, on=combo))
        deviation = max(deviation, (on_subsample['mean'] - strategy_curves[arm]['mean']).dropna().abs().max())
    put('boost_subset_max_deviation', 'Supplementary Table 3 caption ("at most 0.003")', deviation,
        f'{stem(DATASET, "lgbm", "federated")}.csv and {stem(DATASET, "lgbm", "pooled")}.csv, restricted to the '
        f'(rep, k, combination) of {stem(DATASET, "lgbm", "fedboost")}.csv, against all combinations')

    # --- full results (deposit only) --------------------------------------------------------------------
    st4 = []
    for profile, plabel in PROFILES:
        for flab, rlab, t in central_tables(profile):
            # The strain linear cells have no table: the linear models cannot be fitted on the NaN-rich
            # strain matrix, and Supplementary Figure 1 shows them as n/a. They stay in the table, empty.
            scores = table(t)[METRIC].dropna() if table.exists(t) else pd.Series(dtype=float)
            st4.append({'evaluation': 'Centralized', 'profiler': plabel, 'model': flab, 'regularisation': rlab,
                        'k': None, 'mean_auroc': scores.mean() if len(scores) else None,
                        'sd_auroc': scores.std() if len(scores) else None})
            if flab == 'SVM' and rlab == 'ElasticNet' and len(scores):
                put(f'centralized_svm_elasticnet_{plabel}', 'Supplementary Figure 1 caption (AUROC 0.500)',
                    scores.mean(), f'{t}.csv')
    runs = [(DATASET, code, 'METEOR2', name, reg) for code, _, name, reg in MODELS]
    runs += [(d, m, plabel, 'LGBM', 'none') for d, m, plabel in OTHER_PROFILE_RUNS]
    for dataset, code, plabel, name, reg in runs:
        for arm, evaluation in (('federated', 'Federated'), ('pooled', 'Matched pooled')):
            c = held_out_curve(table(stem(dataset, code, arm)))
            st4 += [{'evaluation': evaluation, 'profiler': plabel, 'model': name, 'regularisation': reg,
                     'k': int(k), 'mean_auroc': r['mean'], 'sd_auroc': r['sd']} for k, r in c.iterrows()]
    for dataset, code, plabel in OTHER_PROFILE_RUNS[:2]:
        t = stem(dataset, code, 'central')
        put(f'centralized_lgbm_{plabel}', 'Methods, Data Selection (LGBM on the other profiles)',
            table(t)[METRIC].mean(), f'{t}.csv')
        for arm, key in (('federated', 'federated'), ('pooled', 'matched_pooled')):
            t = stem(dataset, code, arm)
            put(f'{key}_k10_lgbm_{plabel}', 'Methods, Data Selection (LGBM on the other profiles)',
                held_out_curve(table(t)).loc[10, 'mean'], f'{t}.csv; k = 10, {HELD_OUT}')

    # --- sensitivity analysis, LGBM (deposit only) -----------------------------------------------------
    st5 = []
    permissive_ks = held_out_curve(table(stem('sensitivity_permissive', 'lgbm', 'federated'))).index
    for data, dataset, model in (('main', DATASET, 'lgbm'), ('permissive', 'sensitivity_permissive', 'lgbm')):
        central = table(stem(dataset, model, 'central'))[METRIC]
        st5.append({'evaluation': 'Centralized', 'data': data, 'k': None,
                    'mean_auroc': central.mean(), 'sd_auroc': central.std()})
        per_cohort = table(stem(dataset, model, 'local')).groupby('val_dataset')[METRIC].mean()
        st5.append({'evaluation': 'Intra-cohort', 'data': data, 'k': None, 'mean_auroc': per_cohort.mean(),
                    'sd_auroc': per_cohort.std()})
        for arm, evaluation in (('federated', 'Federated'), ('pooled', 'Matched pooled')):
            c = held_out_curve(table(stem(dataset, model, arm)))
            # the k the permissive run swept; its full participation is k = 11 (12 training cohorts with
            # China4), so the main data adds its own, k = 10
            ks = permissive_ks if data == 'permissive' else [k for k in permissive_ks if k in c.index] + [c.index.max()]
            st5 += [{'evaluation': evaluation, 'data': data, 'k': int(k), 'mean_auroc': c.loc[k, 'mean'],
                     'sd_auroc': c.loc[k, 'sd']} for k in sorted(set(ks))]

    # --- intra-cohort evaluation (B) ---------------------------------------------------------------------
    t = stem(DATASET, 'lgbm', 'local')
    local = table(t).groupby('val_dataset')[METRIC]
    for cohort, mean in local.mean().items():
        put(f'intra_mean_{cohort}', '(B)', mean, f'{t}.csv')
        put(f'intra_sd_{cohort}', '(B) (SD across the 25 test folds)', local.std()[cohort], f'{t}.csv')
    put('intra_weakest', '(B) ("the three weakest standalone models")',
        ', '.join(local.mean().nsmallest(3).index), f'{t}.csv')
    put('intra_strongest', '(D) ("the cohorts with the strongest intra-cohort models")',
        ', '.join(local.mean().nlargest(2).index), f'{t}.csv')

    # --- inversions (C) and (D) --------------------------------------------------------------------------
    t = stem(DATASET, 'lgbm', 'federated')
    fed = table(t)
    inv = pd.concat([inversions(fed, training_cohorts(fed)).assign(held_out='training cohorts'),
                     inversions(fed, set(fed['val_dataset'])).assign(held_out='all cohorts')])
    inv = inv[['held_out', 'k', 'inverted', 'scorable', 'percent']]
    tr = inv[inv['held_out'] == 'training cohorts'].set_index('k')
    for k in (1, 2):
        src = f'{t}.csv; k = {k}, {HELD_OUT}, mean over repetitions per pair'
        put(f'inversions_k{k}', '(C) and (D)', int(tr.loc[k, 'inverted']), src)
        put(f'pairs_k{k}', '(C) and (D)', int(tr.loc[k, 'scorable']), src)
        put(f'inversion_pct_k{k}', '(C) and (D)', tr.loc[k, 'percent'], src)
    clean_from = next(k for k in tr.index if tr.loc[k:, 'inverted'].sum() == 0)
    put('first_k_no_inversions', '(D) ("0 % starting at three")', int(clean_from), f'{t}.csv; {HELD_OUT}')

    # --- federation against the intra-cohort models, (D) and Conclusion -----------------------------------
    # Each training cohort held out at k = 10, i.e. the federation of the other ten, against the
    # intra-cohort model of that cohort. Uses `fed` and `t` from the inversions section above and `local`
    # from section (B).
    k10 = fed[(fed['n_clients'] == 10) & fed['val_dataset'].isin(training_cohorts(fed))]
    held_out_k10 = k10.groupby('val_dataset')[METRIC].mean()
    src = f'{t}.csv; k = 10, the cohort held out, mean over repetitions'
    for cohort, mean in held_out_k10.items():
        put(f'federated_k10_lgbm_heldout_{cohort}',
            '(D) (the cohorts with the weakest and strongest intra-cohort models, held out)', mean, src)
    gain = held_out_k10 - local.mean()[held_out_k10.index]
    src = f'{t}.csv at k = 10 against {stem(DATASET, "lgbm", "local")}.csv, per cohort'
    put('federated_k10_lgbm_beats_intra', 'Conclusion ("performs better on five of the eleven training cohorts")',
        int((gain > 0).sum()), src)
    put('federated_k10_lgbm_largest_gains', 'Conclusion ("its gain is largest for the cohorts with the weakest '
        'local models"); (D)', ', '.join(gain.nlargest(3).index), src)

    # --- weighting check (Methods, Model Selection) --------------------------------------------------------
    t = 'chk_weighting_' + stem(DATASET, 'lgbm', 'federated')
    for weighting, g in table(t).groupby('weighting'):
        put(f'weighting_k10_lgbm_{weighting}', 'Methods, Model Selection (uniform against size weighting)',
            held_out_curve(g).loc[10, 'mean'], f'{t}.csv; weighting = {weighting}, k = 10, {HELD_OUT}')

    # --- cohorts and samples (Methods; response R2.3) ------------------------------------------------------
    t = 'supplementary_table1_cohorts'
    cohorts = table(t)
    for _, r in cohorts.iterrows():
        put(f'cohort_n_{r.cohort_name}', '(B); Supplementary Table 2; response R2.3', int(r.n), f'{t}.csv')
    trainable = cohorts['trainable'].astype(bool)
    put('cohorts_total', 'Abstract; Methods, Cohort Quality Control', len(cohorts), f'{t}.csv')
    put('cohorts_training', 'Methods, Cohort Quality Control', int(trainable.sum()), f'{t}.csv')
    put('cohorts_validation', 'Methods, Cohort Quality Control', int((~trainable).sum()), f'{t}.csv')
    put('samples_final', 'Methods, Cohort Quality Control', int(cohorts['n'].sum()), f'{t}.csv')
    put('samples_training', 'Methods, Cohort Quality Control', int(cohorts.loc[trainable, 'n'].sum()), f'{t}.csv')
    put('samples_validation', 'Methods, Cohort Quality Control', int(cohorts.loc[~trainable, 'n'].sum()), f'{t}.csv')
    anno_dir = Path(args.anno_dir)
    clean, permissive = pd.read_csv(anno_dir / 'anno_full_clean.csv'), pd.read_csv(anno_dir / 'anno_full_nan.csv')
    put('samples_after_exclusions', 'Methods, Cohort Quality Control', len(permissive), 'anno_full_nan.csv')
    put('samples_missing_metadata', 'Methods, Cohort Quality Control', len(permissive) - len(clean),
        'anno_full_nan.csv minus anno_full_clean.csv')
    germany2 = clean[clean['cohort_name'] == 'Germany2']
    put('germany2_n', 'Response R2.3 (before the merge into Germany)', len(germany2), 'anno_full_clean.csv')
    put('germany2_healthy', 'Response R2.3', int((germany2['class'] == 'healthy').sum()), 'anno_full_clean.csv')

    # --- write ---------------------------------------------------------------------------------------------
    out = Path(args.table_dir)
    outputs = {'paper_numbers': pd.DataFrame(numbers),
               'supplementary_table3_aggregation': pd.DataFrame(st3),
               'supplementary_table4_full_results': pd.DataFrame(st4).astype({'k': 'Int64'}),
               'supplementary_table5_sensitivity': pd.DataFrame(st5).astype({'k': 'Int64'}),
               'inversion_rates': inv}
    for name, df in outputs.items():
        df.round(6).to_csv(out / f'{name}.csv', index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 70)
    print(outputs['paper_numbers'].drop(columns='source').to_string(index=False))

    def mean_sd(df, index, columns):
        df = df.assign(cell=df['mean_auroc'].map('{:.3f}'.format) + ' (' + df['sd_auroc'].map('{:.3f}'.format) + ')')
        return df.pivot(index=index, columns=columns, values='cell')

    print('\nSupplementary Table 3, mean AUROC (SD across held-out cohorts):')
    print(mean_sd(outputs['supplementary_table3_aggregation'], 'k', 'strategy')[[s for _, s in STRATEGIES]].to_string())
    print('\nSensitivity analysis (deposit only):')
    print(mean_sd(outputs['supplementary_table5_sensitivity'].fillna({'k': 0}), ['evaluation', 'k'], 'data').to_string())
    print(f'\nwrote {", ".join(f"{out / n}.csv" for n in outputs)}')


if __name__ == '__main__':
    main()
