# Federated Learning Allows for Privacy-Aware Microbiome Profiling for Colorectal Cancer Detection

Code for the paper of the same name. It simulates federated learning for colorectal cancer (CRC)
detection from gut metagenomes: 14 cohorts from 15 public studies, each cohort acting as one site. A
federated model is a one-shot voting ensemble of models trained separately on k cohorts and tested on a
held-out cohort. It is compared with centralized cross-validation, models trained and tested within a
single cohort, models transferred from one cohort to another, a model trained on the pooled data of the
same k cohorts, and two multi-round federated boosting schemes. The models are LightGBM, a random
forest (LightGBM in random-forest mode), a linear SVM and logistic regression, on METEOR2, MetaPhlAn4
and Sylph profiles.

The code, the input matrices and every result table and figure are deposited as one zip in the research
data repository of Universität Hamburg:
[doi:10.25592/uhhfdm.21960](https://doi.org/10.25592/uhhfdm.21960). In the unzipped folder, `plot.py`
and `paper_numbers.py` rebuild every figure and number in minutes; the full simulation takes several
days on a large server.

## Running

```bash
uv sync                          # Python 3.12 environment from uv.lock
./run_full_server.sh             # every step below, in order
uv run python plot.py figure1    # one step on its own, for example in the unzipped deposit
uv run python paper_numbers.py
```

Run every script from the repository root; all paths are relative to it. `config.toml` holds the
parameters shared by every run, and every command-line flag of `run_analysis.py` overrides the matching
key. The runs are deterministic: every seed derives from `random_state` in `config.toml`. Hyperparameter
searches are cached in `data/simulations/tune_cache/`; the cache changes speed, never results, and
`FEMAI_TUNE_CACHE=off` turns it off.

`run_analysis.py --simulations` takes any of:

| Simulation | What it does |
| --- | --- |
| `central` | stratified fivefold cross-validation on all training cohorts pooled |
| `local` | stratified fivefold cross-validation within each cohort |
| `federated` | one-shot voting over every combination of k training cohorts, tested on each held-out cohort; at k = 1, a model transferred from one cohort to another |
| `pooled` | one model on the pooled data of the same k cohorts, tested on the same held-out cohorts |
| `fedboost` | parallel federated boosting (LightGBM): every site boosts from the shared model in each round, and the server averages the increments |
| `cyclic_boost` | sequential federated boosting (LightGBM): the model passes from site to site |

## Steps

`run_full_server.sh` runs these steps in order. Tables go to `data/simulations/tables/`, figures to
`data/simulations/figures/`; the comment over each command names its tables.

| Step | Script | Tables |
| --- | --- | --- |
| (0) | `run_analysis.py --cohort-table` | `supplementary_table1_cohorts.csv`: the cohorts after quality control |
| (1) | `run_analysis.py`, trees on METEOR2 species and strain, every combination | `count_taxa_meteor_0_{rf,lgbm}_{central,local,federated,pooled}.csv`, `count_strain_meteor_nan_0_{central,local,federated,pooled}.csv` |
| (2) | `run_analysis.py`, SVM and LogReg on METEOR2 species, every combination | `count_taxa_meteor_0_{svm_l2,log_l2}_{central,federated,pooled}.csv` |
| (3) | `run_analysis.py`, boosting, LGBM on METEOR2 species | `count_taxa_meteor_0_lgbm_{fedboost,cyclic_boost}.csv` |
| (4) | `run_analysis.py`, boosting on METEOR2 strain | `count_strain_meteor_nan_0_{fedboost,cyclic_boost}.csv` |
| (5) | `run_analysis.py`, LGBM on MetaPhlAn4 and Sylph | `count_taxa_{metaphlan4,sylph}_0_lgbm_{central,local,federated,pooled}.csv` |
| (6) | `run_analysis.py`, LGBM keeping the samples without age, sex or BMI | `sensitivity_permissive_lgbm_{central,local,federated,pooled}.csv` |
| (7) | `run_analysis.py`, uniform against size-weighted voting | `chk_weighting_count_taxa_meteor_0_lgbm_federated.csv` |
| (8) | `run_analysis.py --check-invariant`: the federation must not beat matched pooled on the tree models | none (prints a report) |
| (9) | `hpo_central.py` (trees) and `run_tunedA_parallel.py` (SVM and LogReg, using `tuned_central.py`): centralized evaluation of the 16 model variants | `hpo_<profile>_{rf,lgbm}*_central.csv`, `tunedA_l2norm_<profile>_svm*_central.csv`, `tunedA_zscore_<profile>_log*_central.csv`, and linear `hpo_` tables that no figure uses |
| (10) | `plot.py figure1`, `supp-fig1`, `supp-fig2`, `supp-fig3` | the figures, each with its plotted values in a `.csv` next to it |
| (11) | `paper_numbers.py` | `paper_numbers.csv`, `supplementary_table{3_aggregation,4_full_results,5_sensitivity}.csv`, `inversion_rates.csv` |

`paper_numbers.csv` lists every number the paper reports, one per row: an id, where it appears, its
value, and the table and filter it is computed from.

## Files

| File | Role |
| --- | --- |
| `run_full_server.sh` | every step, in order |
| `run_analysis.py` | the simulations, the cohort table and the invariant check; steps (0) – (8) |
| `utility.py` | shared helpers of `run_analysis.py`: experiments, cross-validation, scoring |
| `PreFittedVotingClassifier.py` | the one-shot voting ensemble and the boosted federated model |
| `hpo_central.py` | centralized evaluation of the 16 model variants on the four profiles, step (9) |
| `tuned_central.py` | centralized evaluation of the SVM and LogReg variants, with C tuned inside each training fold |
| `run_tunedA_parallel.py` | runs `tuned_central.py`'s folds in parallel for step (9) |
| `plot.py` | the figures, step (10) |
| `paper_numbers.py` | the paper's numbers and summary tables, step (11) |
| `prep_data_modular.py` | the data preparation that wrote the six input matrices from the raw profiles. It needs the raw data, which is not included, and is not part of `run_full_server.sh`. |
| `config.toml` | the shared run parameters: seed, repetitions, folds, tuning, model grid, cohort definitions |
| `pyproject.toml`, `uv.lock`, `.python-version` | the Python environment |
| `deposit_list.txt` | every data file in the deposit, and how to build the deposit zip |
| `data/simulations/figures/**/*.csv` | the plotted values of the four figures |

## Data

The pipeline starts from six files in `prep_data/`, all in the deposit:

| File | Content |
| --- | --- |
| `anno_full_clean.csv` | sample annotation after quality control, 1,722 samples with age, sex and BMI |
| `anno_full_nan.csv` | the same before removing samples without age, sex or BMI, 1,953 samples |
| `taxa_counts_meteor_0.csv` | METEOR2 species pseudocounts |
| `strain_meteor_nan.csv` | METEOR2 strain pseudocounts, missing values kept |
| `taxa_meta4.csv` | MetaPhlAn4 species counts |
| `taxa_sylph.csv` | Sylph species counts |

They are built from the preprocessed pseudocounts and curated metadata of Barbet et al., Recherche Data
Gouv, [doi:10.57745/7IVO3E](https://doi.org/10.57745/7IVO3E). The raw sequence data are in the European
Nucleotide Archive under the accessions listed in the paper.

## License

The code and the deposited data are licensed under the Creative Commons Attribution 4.0 International
license (CC BY 4.0); see `LICENSE`. Please cite the paper when you use them.
