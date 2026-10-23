#!/usr/bin/env bash
#
# run_full_server.sh — every command behind a published table of the paper, in order, plus the
# strain-level boosting of step (4), which completes the strain-level run but feeds no paper item.
#
# Each step writes the tables named in its comment into data/simulations/tables/ (figures into
# data/simulations/figures/); README.md lists the steps. The flags are the ones the published tables
# were run with; config.toml holds the shared parameters (seed, repetitions, folds, tuning, model grid,
# cohort definitions).
#
# The full run takes several days on a large server, most of it in the exhaustive federated sweep of
# steps (1) and (2) and the strain-level boosting of step (4). The deposit (doi:10.25592/uhhfdm.21960)
# holds every published table, so steps (10) and (11) can be run on it directly.
#
# Cores: (nproc - 5) for the run, THREADS_PER_JOB threads per experiment (default 4). These change
# speed, never results.
#
#   ./run_full_server.sh
#   THREADS_PER_JOB=8 ./run_full_server.sh
#
set -euo pipefail
cd "$(dirname "$0")"

TOTAL_CORES="$(nproc)"
CORES=$(( TOTAL_CORES - 5 )); (( CORES < 1 )) && CORES=1
TPJ="${THREADS_PER_JOB:-4}"
RES=(--n-jobs "$CORES" --threads-per-job "$TPJ")

echo "=================================================================="
echo " Federated microbiome CRC paper — every published table"
echo "   core budget : ${CORES} of ${TOTAL_CORES}  (threads/job ${TPJ})"
echo "=================================================================="

# Environment (Python 3.12 venv from uv.lock)
uv sync

# (0) Cohorts after quality control (Supplementary Table 2). -> supplementary_table1_cohorts.csv
uv run python run_analysis.py --cohort-table "${RES[@]}"

# (1) Trees on METEOR2 species and strain: centralized, intra-cohort, federated and matched pooled,
#     every combination of training cohorts at every k (--max-combos-per-k 0). --dataset meteor matches
#     count_taxa_meteor_0_{rf,lgbm} and count_strain_meteor_nan_0 (LGBM).
#     -> count_taxa_meteor_0_{rf,lgbm}_{central,local,federated,pooled}.csv,
#        count_strain_meteor_nan_0_{central,local,federated,pooled}.csv
uv run python run_analysis.py --dataset meteor --models rf,lgbm \
    --simulations central,local,federated,pooled --redo "${RES[@]}" --max-combos-per-k 0

# (2) Linear models on METEOR2 species, one setup per model (SVM unit-length, LogReg z-score):
#     centralized, federated and matched pooled, every combination. No paper item uses their intra-cohort evaluation.
#     -> count_taxa_meteor_0_{svm_l2,log_l2}_{central,federated,pooled}.csv
uv run python run_analysis.py --dataset count_taxa_meteor_0 --models svm_l2,log_l2 \
    --simulations central,federated,pooled --redo "${RES[@]}" --max-combos-per-k 0

# (3) Parallel and sequential boosting, LGBM on METEOR2 species: at most 60 combinations per k and
#     repetition, k in {1, 2, 3, 5, 10}. -> count_taxa_meteor_0_lgbm_{fedboost,cyclic_boost}.csv
uv run python run_analysis.py --dataset count_taxa_meteor_0 --models lgbm \
    --simulations fedboost,cyclic_boost --redo "${RES[@]}" --max-combos-per-k 60 --clients 1,2,3,5,max-1

# (4) The same boosting on METEOR2 strain, 3 repetitions, k in {1, 2, 3, 5, 11}. Part of the full
#     strain-level run; no paper item reads these two tables.
#     -> count_strain_meteor_nan_0_{fedboost,cyclic_boost}.csv
uv run python run_analysis.py --dataset count_strain_meteor_nan_0 \
    --simulations fedboost,cyclic_boost --redo "${RES[@]}" --repetition 3 --max-combos-per-k 60 \
    --clients 1,2,3,5,max-1

# (5) LGBM on the other two profilers, MetaPhlAn4 and Sylph: at most 20 combinations per k and
#     repetition, k in {1, 2, 3, 5, 10}.
#     -> count_taxa_{metaphlan4,sylph}_0_lgbm_{central,local,federated,pooled}.csv
uv run python run_analysis.py --dataset metaphlan4 --models lgbm \
    --simulations central,local,federated,pooled --redo "${RES[@]}" --max-combos-per-k 20 --clients 1,2,3,5,max-1
uv run python run_analysis.py --dataset sylph --models lgbm \
    --simulations central,local,federated,pooled --redo "${RES[@]}" --max-combos-per-k 20 --clients 1,2,3,5,max-1

# (6) Sensitivity analysis (deposit only): LGBM on the annotation that keeps the 231 samples
#     without age, sex or BMI. China4 returns as a 12th training cohort, so k is in {1, 2, 3, 5, 11}.
#     -> sensitivity_permissive_lgbm_{central,local,federated,pooled}.csv
uv run python run_analysis.py --dataset sensitivity_permissive --models lgbm \
    --simulations central,local,federated,pooled --redo "${RES[@]}" --max-combos-per-k 20 --clients 1,2,3,5,max-1

# (7) Weighting check (Methods, Model Selection): uniform against sample-size weighting of the vote,
#     LGBM on METEOR2 species at k = 10. -> chk_weighting_count_taxa_meteor_0_lgbm_federated.csv
uv run python run_analysis.py --dataset count_taxa_meteor_0 --models lgbm \
    --simulations federated --redo "${RES[@]}" --clients max-1 --weighting none,size --prefix chk_weighting_

# (8) Invariant gate: the federation must not beat matched pooled on the tree models; the linear
#     models are only reported, since federated and matched pooled are tuned alike. Reads the tables of
#     steps (1) – (4), writes nothing. Loud,
#     but non-fatal so that the figures still run; the script exits non-zero at the end if it failed.
GATE_OK=1
uv run python run_analysis.py --dataset meteor --check-invariant "${RES[@]}" || {
  GATE_OK=0
  echo "!!! INVARIANT GATE FAILED — the federation beat matched pooled on a tree model. Inspect before using results."
}

# (9) Centralized evaluation of the 16 model variants on the four profilers (Supplementary Figure 1,
#     the full results table). The tree cells come from the centralized screen, the SVM and LogReg cells from
#     the tuned centralized runs: SVM with unit-length normalisation, LogReg with z-score
#     standardisation, as in every other evaluation. The strain matrix has no linear cells (n/a).
#     -> hpo_<profile>_{rf,lgbm}*_central.csv, tunedA_l2norm_<profile>_svm*_central.csv,
#        tunedA_zscore_<profile>_log*_central.csv
uv run python hpo_central.py "${RES[@]}"
TUNED_CELLS=()
for d in count_taxa_meteor_0 count_taxa_metaphlan4_0 count_taxa_sylph_0; do
  for m in svm svm_l1 svm_l2 svm_elasticnet; do TUNED_CELLS+=("$d:$m:l2norm"); done
  for m in log log_l1 log_l2 log_elasticnet; do TUNED_CELLS+=("$d:$m:zscore"); done
done
uv run python run_tunedA_parallel.py --n-jobs "$CORES" --out data/simulations/tables \
    --cells "$(IFS=,; echo "${TUNED_CELLS[*]}")"

# (10) Figure 1 and Supplementary Figures 1, 2 and 3 (read the published tables, fit nothing)
uv run python plot.py figure1 \
    || echo "WARNING: Figure 1 failed (analysis results are saved)."
uv run python plot.py supp-fig1 \
    || echo "WARNING: Supplementary Figure 1 failed (analysis results are saved)."
uv run python plot.py supp-fig2 \
    || echo "WARNING: Supplementary Figure 2 failed (analysis results are saved)."
uv run python plot.py supp-fig3 \
    || echo "WARNING: Supplementary Figure 3 failed (analysis results are saved)."

# (11) In-text numbers and Supplementary Tables 3, 4 and 5 (read the published tables, fit nothing)
#      -> paper_numbers.csv, supplementary_table{3_aggregation,4_full_results,5_sensitivity}.csv,
#         inversion_rates.csv
uv run python paper_numbers.py \
    || echo "WARNING: paper_numbers.py failed (analysis results are saved)."

echo "DONE — results under data/simulations/{raw_results,tables,figures}/"
if [[ "${GATE_OK}" -eq 0 ]]; then
  echo "COMPLETED WITH AN INVARIANT VIOLATION — review the gate output before trusting the results."
  exit 1
fi
