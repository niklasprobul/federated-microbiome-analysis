"""Data preparation: how the raw profiles became the input matrices. Kept as documentation only.

Status: this module is not part of the pipeline. No script imports it, and it cannot be re-run from
the data deposit, because the raw profiles and metadata it reads are not deposited. The pipeline starts
from the six files below.

Source: `prep_data_modular.py` from the `femai_publication` repository at commit 47d7967 (2025-06-04,
"fix strain+taxa file generation"). Everything below this header is that file, byte for byte. One run
of it at that commit wrote the six input files. The functions it calls from `utility.py` (`parse_df`)
are unchanged since then. Re-run on the raw inputs, it writes the four count matrices byte for byte,
and the two annotation files byte for byte under pandas 2.2 (the version pinned at that commit). Under
pandas 3, the annotation files write integers such as `0` where pandas 2 wrote `0.0`, and they read
back as the same values and dtypes.

Run from the repository root (`uv run python prep_data_modular.py`), it reads `./prep_data/raw/` and
writes into `./prep_data/`. The pipeline reads six of the files it writes:

    anno_full_clean.csv       sample annotation without the samples flagged for exclusion, without
                              health status or with adenoma, and without the samples that lack age,
                              sex or BMI
    anno_full_nan.csv         the same, but keeping the samples that lack age, sex or BMI (used by the
                              strain-level runs and the sensitivity analysis)
    taxa_counts_meteor_0.csv  METEOR2 species counts
    strain_meteor_nan.csv     METEOR2 strain-level profile, rescaled to integer counts; NaN where a
                              species was found but its strain could not be resolved
    taxa_meta4.csv            MetaPhlAn4 species counts
    taxa_sylph.csv            Sylph species counts

It needs these raw inputs under `./prep_data/raw/`:

    metadata_2340_CRC_cohort_20240704.xlsx, sheet metadata_2340_CRC_cohort    both annotation files
    batch_effect_corrected_species_filtered0_2340_ech/
        species_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev0.csv
                                                                              taxa_counts_meteor_0.csv
    abundance-strain-gut-oral.tsv                                             strain_meteor_nan.csv
    metaphlan4_CRC_count_table_2340_CRC_cohort_20240607.tsv                   taxa_meta4.csv
    species_signal_sylph_count_table_2340_CRC_cohort_20250306.tsv             taxa_sylph.csv

Every other entry in `datasets` (functional profiles, batch-effect corrections, the imputed strain
profile and others) is written whenever its sources exist (raw inputs, or for the merged entries
earlier outputs in `./prep_data/`), and no paper item uses it. The external-drive path at the top
of `prep_data` is the author's machine: where it does not exist, the raw inputs are read from
`./prep_data/raw/`.
"""
import pandas as pd
import os 
from scipy.stats import zscore
import utility
from pathlib import Path
import numpy as np

def prep_data(normalize=False, verbose=False, anno_only=False, split_cohort_by_country=True):
    # %%
    parent_dir = './prep_data/raw/'
    ssd_path_in = '/media/niklas/T7/data/FeMAI_RAW/'
    if Path(ssd_path_in).is_dir():   
        parent_dir = os.path.join(ssd_path_in, parent_dir)
        
    output_dir = 'prep_data/'

    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir(exist_ok=True)
    
    # anno_df = pd.read_excel(os.path.join(parent_dir, "metadata_2090_CRC_cohort_20230117.xlsx"), 'metadata_2090_CRC_cohort')
    # expr_df = pd.read_csv(os.path.join(parent_dir, "species_signal_2090_CRC_cohort_20230201.tsv"), sep='\t', index_col=0)
    # func_df = pd.read_csv(os.path.join(parent_dir, "functional_signal_2090_CRC_cohort_20221207.tsv"), sep='\t', index_col=0)
    # taxa_df = pd.read_csv(os.path.join(parent_dir, "species_taxo_phylo_20230201.tsv"), sep='\t', index_col=0)
    # modu_df = pd.read_csv(os.path.join(parent_dir, "modules_definition_GMM_GBM_KEGG_92.tsv"), sep='\t')
    # expr_meta_df = pd.read_csv(os.path.join(parent_dir, "metaphlan2_CRC_2090_species_signal_20230710.tsv"), sep='\t', index_col=0)
    
    # anno_df = pd.read_excel(os.path.join(parent_dir, "metadata_2250_CRC_cohort_20231114.xlsx"), 'metadata_1696_CRC_cohort')
    # expr_df = pd.read_csv(os.path.join(parent_dir, "species_signal_2250_CRC_cohort_20231115.tsv"), sep='\t', index_col=0)
    # func_df = pd.read_csv(os.path.join(parent_dir, "functional_signal_2250_CRC_cohort_20231114.tsv"), sep='\t', index_col=0)
    # taxa_df = pd.read_csv(os.path.join(parent_dir, "species_taxo_phylo_20231115.tsv"), sep='\t', index_col=0)
    # modu_df = pd.read_csv(os.path.join(parent_dir, "modules_definition_GMM_GBM_KEGG_92_20231114.tsv"), sep='\t')
    # expr_meta_df = pd.read_csv(os.path.join(parent_dir, "metaphlan2_CRC_2090_species_signal_20230710.tsv"), sep='\t', index_col=0)
    
    # # input file names 2250
    # anno = "metadata_2250_CRC_cohort_20231114.xlsx"
    # anno_sheet = "metadata_1696_CRC_cohort"
    # expr = "species_signal_2250_CRC_cohort_20231115.tsv"
    # func = "functional_signal_2250_CRC_cohort_20231114.tsv"
    # taxa = "species_taxo_phylo_20231115.tsv"
    # modu = "modules_definition_GMM_GBM_KEGG_92_20231114.tsv"
    # expr_meta = "metaphlan2_CRC_2090_species_signal_20230710.tsv"
    
    datasets = {
        "anno": {
            # "input": "metadata_2340_CRC_cohort_20240426.xlsx", # less strict quality controll
            "input": "metadata_2340_CRC_cohort_20240704.xlsx",
            "sheet": "metadata_2340_CRC_cohort",
            "output": "anno_full_clean.csv",
        },
        "anno_nan": {
            # "input": "metadata_2340_CRC_cohort_20240426.xlsx", # less strict quality controll
            "input": "metadata_2340_CRC_cohort_20240704.xlsx",
            "sheet": "metadata_2340_CRC_cohort",
            "output": "anno_full_nan.csv",
        },
        "anno_multi": {
            "input": "metadata_2340_CRC_cohort_20240704.xlsx",
            "sheet": "metadata_2340_CRC_cohort",
            "output": "anno_multi.csv",
        },
        "anno_raw": {
            "input": "metadata_2340_CRC_cohort_20240704.xlsx",
            "sheet": "metadata_2340_CRC_cohort",
            "output": "anno_raw.csv",
        },
        "taxa_meteor": {
            "input": "species_signal_2340_CRC_cohort_20240322.tsv",
            "output": "taxa_meteor.csv",
        },
        "taxa_counts_meteor_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev0.csv",
            "output": "taxa_counts_meteor_0.csv",
        },
        "taxa_counts_meteor_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev10.csv",
            "output": "taxa_counts_meteor_10.csv",
        },
        "taxa_counts_meteor_0_log": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev0.csv",
            "output": "taxa_counts_meteor_0_log.csv",
        },
        "strain_meteor_imputed": {
            "input": "abundance-strain-gut-oral.tsv",
            "output": "strain_meteor_imputed.csv",
        },
        "strain_meteor_nan": {
            "input": "abundance-strain-gut-oral.tsv",
            "output": "strain_meteor_nan.csv",
        },
        "func_meteor": {
            "input": "functional_signal_2340_CRC_cohort_20240329.tsv",
            "output": "func_meteor.csv",
        },
        "func_counts_meteor_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev0.tsv",
            "output": "func_counts_meteor_0.csv",
        },
        "func_counts_meteor_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev10.tsv",
            "output": "func_counts_meteor_10.csv",
        },
        "taxa_meta": {
            "input": "metaphlan4_CRC_count_table_2340_CRC_cohort_20240607.tsv",
            "output": "taxa_meta4.csv",
        },
        "taxa_sylph":{
            "input": "species_signal_sylph_count_table_2340_CRC_cohort_20250306.tsv",
            "output": "taxa_sylph.csv",
        },
        "modules": {
            "input": "modules_definition_GMM_GBM_KEGG_92_20240329.tsv",
            "output": "modules.csv",
        },
        "species": {
            "input": "species_taxo_phylo_20240322.tsv",
            "output": "species.csv",
        },
        "french_batch_bat_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_combat_prev0.csv",
            "output": "mbec_corrected_bat_prev0.csv",
        },
        "french_batch_pls_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_plsda_prev0.csv",
            "output": "mbec_corrected_pls_prev0.csv",
        },
        "french_batch_rbe_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_rbe_prev0.csv",
            "output": "mbec_corrected_rbe_prev0.csv",
        },
        "french_batch_svd_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_svd_prev0.csv",
            "output": "mbec_corrected_svd_prev0.csv",
        },
        "french_batch_bmc_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_BMC_prev0.csv",
            "output": "mbec_corrected_bmc_prev0.csv",
        },
        "french_batch_bat_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_combat_prev10.csv",
            "output": "mbec_corrected_bat_prev10.csv",
        },
        "french_batch_pls_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_plsda_prev10.csv",
            "output": "mbec_corrected_pls_prev10.csv",
        },
        "french_batch_rbe_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_rbe_prev10.csv",
            "output": "mbec_corrected_rbe_prev10.csv",
        },
        "french_batch_svd_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_svd_prev10.csv",
            "output": "mbec_corrected_svd_prev10.csv",
        },
        "french_batch_bmc_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_BMC_prev10.csv",
            "output": "mbec_corrected_bmc_prev10.csv",
        },
        "french_batch_func_bat_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_combat_correction_prev0.tsv",
            "output": "mbec_corrected_func_bat_prev0.csv",
        },
        "french_batch_func_pls_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_plsda_correction_prev0.tsv",
            "output": "mbec_corrected_func_pls_prev0.csv",
        },
        "french_batch_func_rbe_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_rbe_correction_prev0.tsv",
            "output": "mbec_corrected_func_rbe_prev0.csv",
        },
        "french_batch_func_bmc_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_BMC_correction_prev0.tsv",
            "output": "mbec_corrected_func_bmc_prev0.csv",
        },
        "french_batch_func_svd_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_svd_correction_prev0.tsv",
            "output": "mbec_corrected_func_svd_prev0.csv",
        },
        "french_batch_func_bat_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_combat_correction_prev10.tsv",
            "output": "mbec_corrected_func_bat_prev10.csv",
        },
        "french_batch_func_pls_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_plsda_correction_prev10.tsv",
            "output": "mbec_corrected_func_pls_prev10.csv",
        },
        "french_batch_func_rbe_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_rbe_correction_prev10.tsv",
            "output": "mbec_corrected_func_rbe_prev10.csv",
        },
        "french_batch_func_bmc_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_BMC_correction_prev10.tsv",
            "output": "mbec_corrected_func_bmc_prev10.csv",
        },
        "french_batch_func_svd_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_svd_correction_prev10.tsv",
            "output": "mbec_corrected_func_svd_prev10.csv",
        },
    }

    datasets.update({
        "strain_taxa_meteor": {
            "strain": 'strain_meteor_nan',
            "taxa": "taxa_meteor",
            "output": "strain_taxa_meteor.csv",
        },
        # "taxa_func_meteor": { # might be a mistake
        "taxa_func": {
            "taxa": datasets["taxa_meteor"]["output"],
            "func": datasets["func_meteor"]["output"],
            "output": "taxa_func_meteor.csv",
        },
        "taxa_func_counts_meteor_0": {
            "taxa": datasets["taxa_counts_meteor_0"]["output"],
            "func": datasets["func_counts_meteor_0"]["output"],
            "output": "taxa_func_counts_meteor_0.csv",
        },
        "taxa_func_counts_meteor_10": {
            "taxa": datasets["taxa_counts_meteor_10"]["output"],
            "func": datasets["func_counts_meteor_10"]["output"],
            "output": "taxa_func_counts_meteor_10.csv",
        },
        "french_batch_taxa_func_bat_0": {
            "taxa": datasets["french_batch_bat_0"]["output"],
            "func": datasets["french_batch_func_bat_0"]["output"],
            "output": "mbec_corrected_taxa_func_bat_prev0.csv",
        },
        "french_batch_taxa_func_pls_0": {
            "taxa": datasets["french_batch_pls_0"]["output"],
            "func": datasets["french_batch_func_pls_0"]["output"],
            "output": "mbec_corrected_taxa_func_pls_prev0.csv",
        },
        "french_batch_taxa_func_rbe_0": {
            "taxa": datasets["french_batch_rbe_0"]["output"],
            "func": datasets["french_batch_func_rbe_0"]["output"],
            "output": "mbec_corrected_taxa_func_rbe_prev0.csv",
        },
        "french_batch_taxa_func_svd_0": {
            "taxa": datasets["french_batch_svd_0"]["output"],
            "func": datasets["french_batch_func_svd_0"]["output"],
            "output": "mbec_corrected_taxa_func_svd_prev0.csv",
        },
        "french_batch_taxa_func_bmc_0": {
            "taxa": datasets["french_batch_bmc_0"]["output"],
            "func": datasets["french_batch_func_bmc_0"]["output"],
            "output": "mbec_corrected_taxa_func_bmc_prev0.csv",
        },
        "french_batch_taxa_func_bat_10": {
            "taxa": datasets["french_batch_bat_10"]["output"],
            "func": datasets["french_batch_func_bat_10"]["output"],
            "output": "mbec_corrected_taxa_func_bat_prev10.csv",
        },
        "french_batch_taxa_func_pls_10": {
            "taxa": datasets["french_batch_pls_10"]["output"],
            "func": datasets["french_batch_func_pls_10"]["output"],
            "output": "mbec_corrected_taxa_func_pls_prev10.csv",
        },
        "french_batch_taxa_func_rbe_10": {
            "taxa": datasets["french_batch_rbe_10"]["output"],
            "func": datasets["french_batch_func_rbe_10"]["output"],
            "output": "mbec_corrected_taxa_func_rbe_prev10.csv",
        },
        "french_batch_taxa_func_svd_10": {
            "taxa": datasets["french_batch_svd_10"]["output"],
            "func": datasets["french_batch_func_svd_10"]["output"],
            "output": "mbec_corrected_taxa_func_svd_prev10.csv",
        },
        "french_batch_taxa_func_bmc_10": {
            "taxa": datasets["french_batch_bmc_10"]["output"],
            "func": datasets["french_batch_func_bmc_10"]["output"],
            "output": "mbec_corrected_taxa_func_bmc_prev10.csv",
        },
        # "french_batch_taxa_func_bat_20": {
        #     "taxa": datasets["french_batch_bat_20"]["output"],
        #     "func": datasets["french_batch_func_bat_20"]["output"],
        #     "output": "mbec_corrected_taxa_func_bat_prev20.csv",
        # },
        # "french_batch_taxa_func_pls_20": {
        #     "taxa": datasets["french_batch_pls_20"]["output"],
        #     "func": datasets["french_batch_func_pls_20"]["output"],
        #     "output": "mbec_corrected_taxa_func_pls_prev20.csv",
        # },
        # "french_batch_taxa_func_rbe_20": {
        #     "taxa": datasets["french_batch_rbe_20"]["output"],
        #     "func": datasets["french_batch_func_rbe_20"]["output"],
        #     "output": "mbec_corrected_taxa_func_rbe_prev20.csv",
        # },
    })
    
    clean_cohort_names = {
        'PRJDB4176 - JPN' : 'Japan1',
        'PRJEB10878 - CHN' : 'China1',
        'PRJEB12449 - USA' : 'USA1',
        'PRJEB27928 - DEU' : 'Germany1',
        'PRJEB6070 - FRA' : 'France1', 
        'PRJEB6070 - DEU' : 'Germany2',
        'PRJEB7774 - AUT' : 'Austria1',
        'PRJNA389927 - USA' : 'USA2',
        'PRJNA389927 - CAN' : 'Canada1',
        'PRJNA397112 - IND' : 'India1',
        'PRJNA447983 - ITA' : 'Italy1',
        'PRJNA531273 - IND' : 'India2',
        'PRJNA608088 - CHN' : 'China2',
        'PRJNA429097 - CHN' : 'China3',
        'PRJNA763023 - CHN' : 'China4',
        'PRJNA731589 - CHN' : 'China5',
        'PRJNA961076 - BRA' : 'Brazil1',
        
        'PRJDB4176' : 'Japan1',
        'PRJEB10878': 'China1',
        'PRJEB12449': 'USA1',
        'PRJEB27928': 'Germany1',
        'PRJEB6070' : 'France1/Germany2',
        'PRJEB7774' : 'Austria1',
        'PRJNA389927' : 'USA2/Canada1',
        'PRJNA397112' : 'India1',
        'PRJNA447983' : 'Italy1',
        'PRJNA531273' : 'India2',
        'PRJNA608088' : 'China2',
        'PRJNA429097' : 'China3',
        'PRJNA763023' : 'China4',
        'PRJNA731589' : 'China5',
        'PRJNA961076' : 'Brazil1',

    }
    def get_counts(_df):
        #  get smallest non 0 value and scale the whole df with it
        min_nonzero = _df[_df > 0].min().min()
        cst = 1.0 / min_nonzero

        return (_df * cst).round(decimals=0)
    
    def add_other_strain_buckets(strain_df: pd.DataFrame, taxa_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each sample/column, whenever all strains of a given taxa in strain_df are NaN:
        - set all those strain rows to 0
        - add (or update) one “<taxa>_other” row whose values come from taxa_df for that taxa

        Assumes:
        • strain_df.index labels like "msp_<taxa>_<strain>"
        • taxa_df.index labels exactly "msp_<taxa>"
        • both DataFrames have the same set of columns (samples)

        Returns a new DataFrame with the “_other” rows inserted.
        """
        result = strain_df.copy()

        # 1) extract the full taxa identifier, e.g. "msp_1234"
        taxa_full = result.index.to_series().str.extract(r'^(msp_[^_]+)_')[0]

        # 2) map each taxa → list of its strain-row labels
        rows_by_taxa = {
            t: taxa_full.index[taxa_full == t].tolist()
            for t in taxa_full.unique()
        }

        # 3) find taxa×column blocks that are entirely NaN
        all_nan = result.isna().groupby(taxa_full).all()
        # all_nan: index=taxa_full, columns=samples, values=bool

        # 4) for each taxa that is all-NaN in some columns:
        for taxon, strain_labels in rows_by_taxa.items():
            nan_cols = all_nan.columns[all_nan.loc[taxon]]
            if not nan_cols.empty:
                # 4a) zero out all its strain rows in those columns
                result.loc[strain_labels, nan_cols] = 0

                # 4b) build the “other” index label
                other_label = f"{taxon}"
                # if not already present, add a new row of NaNs
                if other_label not in result.index:
                    result.loc[other_label] = pd.NA

                # 4c) fill the “other” row from taxa_df
                result.loc[other_label, nan_cols] = taxa_df.loc[taxon, nan_cols]

        return result
    

    def read_file(dataset, values):
        # read and format file
        if "input" in values:
            if not os.path.isfile(os.path.join(parent_dir, values["input"])):
                if verbose:
                    print(f"Did not find {dataset} input file: {values['input']}")
                return None
                
            if dataset in ["anno", "anno_raw", "anno_multi", "anno_nan"]:
                _df = pd.read_excel(os.path.join(parent_dir, values["input"]), values["sheet"])
            else:
                # sep = '\t' if values["input"].endswith('.tsv') else ','
                sep = '\t'
                _df = pd.read_csv(os.path.join(parent_dir, values["input"]), sep=sep, index_col=0)

                if dataset not in ["modules", "species"]:
                    _df = _df.T
                    
                
            # if dataset.startswith("functional_"):
            #     pref = dataset.split("_")[2] + "."
            #     _df.index = _df.index.str.replace(pref, '', regex=False)

        else:
            try:
                if dataset in ['strain_taxa_meteor']:
                    strain = read_file(values['strain'], datasets[values['strain']])
                    taxa = read_file(values['taxa'], datasets[values['taxa']])
                    return add_other_strain_buckets(strain, taxa)
                else:
                    left = pd.read_csv(os.path.join(output_dir, values["taxa"]), index_col=0)
                    right = pd.read_csv(os.path.join(output_dir, values["func"]), index_col=0)
            except Exception as e:
                if verbose:
                    print(f"Error merging base files for {dataset}: {e}")
                return None
            
            _df = pd.merge(left=left, left_index=True, right=right, right_index=True)
        
        return _df    
        
    def write_file(_df, dataset, values):
        if "output" in values:
            if os.path.isfile(os.path.join(output_dir, values["output"])):
                if verbose:
                    print(f"overwriting {dataset} output file: {os.path.join(output_dir, values['output'])}")
                    
        _df.to_csv(os.path.join(output_dir, values["output"]))
        
    for dataset, values in datasets.items():
        if anno_only and "anno" not in dataset:
            continue
        _df = read_file(dataset, values)
        
        if _df is None:
            continue
        
        if dataset in ["anno", "anno_multi"]:    
            _df = _df.dropna(subset=["gender", "bmi", "age"])
        
        if dataset in ["anno", "anno_multi", "anno_nan"]:
            _df = _df.loc[_df['to_exclude'].isna()]
            _df = _df.dropna(subset=["health_status"])
        
            _df = utility.parse_df(_df)
            _df = _df.loc[_df["host_phenotype"] != "adenoma"]
            
        if dataset in ["anno", "anno_raw", "anno_multi", "anno_nan"]:
            _df.set_index("sample", inplace=True, drop=True)
            
            if split_cohort_by_country:
                _df['cohort_name'] = _df['study_accession'] + ' - ' + _df['country']
                _df['cohort_name'] = _df['cohort_name'].map(clean_cohort_names).fillna(_df['cohort_name'])
            else:
                _df['cohort_name'] = _df['study_accession'].map(clean_cohort_names).fillna(_df['study_accession'])
        
        if "log" in dataset:
            _df = np.log1p(_df)
        
        if dataset in ['strain_meteor_imputed']:
            # To generate the strain table, we followed this logic:
            # 1.       We assume that there is no mixing of strains. So the abundance of the strain present is equal to the abundance of the species detected. The abundance of the other strains is 0.
            # 2.       If a species has been detected, but the strain analysis gives no result for a sample, then its abundance is NaN.
            # 3.       If a species was not detected for a sample, then its abundance 0 for all strains.
            #  --> Instead of setting NaNs, we could try to rescue the samples by using is msp_xxx_unknown species
            _df = _df.fillna(0.0)
            
        if dataset in ['strain_meteor_nan', 'strain_meteor_imputed', 'strain_taxa_meteor']:
            _df = get_counts(_df)
        
        if dataset in ["anno_multi"]:
            to_keep = []
            for cohort, c_df in _df.groupby("cohort_name"):
                if c_df.stage.dropna().unique().__len__() > 1 and c_df.health_status.unique().__len__() >1:
                    to_keep.append(cohort)
            _df = _df.loc[_df.cohort_name.isin(to_keep)]
            _df = _df.dropna(subset=["stage"])
            
        write_file(_df, dataset, values)
        if verbose:
            print(f"Processed {dataset}")


if __name__ == "__main__":
    prep_data()
# %%
