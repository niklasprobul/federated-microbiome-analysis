import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer
from sklearn.metrics import roc_auc_score
from seaborn import color_palette

import math
from collections import defaultdict
from pathlib import Path
from os.path import join as path_join
import os
import traceback
from dataclasses import dataclass


@dataclass
class Experiment:
    name: str
    anno: str|Path
    count: str|Path
    datasets: dict[str, str|Path] | None = None
    clean_dataset: str = ''
    clean_name: str = ''
    label: str = 'health_status'
    model: str | None =  None
    
    def __post_init__(self):
        if self.clean_name == '':
            self.clean_name = clean(self.name)
            
        self.clean_dataset = os.path.splitext(os.path.basename(self.count))[0]
            
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.name
    

def propagate_experiments(experiments, models):
    experiments_full = []
    
    for exp in experiments:
        if exp.model is None:
            for model in models:
                exp_copy = Experiment(name=f'{exp.name}_{model}', anno=exp.anno, count=exp.count, datasets=exp.datasets, clean_dataset=exp.clean_dataset, 
                                      clean_name=exp.clean_name, label=exp.label, model=model)
                experiments_full.append(exp_copy)
        else: 
            experiments_full.append(exp)
    
    return experiments_full

clean_names = {
    'Taxa + Function':'Taxa + Function ',
    'taxa_func':'Taxa + Function ',
    'taxa' : 'Taxa',
    'func' : 'Function',
    'brackets_taxa' : 'Taxa binned',
    'genus_taxa': 'IGC2 - genus level taxa',
    'count_taxa_metaphlan4_0': 'MetaPhlAn4',
    'taxa_reduced85': "Taxa + Function (corr < 0.85)",
    'taxa_reduced90': "Taxa + Function (corr < 0.90)",
    'taxa_reduced95': "Taxa + Function  (corr < 0.95)",
    'taxa_func_batch_bat_0': "Taxa + Function combat",
    'taxa_func_batch_pls_0': "Taxa + Function pls",
    'taxa_func_batch_svd_0': "Taxa + Function svd",
    'taxa_func_batch_bmc_0': "Taxa + Function bmc",
    'taxa_func_batch_rbe_0': "Taxa + Function rbe",
    'taxa_french_batch_rbe_0' : 'Taxa rbe',
    'taxa_french_batch_bat_0' : 'Taxa combat',
    'taxa_french_batch_pls_0' : 'Taxa pls',
    'taxa_french_batch_svd_0' : 'Taxa svd',
    'taxa_french_batch_bmc_0' : 'Taxa bmc',
    'count_taxa_0' : "Taxa raw 0%",
    'count_func_0' : "Function raw 0%",
    'count_taxa_func_0' : "Taxa + Function 0%",
    'count_taxa_10' : "Taxa raw 10%",
    'count_func_10' : "Function raw 10%",
    'count_taxa_func_10' : "Taxa + Function raw 10%",
    'taxa_french_batch_rbe_0' : 'Taxa rbe',
    'func_french_batch_rbe_0' : 'Function rbe',
    'taxa_func_french_batch_rbe_0' : 'Taxa + Function rbe',
    'taxa_french_batch_bat_0' : 'Taxa bat',
    'func_french_batch_bat_0' : 'Function bat',
    'taxa_func_french_batch_bat_0' : 'Taxa + Function bat',
    'taxa_french_batch_pls_0' : 'Taxa pls',
    'func_french_batch_pls_0' : 'Function pls',
    'taxa_func_french_batch_pls_0' : 'Taxa + Function pls',
    'taxa_french_batch_svd_10' : 'Taxa svd',
    'func_french_batch_svd_10' : 'Function svd',
    'taxa_func_french_batch_svd_0' : 'Taxa + Function svd',
    'taxa_french_batch_bmc_0' : 'Taxa bmc',
    'func_french_batch_bmc_0' : 'Function bmc',
    'taxa_func_french_batch_bmc_0' : 'Taxa + Function bmc',
    'taxa_french_batch_rbe_10' : 'Taxa rbe 10% filter',
    'func_french_batch_rbe_10' : 'Function rbe 10% filter',
    'taxa_func_french_batch_rbe_10' : 'Taxa + Function rbe 10% filter',
    'taxa_french_batch_bat_10' : 'Taxa bat 10% filter',
    'func_french_batch_bat_10' : 'Function bat 10% filter',
    'taxa_func_french_batch_bat_10' : 'Taxa + Function bat 10% filter',
    'taxa_french_batch_pls_10' : 'Taxa pls 10% filter',
    'func_french_batch_pls_10' : 'Function pls 10% filter',
    'taxa_func_french_batch_pls_10' : 'Taxa + Function pls 10% filter',
    'taxa_french_batch_svd_10' : 'Taxa svd 10% filter',
    'func_french_batch_svd_10' : 'Function svd 10% filter',
    'taxa_func_french_batch_svd_10' : 'Taxa + Function svd 10% filter',
    'taxa_french_batch_bmc_10' : 'Taxa bmc 10% filter',
    'func_french_batch_bmc_10' : 'Function bmc 10% filter',
    'taxa_func_french_batch_bmc_10' : 'Taxa + Function bmc 20% filter',
    'taxa_french_batch_rbe_20' : 'Taxa rbe 20% filter',
    'func_french_batch_rbe_20' : 'Function rbe 20% filter',
    'taxa_func_french_batch_rbe_20' : 'Taxa + Function rbe 20% filter',
    'taxa_french_batch_bat_20' : 'Taxa bat 20% filter',
    'func_french_batch_bat_20' : 'Function bat 20% filter',
    'taxa_func_french_batch_bat_20' : 'Taxa + Function bat 20% filter',
    'taxa_french_batch_pls_20' : 'Taxa pls 20% filter',
    'func_french_batch_pls_20' : 'Function pls 20% filter',
    'taxa_func_french_batch_pls_20' : 'Taxa + Function pls 20% filter',
    'taxa_french_batch_svd_20' : 'Taxa svd 20% filter',
    'func_french_batch_svd_20' : 'Function svd 20% filter',
    'taxa_func_french_batch_svd_20' : 'Taxa + Function svd 20% filter',
    'taxa_french_batch_bmc_20' : 'Taxa bmc 20% filter',
    'func_french_batch_bmc_20' : 'Function bmc 20% filter',
    'taxa_func_french_batch_bmc_20' : 'Taxa + Function bmc 20% filter',
    "test_mcc" : "MCC",
    "test_roc_auc" : "AUROC",
    "test_f1" : "F1",
    "test_pr_auc" : "AUPRC",
    "test_precision" : "Precision",
    "test_balanced_accuracy" : "Balanced Accuracy",
    "test_accuracy" : "Accuracy",
    "test_recall" : "Recall",
}

def clean(name:str, dict=clean_names):
    return dict.get(name, name)

def nested_dict():
    return defaultdict(dict)


def parse_df(df):
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({"female": 0, "male": 1})
    if "health_status" in df.columns:
        df["health_status"] = df["health_status"].replace({"H": 0, "P": 1})
    if 'study_accession' in df.columns:
        thomas_alias = {
            'PRJDB4176': 'V_Cohort2',
            'PRJEB10878': 'YuJ_2015',
            'PRJEB12449': 'VogtmannE_2016',
            'PRJEB27928': 'V_Cohort1',
            'PRJEB6070':  'ZellerG_2014',
            'PRJEB7774': 'FengQ_2015',
            'PRJNA389927': 'HanniganGD_2018',
        }
        
        df.loc[:, 'thomas_alias'] = df['study_accession'].map(thomas_alias)
        if "mgp_sample_alias" in df.columns:
            df.loc[df['mgp_sample_alias']=='Vercelli', 'thomas_alias'] = 'Cohort1'
            df.loc[df['mgp_sample_alias']=='Milan', 'thomas_alias'] = 'Cohort2'
    if 'host_phenotype' in df.columns and 'host_subphenotype' in df.columns:
        df['stage'] = np.nan
        df['stage'] = df.loc[df['host_subphenotype'].isin(['0', 'I', 'II', 'III', 'IV']), 'host_subphenotype'].replace({'0': 1, 'I': 2, 'II': 3, 'III': 4, 'IV': 5})
        df.loc[(df['host_phenotype'].isna()) & (df['host_subphenotype'].isna()), 'stage'] = 0
        
    return df

def evaluate(y_true, y_pred, metrics, y_scores=None, prefix=''):
    # Convert inputs to numpy arrays for fast vectorized operations.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    
    # count positives/negatives
    n_pos = int((y_true == 1).sum())
    n_neg = n - n_pos
    has_pos = n_pos > 0
    has_neg = n_neg > 0

    # Compute confusion matrix counts in one pass.
    # Assumes binary classification with labels 0 and 1.
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    # Precompute common rates for reuse.
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall for class 1)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Recall for class 0)
    
    results = {}
    
    # Accuracy: overall proportion correct.
    if 'accuracy' in metrics:
        results['accuracy'] = (tp + tn) / n if n > 0 else np.nan
        
    # Balanced accuracy: average of TPR and TNR.
    if 'balanced_accuracy' in metrics:
        results['balanced_accuracy'] = ((tpr + tnr) / 2) if has_pos and has_neg else np.nan

    # ROC AUC: for binary predictions, when using 0/1 outputs,
    # it is equivalent to the balanced accuracy.
    # if y_scores (probabilities) are available, use them.
    if 'roc_auc' in metrics:
        if not (has_pos and has_neg):
            results['roc_auc'] = np.nan
        else:
            if y_scores is not None:
                results['roc_auc'] = roc_auc_score(y_true, y_scores)
            else:
                results['roc_auc'] = (tpr + tnr) / 2  # fallback if no probabilities
        
    # F1 score: harmonic mean of precision and recall.
    # Precision = TP / (TP + FP) and Recall = TP / (TP + FN)
    if 'f1' in metrics:
        denom = 2 * tp + fp + fn
        results['f1'] = (2 * tp / denom) if denom > 0 else 0.0
        
    # Matthews Correlation Coefficient (MCC):
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if 'mcc' in metrics:
        numerator = tp * tn - fp * fn
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        results['mcc'] = numerator / denominator if denominator > 0 else 0.0
        
    if 'sensitivity' in metrics:
        results['sensitivity'] = tpr  # Same as recall
    if 'specificity' in metrics:
        results['specificity'] = tnr

    if 'precision' in metrics:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        results['precision'] = precision
    if 'recall' in metrics:
        results['recall'] = tpr
    
    if prefix:
        results = {f"{prefix}_{key}": value for key, value in results.items()}
        
    return results


def custom_cross_validate(estimator, X, y, features, cv:int|StratifiedKFold=5, metrics=None, feature_importance = 'gini'):
    #feature importance can be 'shap', 'permutation', 'both' or None

    # Initialize the cross-validation splitter if not provided
    if not isinstance(cv, StratifiedKFold):
        cv = StratifiedKFold(n_splits=cv)        
    
    # Initialize the scoring metrics
    metrics = ['accuracy'] if metrics is None else metrics
    
    # Perform cross-validation
    all_scores = []
    all_importances = []
    
    for cv_n, (train_index, test_index) in enumerate(cv.split(X, y)):
        scores = {}
        importances = defaultdict(nested_dict)

        datasets = {
            'test': (X[test_index], y[test_index]),
            'train': (X[train_index], y[train_index]),
        }
        
        # Fit the estimator on the training data
        estimator.fit(datasets['train'][0], datasets['train'][1])
        
        # Evaluate the estimator on the testing data
        
        # for dat in ('train', 'test'):
        for dat, (_X, _y) in datasets.items():
            # only eval test results for now
            if dat == 'train':
                continue

            if hasattr(estimator, 'predict_proba'):
                # Get probability estimates for the positive class.
                y_scores = estimator.predict_proba(_X)[:, 1]
                # Derive class predictions from probabilities.
                y_pred = (y_scores >= 0.5).astype(int)
            elif hasattr(estimator, 'decision_function'):
                # Models without predict_proba (e.g. hinge-loss SVM) expose a continuous decision
                # score; use it for score-based metrics (ROC AUC) instead of thresholded labels.
                y_scores = estimator.decision_function(_X)
                y_pred = estimator.predict(_X)
            else:
                y_pred = estimator.predict(_X)
                y_scores = None

            scores = evaluate(_y, y_pred, metrics=metrics, y_scores=y_scores, prefix=dat)

        if scores:
            scores['cv'] = cv_n
            
        # find a way to do this cleanly
        # if importances:
        #     importances['cv'] = cv_n

        all_importances.append(importances)
        all_scores.append(scores)
        
    return all_scores, all_importances
    
def custom_comb_validate(estimator, X_test, y_test, features, metrics=None, feature_importance = 'gini'):
    #feature importance can be 'shap', 'permutation', 'both' or None

    # Initialize the scoring metrics
    metrics = ['accuracy'] if metrics is None else metrics

    # Perform cross-validation
    scores = {}
    importances = defaultdict(nested_dict)    
        
    # Evaluate the estimator on the testing data
    datasets = {
        'test': (X_test, y_test),
    }
    
    voting = getattr(estimator, 'voting', None)

    for dat, (X, y) in datasets.items():

        y_scores, y_pred = None, None

        # PreFittedVotingClassifier always defines predict_proba, but under hard voting that returns
        # vote *proportions* -- only k+1 distinct values. ROC AUC on such a quantised score is not
        # comparable to a continuous one (at k=1 it is binary and AUROC collapses onto balanced
        # accuracy). Prefer the averaged margin, which stays continuous. AttributeError means at
        # least one member has no decision_function, so fall through.
        if voting == 'hard':
            try:
                y_scores = estimator.decision_function(X)
                y_pred = estimator.predict(X)
            except AttributeError:
                y_scores, y_pred = None, None

        if y_scores is None:
            if hasattr(estimator, 'predict_proba'):
                # Get probability estimates for the positive class.
                y_scores = estimator.predict_proba(X)[:, 1]
                # Derive class predictions from probabilities.
                y_pred = (y_scores >= 0.5).astype(int)
            elif hasattr(estimator, 'decision_function'):
                # Models without predict_proba (e.g. hinge-loss SVM) expose a continuous decision
                # score; use it for score-based metrics (ROC AUC) instead of thresholded labels.
                y_scores = estimator.decision_function(X)
                y_pred = estimator.predict(X)
            else:
                y_pred = estimator.predict(X)
                y_scores = None

        scores = evaluate(y, y_pred, metrics=metrics, y_scores=y_scores, prefix=dat)

    return scores, importances
