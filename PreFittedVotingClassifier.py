import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import accuracy_score


def _check_array_allow_nan(X):
    """check_array allowing NaN, robust to the force_all_finite -> ensure_all_finite
    rename in scikit-learn 1.6 (force_all_finite removed by 1.8)."""
    try:
        return check_array(X, ensure_all_finite='allow-nan')
    except TypeError:  # scikit-learn < 1.6
        return check_array(X, force_all_finite='allow-nan')


class FederatedBoostedModel(BaseEstimator, ClassifierMixin):
    """Global model of PARALLEL federated gradient boosting (a.k.a. bagging / scaled-shrinkage boosting).

    F(x) = sum_i w_i * booster_i.raw_score(x), i.e. a weighted sum of the per-round, per-client boosting
    increments. Every client starts each round from the SAME global margin F, so -- unlike cyclic
    boosting -- there is no sequential drift and no dependence on the ring order; the last cohort visited
    cannot dominate. The server aggregates the k increments as a weighted AVERAGE (weights sum to 1 per
    round), which is why one round adds ~trees_per_round of effective capacity rather than k times that.

    Sample-size weighting here is NVFlare's documented "scaled shrinkage", the remedy for the quantity
    skew that makes cyclic boosting collapse (this cohort set spans 44..546 samples, a 12.4x range).

    Only fitted trees ever cross the wire: a client computes its init_score locally from the global model
    it already holds. No gradients, no per-node histograms.
    """

    def __init__(self, boosters=None, classes=None):
        self.boosters = boosters or []      # list of (weight, lightgbm.Booster)
        self.classes = classes

    def fit(self, X=None, y=None):
        self.classes_ = np.asarray(self.classes if self.classes is not None else [0, 1])
        self.boosters_ = list(self.boosters)
        return self

    def decision_function(self, X):
        X = _check_array_allow_nan(X)
        if not self.boosters_:
            return np.zeros(X.shape[0])
        return sum(w * b.predict(X, raw_score=True) for w, b in self.boosters_)

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def _more_tags(self):
        return {'requires_fit': False}


class PreFittedVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    A VotingClassifier-like estimator that accepts pre-fitted base classifiers,
    computing the union of all classes across estimators.

    Parameters
    ----------
    estimators : list of (str, estimator)
        List of pre-fitted classifiers.
    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of the weighted sum
        of predicted probabilities.
    weights : array-like, shape (n_estimators,), default=None
        Sequence of weights (float or int) to weight the contribution of each classifier.
        Weights need not be normalised; they are divided by their sum.
    pred_cache : dict, optional
        Memo for raw member outputs, shared across combinations. A member's prediction on a given
        cohort does not depend on which *other* members it is voted with, nor on the weights -- yet
        the combination x weighting loops recompute it every time. Over one
        federated sweep that is ~20,000 inferences per weighting where only ~150 distinct
        (member, cohort) pairs exist. Supplying a cache makes the sweep a lookup instead.

        Purely a memo: cached values are the *same arrays* the member would have returned, so results
        are arithmetically identical. Requires `member_keys` (which member) and `batch_key` (which
        evaluation cohort) to identify a row; with either absent it degrades silently to recomputing.
    member_keys : list, optional
        Per-estimator identity, parallel to `estimators`. Must distinguish any two *differently fitted*
        models -- use the same key that identifies the fitted estimator itself.
    batch_key : hashable, optional
        Identity of the evaluation batch X (in practice the held-out cohort's name). Set it before each
        evaluation; it is a plain attribute, so `clf.batch_key = cohort` between calls is fine.
    """
    def __init__(self, estimators, voting='hard', weights=None,
                 pred_cache=None, member_keys=None, batch_key=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.pred_cache = pred_cache
        self.member_keys = member_keys
        self.batch_key = batch_key

    def _cached(self, e_idx, kind, compute):
        """Raw member output, from the memo when one is available.

        `compute` is called at most once per (member, batch, kind) and its result stored verbatim.
        The returned array is NOT copied: every caller downstream either reads it or writes into a
        fresh array, so a copy per lookup would cost more than it protects.
        """
        cache = self.pred_cache
        keys = self.member_keys
        if cache is None or keys is None or self.batch_key is None:
            return compute()
        key = (keys[e_idx], self.batch_key, kind)
        out = cache.get(key)
        if out is None:
            out = compute()
            cache[key] = out
        return out

    def _validated(self, X):
        """check_array is an O(n_samples x n_features) scan; with a memo it runs once per cohort."""
        if self.pred_cache is None or self.batch_key is None:
            return _check_array_allow_nan(X)
        key = ('__X__', self.batch_key)
        out = self.pred_cache.get(key)
        if out is None:
            out = _check_array_allow_nan(X)
            self.pred_cache[key] = out
        return out

    def fit(self, X=None, y=None):
        """
        Validates that the provided estimators are pre-fitted and computes the union of
        all classes from each estimator.
        
        Parameters
        ----------
        X : Ignored, for API compatibility.
        y : Ignored, for API compatibility.
        
        Returns
        -------
        self
        """
        # Use the provided estimators (assumed to be pre-fitted)
        self.estimators_ = [(name, estimator) for name, estimator in self.estimators]
        
        # Compute the union of classes across all estimators
        classes_set = set()
        for name, clf in self.estimators_:
            if not hasattr(clf, 'classes_'):
                raise ValueError(
                    f"The estimator '{name}' is not fitted: missing 'classes_' attribute."
                )
            classes_set.update(clf.classes_)
        self.classes_ = np.array(sorted(classes_set))
        
        # Build a mapping from class label to index in the union
        self.class_mapping_ = {label: idx for idx, label in enumerate(self.classes_)}
        
        # For soft voting, ensure each estimator supports predict_proba.
        if self.voting == 'soft':
            for name, clf in self.estimators_:
                if not hasattr(clf, "predict_proba"):
                    raise ValueError(
                        f"The estimator '{name}' does not support predict_proba, which is required for {self.voting} voting."
                    )

        return self

    def _member_predictions(self, X):
        """Hard-vote member labels, shape (n_samples, n_estimators)."""
        return np.asarray([
            self._cached(e_idx, 'predict', lambda clf=clf: clf.predict(X))
            for e_idx, (_, clf) in enumerate(self.estimators_)
        ]).T

    def _mapped_probas(self, X):
        """Per-member probabilities remapped onto the union of classes.

        Returns an array of shape (n_estimators, n_samples, n_classes).
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        out = np.zeros((len(self.estimators_), n_samples, n_classes))

        for e_idx, (_, clf) in enumerate(self.estimators_):
            clf_proba = self._cached(e_idx, 'proba', lambda clf=clf: clf.predict_proba(X))
            mapped = np.zeros((n_samples, n_classes))
            for idx, cls in enumerate(clf.classes_):
                mapped[:, self.class_mapping_[cls]] = clf_proba[:, idx]

            out[e_idx] = mapped
        return out

    def predict(self, X):
        """
        Predict class labels for samples in X based on the voting rule.
        """
        if self.voting == 'soft':
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        else:  # hard voting

            X = self._validated(X)
            # Collect predictions from each estimator.
            predictions = self._member_predictions(X)  # shape: (n_samples, n_estimators)

            if self.weights is None:
                # Unweighted majority vote
                maj_vote = np.apply_along_axis(
                    lambda x: np.bincount([self.class_mapping_[cls] for cls in x],
                                          minlength=len(self.classes_)).argmax(),
                    axis=1,
                    arr=predictions
                )
            else:
                # Weighted vote: accumulate weighted votes for each class.
                weights = np.array(self.weights)
                maj_vote = []
                for preds in predictions:
                    vote_count = np.zeros(len(self.classes_))
                    for weight, pred in zip(weights, preds):
                        vote_count[self.class_mapping_[pred]] += weight
                    maj_vote.append(np.argmax(vote_count))
                maj_vote = np.array(maj_vote)
            return self.classes_[maj_vote]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        For 'soft' voting: returns weighted average of estimator probabilities.
        For 'hard' voting: returns the proportion of votes each class received.
        
        Returns
        -------
        probas : array-like, shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        X = self._validated(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        if self.voting == 'soft':
            mapped = self._mapped_probas(X)  # (n_estimators, n_samples, n_classes)

            weights = self.weights if self.weights is not None else np.ones(len(self.estimators_))
            weights = np.asarray(weights, dtype=float)
            weights_sum = np.sum(weights)
            if weights_sum <= 0:
                # All-zero weights (e.g. an excess-over-chance vector where every client is at or
                # below chance) would make this 0/0 -> all-NaN -> roc_auc_score raises. Fall back
                # to uniform rather than crashing the combination.
                weights = np.ones(len(self.estimators_))
                weights_sum = float(len(self.estimators_))

            total_proba = np.tensordot(weights, mapped, axes=(0, 0))
            return total_proba / weights_sum
        else:
            # For hard voting, calculate vote proportions as probabilities
            predictions = self._member_predictions(X)

            # Initialize array to hold vote counts for each class
            vote_counts = np.zeros((n_samples, n_classes))
            
            # Convert predictions to class indices and count them
            if self.weights is None:
                # Unweighted voting
                for i, sample_preds in enumerate(predictions):
                    for pred in sample_preds:
                        vote_counts[i, self.class_mapping_[pred]] += 1
                # Normalize to get probabilities
                return vote_counts / len(self.estimators_)
            else:
                # Weighted voting
                weights = np.array(self.weights)
                for i, sample_preds in enumerate(predictions):
                    for weight, pred in zip(weights, sample_preds):
                        vote_counts[i, self.class_mapping_[pred]] += weight
                # Normalize to get probabilities
                return vote_counts / np.sum(weights)

    def decision_function(self, X):
        """Weighted mean of the members' decision functions (binary).

        Without this, margin-based members (hinge-loss SVM) have no continuous ensemble score:
        they fall back to `voting='hard'`, whose predict_proba returns *vote proportions*, so at
        k clients the score takes only k+1 distinct values (at k=1 it is binary and AUROC collapses
        onto balanced accuracy). Averaging the margins keeps the score continuous.

        Raises AttributeError if any member lacks decision_function, so hasattr() probes stay honest.
        """
        for name, clf in self.estimators_:
            if not hasattr(clf, 'decision_function'):
                raise AttributeError(
                    f"The estimator '{name}' has no decision_function."
                )

        X = self._validated(X)
        weights = self.weights if self.weights is not None else np.ones(len(self.estimators_))
        weights = np.asarray(weights, dtype=float)
        if np.sum(weights) <= 0:
            weights = np.ones(len(self.estimators_))

        scores = []
        for e_idx, (_, clf) in enumerate(self.estimators_):
            scores.append(self._cached(
                e_idx, 'decision',
                lambda clf=clf: np.asarray(clf.decision_function(X), dtype=float).ravel(),
            ))

        return np.average(np.vstack(scores), axis=0, weights=weights)

    def score(self, X, y, sample_weight=None):
        """
        Returns the accuracy score on the given test data and labels.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def transform(self, X):
        """
        Returns a matrix with the predictions of each classifier.
        """
        X = check_array(X)
        return np.asarray([clf.predict(X) for _, clf in self.estimators_]).T

    def get_params(self, deep=True):
        """
        Get parameters for this estimator, necessary for grid search and cross-validation.
        """
        return super().get_params(deep=deep)

    def _more_tags(self):
        # Helps scikit-learn to recognize this as a classifier that does not need refitting.
        return {'requires_fit': False}
