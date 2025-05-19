import numpy as np
import sklearn.metrics as skm
import torch

from metal.utils import arraylike_to_numpy, pred_to_prob
from collections import Counter


def accuracy_score(gold, pred, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate (micro) accuracy.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.

    Returns:
        A float, the (micro) accuracy score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    if len(gold) and len(pred):
        acc = np.sum(gold == pred) / len(gold)
    else:
        acc = 0

    return acc


def coverage_score(gold, pred, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate (global) coverage.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.

    Returns:
        A float, the (global) coverage score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    return np.sum(pred != 0) / len(pred)


def precision_score(gold, pred, pos_label=1, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate precision for a single class.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.
        pos_label: The class label to treat as positive for precision

    Returns:
        pre: The (float) precision score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    positives = np.where(pred == pos_label, 1, 0).astype(bool)
    trues = np.where(gold == pos_label, 1, 0).astype(bool)
    TP = np.sum(positives * trues)
    FP = np.sum(positives * np.logical_not(trues))

    if TP or FP:
        pre = TP / (TP + FP)
    else:
        pre = 0

    return pre


def recall_score(gold, pred, pos_label=1, ignore_in_gold=[], ignore_in_pred=[]):
    """
    Calculate recall for a single class.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.
        pos_label: The class label to treat as positive for recall

    Returns:
        rec: The (float) recall score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)

    positives = np.where(pred == pos_label, 1, 0).astype(bool)
    trues = np.where(gold == pos_label, 1, 0).astype(bool)
    TP = np.sum(positives * trues)
    FN = np.sum(np.logical_not(positives) * trues)

    if TP or FN:
        rec = TP / (TP + FN)
    else:
        rec = 0

    return rec


def fbeta_score(
    gold, pred, pos_label=1, beta=1.0, ignore_in_gold=[], ignore_in_pred=[]
):
    """
    Calculate recall for a single class.
    Args:
        gold: A 1d array-like of gold labels
        pred: A 1d array-like of predicted labels (assuming abstain = 0)
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.
        ignore_in_pred: A list of labels for which elements having that pred
            label will be ignored.
        pos_label: The class label to treat as positive for f-beta
        beta: The beta to use in the f-beta score calculation

    Returns:
        fbeta: The (float) f-beta score
    """
    gold, pred = _preprocess(gold, pred, ignore_in_gold, ignore_in_pred)
    pre = precision_score(gold, pred, pos_label=pos_label)
    rec = recall_score(gold, pred, pos_label=pos_label)

    if pre or rec:
        fbeta = (1 + beta**2) * (pre * rec) / ((beta**2 * pre) + rec)
    else:
        fbeta = 0

    return fbeta


def f1_score(gold, pred, **kwargs):
    return fbeta_score(gold, pred, beta=1.0, **kwargs)


def roc_auc_score(gold, probs, ignore_in_gold=[], ignore_in_pred=[]):
    """Compute the ROC AUC score, given the gold labels and predicted probs.

    Args:
        gold: A 1d array-like of gold labels
        probs: A 2d array-like of predicted probabilities
        ignore_in_gold: A list of labels for which elements having that gold
            label will be ignored.

    Returns:
        roc_auc_score: The (float) roc_auc score
    """
    gold = arraylike_to_numpy(gold)

    # Filter out the ignore_in_gold (but not ignore_in_pred)
    # Note the current sub-functions (below) do not handle this...
    if len(ignore_in_pred) > 0:
        raise ValueError("ignore_in_pred not defined for ROC-AUC score.")
    keep = [x not in ignore_in_gold for x in gold]
    gold = gold[keep]
    probs = probs[keep, :]

    # Convert gold to one-hot indicator format, using the k inferred from probs
    gold_s = pred_to_prob(torch.from_numpy(gold), k=probs.shape[1]).numpy()
    return skm.roc_auc_score(gold_s, probs)


def _drop_ignored(gold, pred, ignore_in_gold, ignore_in_pred):
    """Remove from gold and pred all items with labels designated to ignore."""
    keepers = np.ones_like(gold).astype(bool)
    for x in ignore_in_gold:
        keepers *= np.where(gold != x, 1, 0).astype(bool)
    for x in ignore_in_pred:
        keepers *= np.where(pred != x, 1, 0).astype(bool)

    gold = gold[keepers]
    pred = pred[keepers]
    return gold, pred


def _preprocess(gold, pred, ignore_in_gold, ignore_in_pred):
    gold = arraylike_to_numpy(gold)
    pred = arraylike_to_numpy(pred)
    if ignore_in_gold or ignore_in_pred:
        gold, pred = _drop_ignored(gold, pred, ignore_in_gold, ignore_in_pred)
    return gold, pred


METRICS = {
    "accuracy": accuracy_score,
    "coverage": coverage_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "fbeta": fbeta_score,
    "roc-auc": roc_auc_score,
}


def metric_score(gold, pred, metric, probs=None, **kwargs):
    if metric not in METRICS:
        msg = f"The metric you provided ({metric}) is not supported."
        raise ValueError(msg)

    # Note special handling because requires the predicted probabilities
    elif metric == "roc-auc":
        if probs is None:
            raise ValueError("ROC-AUC score requries the predicted probs.")
        return roc_auc_score(gold, probs, **kwargs)

    else:
        return METRICS[metric](gold, pred, **kwargs)


class ConfusionMatrix(object):
    """
    An iteratively built abstention-aware confusion matrix with pretty printing

    Assumed axes are true label on top, predictions on the side.
    """

    def __init__(self, null_pred=False, null_gold=False):
        """
        Args:
            null_pred: If True, include the row corresponding to null
                predictions
            null_gold: If True, include the col corresponding to null gold
                labels

        """
        self.counter = Counter()
        self.mat = None
        self.null_pred = null_pred
        self.null_gold = null_gold

    def __repr__(self):
        if self.mat is None:
            self.compile()
        return str(self.mat)

    def add(self, gold, pred):
        """
        Args:
            gold: a np.ndarray of gold labels (ints)
            pred: a np.ndarray of predictions (ints)
        """
        self.counter.update(zip(gold, pred))

    def compile(self, trim=True):
        k = max([max(tup) for tup in self.counter.keys()]) + 1  # include 0

        mat = np.zeros((k, k), dtype=int)
        for (y, l), v in self.counter.items():
            mat[l, y] = v

        if trim and not self.null_pred:
            mat = mat[1:, :]
        if trim and not self.null_gold:
            mat = mat[:, 1:]

        self.mat = mat
        return mat

    def display(self, normalize=False, indent=0, spacing=2, decimals=3, mark_diag=True):
        mat = self.compile(trim=False)
        m, n = mat.shape
        tab = " " * spacing
        margin = " " * indent

        # Print headers
        s = margin + " " * (5 + spacing)
        for j in range(n):
            if j == 0 and not self.null_gold:
                continue
            s += f" y={j} " + tab
        print(s)

        # Print data
        for i in range(m):
            # Skip null predictions row if necessary
            if i == 0 and not self.null_pred:
                continue
            s = margin + f" l={i} " + tab
            for j in range(n):
                # Skip null gold if necessary
                if j == 0 and not self.null_gold:
                    continue
                else:
                    if i == j and mark_diag and normalize:
                        s = s[:-1] + "*"
                    if normalize:
                        s += f"{mat[i, j] / sum(mat[i, 1:]):>5.3f}" + tab
                    else:
                        s += f"{mat[i, j]:^5d}" + tab
            print(s)


def confusion_matrix(
    gold, pred, null_pred=False, null_gold=False, normalize=False, pretty_print=True
):
    """A shortcut method for building a confusion matrix all at once.

    Args:
        gold: an array-like of gold labels (ints)
        pred: an array-like of predictions (ints)
        null_pred: If True, include the row corresponding to null predictions
        null_gold: If True, include the col corresponding to null gold labels
        normalize: if True, divide counts by the total number of items
        pretty_print: if True, pretty-print the matrix before returning
    """
    conf = ConfusionMatrix(null_pred=null_pred, null_gold=null_gold)
    gold = arraylike_to_numpy(gold)
    pred = arraylike_to_numpy(pred)
    conf.add(gold, pred)
    mat = conf.compile()

    if normalize:
        mat = mat / len(gold)

    if pretty_print:
        conf.display(normalize=normalize)

    return mat


def score_model(
    model,
    L,
    Y,
    metric="accuracy",
    break_ties="random",
    verbose=True,
    print_confusion_matrix=True,
    **kwargs,
):
    """Scores the predictive performance of the Classifier on all tasks

    Args:
        data: a Pytorch DataLoader, Dataset, or tuple with Tensors (X,Y):
            X: The input for the predict method
            Y: An [n] or [n, 1] torch.Tensor or np.ndarray of target labels
                in {1,...,k}
        metric: A metric (string) with which to score performance or a
            list of such metrics
        break_ties: A tie-breaking policy (see Classifier._break_ties())
        verbose: The verbosity for just this score method; it will not
            update the class config.
        print_confusion_matrix: Print confusion matrix (overwritten to False if
            verbose=False)

    Returns:
        scores: A (float) score or a list of such scores if kwarg metric
            is a list
    """
    Y_p = model.predict(L, break_ties=break_ties, return_probs=False)

    # pdb.set_trace()
    # Evaluate on the specified metrics
    return_list = isinstance(metric, list)
    metric_list = metric if isinstance(metric, list) else [metric]
    scores = []
    for metric in metric_list:
        score = metric_score(Y, Y_p, metric, probs=None, ignore_in_gold=[0], **kwargs)
        scores.append(score)
        if verbose:
            print(f"{metric.capitalize()}: {score:.3f}")

    # Optionally print confusion matrix
    if print_confusion_matrix and verbose:
        confusion_matrix(Y, Y_p, pretty_print=True)

    # If a single metric was given as a string (not list), return a float
    if len(scores) == 1 and not return_list:
        return scores[0]
    else:
        return scores
