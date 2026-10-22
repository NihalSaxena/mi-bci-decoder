"""Scoring a decoder: accuracy, Cohen's kappa, confusion matrix, Wolpaw ITR.

See docs/DESIGN.md, "Metrics".
"""
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of trials predicted correctly."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) counts; row = true class, column = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Cohen's kappa: accuracy rescaled so chance = 0 and perfect = 1.

        kappa = (p_o - p_e) / (1 - p_e)

    p_o is the observed accuracy. p_e is the accuracy you'd expect by luck,
    given how often each class appears in y_true and in y_pred:

        p_e = sum over classes k of  (fraction of y_true that is k) * (fraction of y_pred that is k)

    With 4 equally common classes and a predictor that also spreads its guesses
    evenly, p_e = 4 * (0.25 * 0.25) = 0.25.
    """
    p_o = np.mean(y_true == y_pred)

    true_counts = np.bincount(y_true, minlength=n_classes)
    true_frac = true_counts / len(y_true)

    pred_counts = np.bincount(y_pred, minlength=n_classes)
    pred_frac = pred_counts / len(y_pred)

    p_e = np.sum(true_frac * pred_frac)
    return float((p_o - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def itr(accuracy: float, n_classes: int, trial_duration_s: float) -> float:
    """Wolpaw information transfer rate, in bits per minute.

    Bits per trial, with N = n_classes and P = accuracy:

        B = log2(N) + P*log2(P) + (1 - P)*log2((1 - P) / (N - 1))

    then bits per minute = B * 60 / trial_duration_s.

    Edge cases:
      - P = 1: the (1 - P) term is 0 * log2(0); treat it as 0, so B = log2(N).
      - P <= 1/N (chance or worse): return 0.0.
    """
    N, P = n_classes, accuracy

    if P <= 1/N:
        return 0.0
    
    if P == 1:
        return float(np.log2(N) * 60 / trial_duration_s)

    B = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))

    return float(B * 60 / trial_duration_s)


def score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, trial_duration_s: float) -> dict:
    """All metrics for one evaluation, JSON-serialisable."""
    acc = accuracy(y_true, y_pred)
    return {
        "accuracy": acc,
        "kappa": kappa(y_true, y_pred, n_classes),
        "itr_bits_per_min": itr(acc, n_classes, trial_duration_s),
        "confusion": confusion_matrix(y_true, y_pred, n_classes).tolist(),
    }
