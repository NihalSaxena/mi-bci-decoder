import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from bci.metrics import accuracy, confusion_matrix, itr, kappa, score

# 16 trials, 4 of each class.
Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])


def test_accuracy_and_confusion_matrix():
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3])
    assert accuracy(Y_TRUE, y_pred) == 13 / 16

    cm = confusion_matrix(Y_TRUE, y_pred, 4)
    assert cm.sum() == 16
    assert np.trace(cm) == 13          # correct guesses sit on the diagonal
    assert cm[0, 1] == 1               # one true-0 trial guessed as 1
    assert cm[2, 3] == 2               # two true-2 trials guessed as 3


def test_kappa_perfect_is_one():
    assert kappa(Y_TRUE, Y_TRUE, 4) == pytest.approx(1.0)


def test_kappa_chance_is_zero():
    # Each class guessed equally often, and exactly 1 in 4 right: pure chance.
    y_pred = np.tile([0, 1, 2, 3], 4)
    assert accuracy(Y_TRUE, y_pred) == 0.25
    assert kappa(Y_TRUE, y_pred, 4) == pytest.approx(0.0)


def test_kappa_matches_sklearn_on_unbalanced_classes():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 200)
    y_pred = np.where(rng.random(200) < 0.6, y_true, rng.integers(0, 4, 200))
    assert kappa(y_true, y_pred, 4) == pytest.approx(cohen_kappa_score(y_true, y_pred))


def test_itr_perfect_is_log2_n_per_trial():
    # 4 classes, always right: 2 bits per trial; 8 s trials -> 7.5 trials/min.
    assert itr(1.0, 4, 8.0) == pytest.approx(2.0 * 7.5)


def test_itr_at_or_below_chance_is_zero():
    assert itr(0.25, 4, 8.0) == 0.0
    assert itr(0.10, 4, 8.0) == 0.0


def test_itr_known_value():
    # Worked by hand: P = 0.7, N = 4 -> 0.6432 bits/trial.
    assert itr(0.7, 4, 8.0) == pytest.approx(0.6432 * 60 / 8, abs=1e-3)


def test_itr_uses_trial_duration():
    # The 2 s decode window would inflate ITR 4x versus the 8 s trial.
    assert itr(0.7, 4, 2.0) == pytest.approx(4 * itr(0.7, 4, 8.0))


def test_score_bundles_the_metrics():
    y_pred = np.tile([0, 1, 2, 3], 4)
    s = score(Y_TRUE, y_pred, 4, 8.0)
    assert s["accuracy"] == 0.25 and s["itr_bits_per_min"] == 0.0
    assert s["kappa"] == pytest.approx(0.0)
    assert np.array(s["confusion"]).sum() == 16
