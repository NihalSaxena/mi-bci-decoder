import numpy as np
import pytest
import torch

from bci.models.eegnet import EEGNet
from bci.train import cv_epoch_curve, fit, fit_fixed, predict, set_seed, split_train_val

N_CHANNELS, N_TIMES, SFREQ = 4, 128, 128.0


def _toy_eeg(n_per_class, seed):
    # Two classes: a 12 Hz rhythm that is strong on channel 0 for class 0 and on
    # channel 1 for class 1, buried in noise. Small so training takes seconds.
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIMES) / SFREQ
    X, y = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            x = rng.standard_normal((N_CHANNELS, N_TIMES))
            x[label] += 2.0 * np.sin(2 * np.pi * 12 * t + rng.uniform(0, 2 * np.pi))
            X.append(x)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def _small_model():
    return EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=2)


def _train(seed=0, max_epochs=30, patience=10, **kwargs):
    X, y = _toy_eeg(40, seed=1)
    X_tr, X_val, y_tr, y_val = split_train_val(X, y, val_fraction=0.25, seed=seed)
    set_seed(seed)
    model = _small_model()
    history = fit(
        model, X_tr, y_tr, X_val, y_val, max_epochs=max_epochs, patience=patience, seed=seed, **kwargs
    )
    return model, history, (X_val, y_val)


@pytest.fixture(scope="module")
def trained():
    # One training run shared by the tests below, to keep the suite fast.
    return _train()


def test_split_is_class_balanced_and_disjoint():
    X, y = _toy_eeg(40, seed=0)
    X_tr, X_val, y_tr, y_val = split_train_val(X, y, val_fraction=0.25, seed=0)
    assert len(X_val) == 20 and len(X_tr) == 60
    assert np.bincount(y_val).tolist() == [10, 10]
    # No trial appears in both sets.
    assert not {x.tobytes() for x in X_tr} & {x.tobytes() for x in X_val}


def test_fit_learns_a_separable_toy_problem(trained):
    model, history, _ = trained
    X_test, y_test = _toy_eeg(30, seed=2)
    assert np.mean(predict(model, X_test) == y_test) > 0.9
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_best_validation_weights_are_restored(trained):
    model, history, (X_val, y_val) = trained
    best = min(h["val_loss"] for h in history)
    model.eval()
    with torch.no_grad():
        val_loss = torch.nn.functional.cross_entropy(
            model(torch.as_tensor(X_val)), torch.as_tensor(y_val)
        ).item()
    assert abs(val_loss - best) < 1e-5


def test_early_stopping_stops_after_patience():
    # A high learning rate overshoots early, so validation loss stops improving.
    _, history, _ = _train(lr=5e-2, max_epochs=60, patience=5)
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    assert len(history) == best_epoch + 5 + 1 < 60


def test_same_seed_same_result():
    _, h1, _ = _train(seed=3, max_epochs=3)
    _, h2, _ = _train(seed=3, max_epochs=3)
    assert [h["val_loss"] for h in h1] == [h["val_loss"] for h in h2]


def test_fit_fixed_trains_exactly_n_epochs_and_learns():
    X, y = _toy_eeg(40, seed=1)
    set_seed(0)
    model = _small_model()
    losses = fit_fixed(model, X, y, n_epochs=25)
    assert len(losses) == 25 and losses[-1] < losses[0]
    X_test, y_test = _toy_eeg(30, seed=2)
    assert np.mean(predict(model, X_test) == y_test) > 0.9


def test_cv_epoch_curve_averages_folds():
    X, y = _toy_eeg(20, seed=1)
    curve = cv_epoch_curve(_small_model, X, y, n_folds=4, max_epochs=15)
    assert curve.shape == (15,)
    assert np.all((0 <= curve) & (curve <= 1))
    assert curve[-5:].mean() > curve[0]    # validation accuracy improves with training
