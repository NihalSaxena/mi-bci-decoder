import numpy as np
import pytest

from bci.data import SESSION_KEYS, epoch, load_epochs

SFREQ = 250.0
WINDOW = dict(cue_offset_s=2.0, tmin_s=0.5, tmax_s=2.5)


def _ramp(n_channels=3, n_samples=5000):
    # Each value is its own sample index (offset per channel), so a slice's
    # contents reveal exactly where it was cut from.
    return np.arange(n_samples)[None, :] + 100_000 * np.arange(n_channels)[:, None]


def test_epoch_shape_and_position():
    data, onsets = _ramp(), np.array([0, 1000, 3000])
    X = epoch(data, onsets, SFREQ, **WINDOW)

    assert X.shape == (3, 3, 500)                       # trials, channels, 2 s at 250 Hz
    # Window starts at onset + (2.0 + 0.5) s = onset + 625 samples.
    np.testing.assert_array_equal(X[:, 0, 0], onsets + 625)
    np.testing.assert_array_equal(X[:, 0, -1], onsets + 625 + 499)
    np.testing.assert_array_equal(X[1, 2], data[2, 1625:2125])  # channels kept intact


def test_epoch_rejects_window_past_end():
    with pytest.raises(ValueError):
        epoch(_ramp(n_samples=1000), np.array([600]), SFREQ, **WINDOW)


def test_session_keys_cover_both_moabb_conventions():
    assert {SESSION_KEYS[k] for k in ("0train", "session_T")} == {"T"}
    assert {SESSION_KEYS[k] for k in ("1test", "session_E")} == {"E"}


def test_load_epochs_reads_what_preprocess_writes(tmp_path):
    X, y = np.zeros((4, 22, 500), dtype=np.float32), np.array([0, 1, 2, 3])
    np.savez(tmp_path / "A03E.npz", X=X, y=y, ch_names=np.array(["Fz"] * 22))
    X_loaded, y_loaded = load_epochs(tmp_path, 3, "E")
    np.testing.assert_array_equal(X_loaded, X)
    np.testing.assert_array_equal(y_loaded, y)
