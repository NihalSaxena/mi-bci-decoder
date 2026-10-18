import json

import numpy as np
from scipy import signal

from bci.filters import causal_filter, design_bandpass, filter_spec

SFREQ = 250.0


def _sos():
    return design_bandpass(8.0, 30.0, SFREQ, order=4)


def _eeg_like(n_channels=22, n_samples=5000, seed=0):
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-50, 50, size=(n_channels, 1))  # per-channel DC, µV
    return offsets + rng.standard_normal((n_channels, n_samples)) * 10


def test_streaming_matches_whole_signal():
    sos, x = _sos(), _eeg_like()
    y_whole, _ = causal_filter(x, sos)

    # Irregular chunk sizes, including single samples, to catch boundary bugs.
    chunks, zi, start = [], None, 0
    for size in [1, 7, 250, 3, 1000, 64] * 20:
        if start >= x.shape[1]:
            break
        if zi is None:
            # The stream initialises from its first sample, same as the batch path.
            y, zi = causal_filter(x[:, start : start + size], sos)
        else:
            y, zi = causal_filter(x[:, start : start + size], sos, zi)
        chunks.append(y)
        start += size

    np.testing.assert_allclose(np.concatenate(chunks, axis=1), y_whole, rtol=0, atol=1e-10)


def test_filter_is_causal():
    sos, x = _sos(), _eeg_like()
    t = 2000
    x_kicked = x.copy()
    x_kicked[:, t] += 1000.0

    y, _ = causal_filter(x, sos)
    y_kicked, _ = causal_filter(x_kicked, sos)

    np.testing.assert_array_equal(y_kicked[:, :t], y[:, :t])  # past untouched
    assert not np.allclose(y_kicked[:, t:], y[:, t:])          # future affected


def test_dc_offset_has_no_startup_transient():
    sos = _sos()
    x = np.full((3, 1000), [[40.0], [-25.0], [5.0]])
    y, _ = causal_filter(x, sos)
    np.testing.assert_allclose(y, 0.0, atol=1e-9)


def test_passband_and_stopband():
    freqs, h = signal.sosfreqz(_sos(), worN=[2.0, 15.0, 50.0], fs=SFREQ)
    gain = np.abs(h)
    assert gain[1] > 0.95   # 15 Hz: mu/beta passes
    assert gain[0] < 0.01   # 2 Hz: drift removed
    assert gain[2] < 0.05   # 50 Hz: line noise removed


def test_spec_reproduces_filter_after_json_round_trip():
    # The engine only sees the JSON, so filtering with the coefficients and
    # initial-state rule read back from it must match causal_filter exactly.
    spec = json.loads(json.dumps(filter_spec(
        "csp", 8.0, 30.0, 4, SFREQ, [f"ch{i}" for i in range(22)],
        cue_offset_s=2.0, tmin_s=0.5, tmax_s=2.5,
    )))
    x = _eeg_like()
    sos = np.array(spec["sos"])
    zi = np.array(spec["zi_unit_step"])[:, None, :] * x[:, 0][None, :, None]
    y_spec, _ = signal.sosfilt(sos, x, axis=-1, zi=zi)
    y_ref, _ = causal_filter(x, _sos())

    np.testing.assert_array_equal(y_spec, y_ref)
    assert len(spec["channels"]) == 22
    assert spec["window"]["n_samples"] == 500
