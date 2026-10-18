"""Causal bandpass filtering.

Everything here must be reproducible sample-by-sample in a live stream, so we
only ever filter forward in time (sosfilt, never filtfilt). See docs/DESIGN.md,
"Causal filtering only".
"""
import numpy as np
from scipy import signal


def design_bandpass(l_freq: float, h_freq: float, sfreq: float, order: int = 4) -> np.ndarray:
    """Butterworth bandpass as second-order sections, shape (n_sections, 6).

    `order` is the prototype order; a bandpass doubles it, so order=4 gives an
    8th-order filter split into 4 sections.
    """
    return signal.butter(order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")


def initial_state(sos: np.ndarray, first_sample: np.ndarray) -> np.ndarray:
    """Filter state as if each channel had been at its first value forever.

    sosfilt_zi gives the steady state for a unit step; scaling it per channel
    means a DC offset produces no startup transient. Returns shape
    (n_sections, n_channels, 2), the layout sosfilt expects for 2-D input.
    """
    zi = signal.sosfilt_zi(sos)  # (n_sections, 2)
    return zi[:, None, :] * np.asarray(first_sample)[None, :, None]


def causal_filter(
    x: np.ndarray, sos: np.ndarray, zi: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Filter x (n_channels, n_samples) along time; return (y, final_state).

    With zi=None the state is initialised from x's first sample. To stream,
    pass the returned state back in with the next chunk.
    """
    if zi is None:
        zi = initial_state(sos, x[:, 0])
    return signal.sosfilt(sos, x, axis=-1, zi=zi)
