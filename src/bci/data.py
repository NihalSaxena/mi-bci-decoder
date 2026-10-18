"""Load BCI Competition IV 2a via MOABB and cut continuous runs into trials.

Loading keeps each run continuous so it can be filtered before epoching. See
docs/DESIGN.md, "Epoch window" and "Evaluation protocol".
"""
from dataclasses import dataclass

import numpy as np

CLASSES = ("left_hand", "right_hand", "feet", "tongue")  # label i <-> CLASSES[i]

# MOABB's session keys depend on its version; map them explicitly, never by sort order.
SESSION_KEYS = {"0train": "T", "1test": "E", "session_T": "T", "session_E": "E"}


@dataclass
class Run:
    """One continuous recording: EEG plus where each trial starts."""

    data: np.ndarray    # (n_channels, n_samples), µV
    onsets: np.ndarray  # (n_trials,) sample index of each trial start
    labels: np.ndarray  # (n_trials,) class index into CLASSES
    sfreq: float


def load_subject(subject: int) -> dict[str, list[Run]]:
    """Return {"T": [6 runs], "E": [6 runs]} for one subject (downloads on first use)."""
    from moabb.datasets import BNCI2014_001

    sessions = BNCI2014_001().get_data(subjects=[subject])[subject]
    out = {}
    for key, runs in sessions.items():
        if key not in SESSION_KEYS:
            raise KeyError(f"unknown MOABB session key {key!r}")
        out[SESSION_KEYS[key]] = [_to_run(runs[k]) for k in sorted(runs, key=int)]
    return out


def _to_run(raw) -> Run:
    # MOABB stores volts with a stim channel holding the class code (1-4) at each
    # trial's first sample. EOG channels are dropped by picking EEG only.
    stim = raw.get_data(picks="stim")[0]
    onsets = np.flatnonzero(stim)
    return Run(
        data=raw.get_data(picks="eeg") * 1e6,
        onsets=onsets,
        labels=stim[onsets].astype(int) - 1,
        sfreq=raw.info["sfreq"],
    )


def epoch(
    data: np.ndarray,
    onsets: np.ndarray,
    sfreq: float,
    cue_offset_s: float,
    tmin_s: float,
    tmax_s: float,
) -> np.ndarray:
    """Cut a continuous (n_channels, n_samples) run into (n_trials, n_channels, n_window).

    Each window starts at onset + cue_offset + tmin and spans tmax - tmin seconds.
    """
    start = round((cue_offset_s + tmin_s) * sfreq)
    length = round((tmax_s - tmin_s) * sfreq)
    starts = np.asarray(onsets) + start
    if starts.min() < 0 or starts.max() + length > data.shape[1]:
        raise ValueError("an epoch window falls outside the run")
    return np.stack([data[:, s : s + length] for s in starts])
