"""Filter and epoch every subject, caching the trials and exporting the filter spec.

    python scripts/preprocess.py                  # csp band, all subjects
    python scripts/preprocess.py --band eegnet --subjects 1 2

Writes, per subject and session:
    data/processed/<band>/A01T.npz   X (288, 22, 500) float32 µV, y (288,), ch_names
and once per band:
    artifacts/filters/<band>.json    the filter spec the real-time engine loads
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bci.config import load_config
from bci.data import epoch, load_subject
from bci.filters import causal_filter, design_bandpass, filter_spec


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Filter and epoch BCI IV 2a.")
    parser.add_argument("--band", default="csp", choices=sorted(cfg["filters"]))
    parser.add_argument("--subjects", type=int, nargs="+", default=cfg["dataset"]["subjects"])
    args = parser.parse_args()

    band, window = cfg["filters"][args.band], cfg["epoching"]
    sfreq = cfg["dataset"]["sfreq"]
    sos = design_bandpass(band["l_freq"], band["h_freq"], sfreq, band["order"])

    out_dir = Path(cfg["paths"]["processed"]) / args.band
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_names = None
    for subject in args.subjects:
        for session, runs in load_subject(subject).items():
            X, y = [], []
            for run in runs:
                if run.sfreq != sfreq:
                    raise ValueError(f"subject {subject}: expected {sfreq} Hz, got {run.sfreq}")
                if ch_names is not None and run.ch_names != ch_names:
                    raise ValueError(f"subject {subject}: channel order differs between runs")
                ch_names = run.ch_names

                # Filter the whole continuous run first, then cut: matches what a stream sees.
                filtered, _ = causal_filter(run.data, sos)
                X.append(epoch(filtered, run.onsets, sfreq, **window))
                y.append(run.labels)

            X, y = np.concatenate(X).astype(np.float32), np.concatenate(y)
            path = out_dir / f"A{subject:02d}{session}.npz"
            np.savez(path, X=X, y=y, ch_names=np.array(ch_names))
            print(f"{path}  X {X.shape}  per class {np.bincount(y, minlength=4)}")

    spec = filter_spec(
        args.band, band["l_freq"], band["h_freq"], band["order"], sfreq, ch_names, **window
    )
    spec_path = Path(cfg["paths"]["artifacts"]) / "filters" / f"{args.band}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"{spec_path}  ({len(spec['sos'])} sections, {len(ch_names)} channels)")


if __name__ == "__main__":
    main()
