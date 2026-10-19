"""Train CSP+LDA on session T and evaluate on session E, for every subject.

    python scripts/train_csp.py
    python scripts/train_csp.py --subjects 1 2

Reads the trials cached by scripts/preprocess.py (csp band). Prints accuracy,
kappa and ITR per subject plus the mean, and writes reports/results/csp_lda.json.
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np

from bci.config import load_config
from bci.metrics import accuracy, confusion_matrix, itr, kappa
from bci.models.csp_lda import make_csp_lda


def load_session(processed_dir: Path, subject: int, session: str) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for one subject and session ("T" or "E")."""
    d = np.load(processed_dir / f"A{subject:02d}{session}.npz")
    return d["X"].astype(np.float64), d["y"]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="CSP+LDA baseline, train T / test E.")
    parser.add_argument("--subjects", type=int, nargs="+", default=cfg["dataset"]["subjects"])
    args = parser.parse_args()
    mne.set_log_level("WARNING")

    processed_dir = Path(cfg["paths"]["processed"]) / "csp"
    n_classes = cfg["dataset"]["n_classes"]
    n_components = cfg["csp"]["n_components"]
    trial_s = cfg["metrics"]["trial_duration_s"]

    results = []
    for subject in args.subjects:
        X_train, y_train = load_session(processed_dir, subject=subject, session='T')
        X_test, y_test = load_session(processed_dir=processed_dir, subject=subject, session='E')

        model = make_csp_lda(n_components=n_components).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy(y_test, y_pred)
        results.append({
            "subject": subject,
            "accuracy": acc,
            "kappa": kappa(y_test, y_pred, n_classes),
            "itr_bits_per_min": itr(acc, n_classes, trial_s),
            "confusion": confusion_matrix(y_test, y_pred, n_classes).tolist(),
        })

    print(f"{'subject':>8} {'accuracy':>9} {'kappa':>6} {'ITR (bits/min)':>15}")
    for r in results:
        print(f"{r['subject']:>8} {r['accuracy']:>9.3f} {r['kappa']:>6.3f} {r['itr_bits_per_min']:>15.2f}")
    mean = {k: float(np.mean([r[k] for r in results])) for k in ("accuracy", "kappa", "itr_bits_per_min")}
    print(f"{'mean':>8} {mean['accuracy']:>9.3f} {mean['kappa']:>6.3f} {mean['itr_bits_per_min']:>15.2f}")
    print(f"ITR assumes {trial_s} s per trial. Reference: FBCSP mean kappa 0.569.")

    out = Path(cfg["paths"]["results"]) / "csp_lda.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"trial_duration_s": trial_s, "mean": mean, "subjects": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
