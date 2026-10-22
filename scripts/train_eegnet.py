"""Train EEGNet on session T and evaluate on session E, one model per subject.

    python scripts/train_eegnet.py
    python scripts/train_eegnet.py --subjects 1 2 --seed 1

Reads the trials cached by `scripts/preprocess.py --band eegnet` (4-40 Hz). Per
subject: k-fold cross-validation on session T picks the number of epochs (best
mean validation accuracy), then a fresh model trains on all of session T for
that many epochs. Session E is only used for the final score. Prints accuracy,
kappa and ITR per subject next to the CSP+LDA kappa (if
reports/results/csp_lda.json exists), and writes reports/results/eegnet_within.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bci.config import load_config
from bci.data import load_epochs
from bci.metrics import score
from bci.models.eegnet import EEGNet
from bci.train import cv_epoch_curve, default_device, fit_fixed, predict, set_seed


def main() -> None:
    cfg = load_config()
    train_cfg = cfg["eegnet"]
    parser = argparse.ArgumentParser(description="EEGNet within-subject, train T / test E.")
    parser.add_argument("--subjects", type=int, nargs="+", default=cfg["dataset"]["subjects"])
    parser.add_argument("--seed", type=int, default=train_cfg["seed"])
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    processed_dir = Path(cfg["paths"]["processed"]) / "eegnet"
    results_dir = Path(cfg["paths"]["results"])
    n_classes = cfg["dataset"]["n_classes"]
    trial_s = cfg["metrics"]["trial_duration_s"]
    csp = _csp_kappas(results_dir / "csp_lda.json")

    print(f"device: {args.device}, seed: {args.seed}")
    print(f"{'subject':>8} {'accuracy':>9} {'kappa':>6} {'ITR':>6} {'CSP kappa':>10} {'epochs':>7} {'CV acc':>7} {'time':>6}")
    results = []
    for subject in args.subjects:
        start = time.time()
        X_t, y_t = load_epochs(processed_dir, subject, "T")
        X_e, y_e = load_epochs(processed_dir, subject, "E")
        opts = {"lr": train_cfg["lr"], "batch_size": train_cfg["batch_size"], "seed": args.seed, "device": args.device}

        def build_model():
            return EEGNet(n_channels=X_t.shape[1], n_times=X_t.shape[2], n_classes=n_classes)

        # 1. Choose the epoch count on session T only: mean validation accuracy over folds.
        curve = cv_epoch_curve(
            build_model, X_t, y_t, n_folds=train_cfg["cv_folds"], max_epochs=train_cfg["max_epochs"], **opts
        )
        n_epochs = int(np.argmax(curve)) + 1

        # 2. Retrain on all of session T for that many epochs, then score session E once.
        set_seed(args.seed)
        model = build_model()
        fit_fixed(model, X_t, y_t, n_epochs, **opts)
        y_pred = predict(model, X_e, device=args.device)

        r = {
            "subject": subject,
            **score(y_e, y_pred, n_classes, trial_s),
            "n_epochs": n_epochs,
            "cv_val_acc": float(curve.max()),
            "cv_curve": curve.round(4).tolist(),
        }
        results.append(r)
        csp_k = f"{csp[subject]:.3f}" if subject in csp else "-"
        print(f"{subject:>8} {r['accuracy']:>9.3f} {r['kappa']:>6.3f} {r['itr_bits_per_min']:>6.2f} "
              f"{csp_k:>10} {n_epochs:>7} {curve.max():>7.3f} {time.time() - start:>5.0f}s", flush=True)

    mean = {k: float(np.mean([r[k] for r in results])) for k in ("accuracy", "kappa", "itr_bits_per_min")}
    csp_mean = np.mean([csp[s] for s in args.subjects]) if all(s in csp for s in args.subjects) else None
    print(f"{'mean':>8} {mean['accuracy']:>9.3f} {mean['kappa']:>6.3f} {mean['itr_bits_per_min']:>6.2f} "
          f"{(f'{csp_mean:.3f}' if csp_mean is not None else '-'):>10}")
    print(f"ITR in bits/min assuming {trial_s} s per trial. Reference: FBCSP mean kappa 0.569.")

    out = results_dir / "eegnet_within.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"seed": args.seed, "trial_duration_s": trial_s, "train": train_cfg, "mean": mean, "subjects": results},
        indent=2,
    ))
    print(f"wrote {out}")


def _csp_kappas(path: Path) -> dict[int, float]:
    """Per-subject CSP+LDA kappa from a previous scripts/train_csp.py run, if any."""
    if not path.exists():
        return {}
    return {r["subject"]: r["kappa"] for r in json.loads(path.read_text())["subjects"]}


if __name__ == "__main__":
    main()
