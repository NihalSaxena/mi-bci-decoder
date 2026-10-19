"""CSP + LDA baseline. See docs/DESIGN.md, "Phase 1 in detail".

make_csp_lda is the decoder the pipeline uses (MNE's CSP handles 4 classes).
csp_two_class is a from-scratch 2-class CSP, kept as a tested reference for
what MNE's implementation does.
"""
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def make_csp_lda(n_components: int = 8) -> Pipeline:
    """Unfitted decoder: (n_trials, n_channels, n_times) -> class labels.

    CSP learns n_components spatial filters and outputs the log-variance of
    each filtered signal (n_trials, n_components); LDA classifies those.
    """
    return Pipeline([
        ("csp", CSP(n_components=n_components, log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def csp_two_class(X_a: np.ndarray, X_b: np.ndarray, n_components: int) -> np.ndarray:
    """Spatial filters W (n_components, n_channels) separating class A from class B.

    X_a, X_b: (n_trials, n_channels, n_times) filtered trials of each class.

    1. Covariance of each trial: x @ x.T / n_times                (n_channels, n_channels)
    2. Average per class -> C_a, C_b
    3. C = C_a + C_b
    4. Whitening: eigendecompose C -> eigenvalues lam, eigenvectors U;
       P = diag(1 / sqrt(lam)) @ U.T                             so P @ C @ P.T = I
    5. S_a = P @ C_a @ P.T
    6. Eigendecompose S_a -> eigenvalues d (each in [0, 1]), eigenvectors B
    7. W_full = B.T @ P   (one filter per row)
    8. Keep n_components // 2 rows with the largest d and n_components // 2 with
       the smallest d. Return them ordered by d, largest first.

    Hint: np.linalg.eigh returns eigenvalues in ASCENDING order, with
    eigenvectors as COLUMNS.
    """
    # 1–2. each trial's covariance (how the channels move together), averaged per class
    C_a = np.mean([x @ x.T / x.shape[1] for x in X_a], axis=0)    # (n_ch, n_ch)
    C_b = np.mean([x @ x.T / x.shape[1] for x in X_b], axis=0)

    # 3. both classes together: the combined blob
    C = C_a + C_b

    # 4. whitening: find the combined blob's axes (U) and widths (lam),
    #    then rotate onto those axes and shrink each to width 1
    lam, U = np.linalg.eigh(C)
    P = np.diag(1 / np.sqrt(lam)) @ U.T                            # P @ C @ P.T = I

    # 5. class A's blob, seen in the whitened (round) space
    S_a = P @ C_a @ P.T

    # 6. axes of class A's blob there; d = A's share of loudness on each axis (0 to 1)
    d, B = np.linalg.eigh(S_a)                                     # d ascending; B's columns = axes

    # 7. turn each axis into weights on the real channels (one filter per row)
    W_full = B.T @ P                                               # row i goes with d[i]

    # 8. sort filters from "loud for A" to "loud for B", keep both ends
    W_full = W_full[np.argsort(d)[::-1]]                           # largest d first
    half = n_components // 2
    return np.concatenate([W_full[:half], W_full[-half:]])         # top half + bottom half
