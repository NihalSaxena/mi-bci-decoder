import numpy as np
from scipy.linalg import eigh

from bci.models.csp_lda import csp_two_class, make_csp_lda

N_CHANNELS, N_TIMES = 8, 250


def _trials(n, loud_channel, rng):
    # White noise, with one channel 1.5x louder (2.25x variance): a fake "this
    # region's rhythm differs for this class" pattern. Keep the effect modest,
    # like real EEG: MNE's 4-class component ranking approximates mutual
    # information and misranks filters when one class is many times louder.
    X = rng.standard_normal((n, N_CHANNELS, N_TIMES))
    X[:, loud_channel] *= 1.5
    return X


def _class_cov(X):
    return np.mean([x @ x.T / x.shape[1] for x in X], axis=0)


def test_csp_lda_learns_which_channel_differs():
    rng = np.random.default_rng(0)
    X = np.concatenate([_trials(60, k, rng) for k in range(4)])
    y = np.repeat(np.arange(4), 60)
    X_test = np.concatenate([_trials(20, k, rng) for k in range(4)])
    y_test = np.repeat(np.arange(4), 20)

    model = make_csp_lda(n_components=4).fit(X, y)
    assert model.predict(X_test).shape == (80,)
    assert np.mean(model.predict(X_test) == y_test) > 0.9


def test_csp_two_class_matches_generalized_eigh():
    rng = np.random.default_rng(1)
    X_a, X_b = _trials(40, 0, rng), _trials(40, 1, rng)
    C_a, C_b = _class_cov(X_a), _class_cov(X_b)

    W = csp_two_class(X_a, X_b, n_components=4)
    assert W.shape == (4, N_CHANNELS)

    # The shortcut: generalized eigenvalues of (C_a, C_a + C_b), ascending.
    d_ref = eigh(C_a, C_a + C_b, eigvals_only=True)
    expected = np.concatenate([d_ref[::-1][:2], d_ref[::-1][-2:]])   # 2 largest, 2 smallest

    # Each filter's share of class-A loudness must equal its eigenvalue.
    d = np.array([w @ C_a @ w / (w @ (C_a + C_b) @ w) for w in W])
    np.testing.assert_allclose(d, expected, rtol=1e-6)

    # Whitened filters: unit total loudness, and outputs don't move together.
    np.testing.assert_allclose(W @ (C_a + C_b) @ W.T, np.eye(4), atol=1e-8)


def test_csp_two_class_finds_the_loud_channel():
    rng = np.random.default_rng(2)
    W = csp_two_class(_trials(40, 0, rng), _trials(40, 1, rng), n_components=2)
    # First filter: loud for A -> weights concentrate on channel 0.
    # Last filter: loud for B -> on channel 1.
    assert np.argmax(np.abs(W[0])) == 0
    assert np.argmax(np.abs(W[-1])) == 1
