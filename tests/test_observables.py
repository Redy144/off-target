"""Testy wielkości mierzonych na polu u. Uruchomienie: uv run pytest"""
import numpy as np
import pytest

from offtarget.model import Params, grid, initial_condition
from offtarget.observables import coverage

X, _ = grid(80.0, 800)  # dx = 0,1, węzeł w x = 0


@pytest.mark.parametrize("u, s, r, expected", [
    (initial_condition(X), 0.0, 20.0, 1.0),                       # guz w całości w polu
    (initial_condition(X), 30.0, 5.0, 0.0),                       # pole [25, 35] obok guza
    (initial_condition(X), -10.0, 10.0, 0.5),                     # brzeg pola w x = 0 przecina guz w połowie
    (initial_condition(X), 0.0, 0.0, 0.0),                        # puste pole
    (initial_condition(X, center=-10.0, radius=2.0)
     + initial_condition(X, center=10.0, radius=2.0), -10.0, 5.0, 0.5),  # dwa ogniska, w polu tylko lewe
])
def test_coverage_simple_cases(u, s, r, expected):
    assert abs(coverage(X, u, Params(s=s, r=r)) - expected) < 1e-12


def test_coverage_nan_without_tumor():
    assert np.isnan(coverage(X, np.zeros_like(X), Params()))


@pytest.mark.parametrize("K", [1.0, 2.0])
def test_coverage_matches_exact_lengths_off_grid(K):
    """Plateau z `initial_condition` przekracza K/2 dokładnie na [c - 6, c + 6], więc η znamy dokładnie.

    Środek guza i brzeg pola leżą między węzłami; samo liczenie węzłów myliłoby się tu o ok. 0,6% (do dx / 12).
    """
    x, _ = grid(80.0, 400)  # dx = 0,2
    c, s, r = 0.37, -3.0, 4.3  # guz [-5,63; 6,37], pole [-7,3; 1,3]
    expected = (s + r - (c - 6.0)) / 12.0
    u = initial_condition(x, center=c, K=K)
    assert abs(coverage(x, u, Params(K=K, s=s, r=r)) - expected) < 1e-4


def test_coverage_works_along_last_axis():
    """Dla u[t, x] daje η w każdej chwili, tak jak osobne wywołania."""
    p = Params(s=-10.0, r=10.0)
    u = np.stack([initial_condition(X), np.zeros_like(X), initial_condition(X, center=-20.0)])
    eta = coverage(X, u, p)
    assert eta.shape == (3,)
    np.testing.assert_allclose(eta, [coverage(X, v, p) for v in u])
    assert np.isnan(eta[1])
