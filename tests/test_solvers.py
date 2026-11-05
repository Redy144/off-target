"""Testy poprawności solvera. Uruchomienie: uv run pytest"""
import numpy as np

from offtarget.data import load_run, save_run
from offtarget.model import NO_TREATMENT, Params, Schedule, beam, grid, initial_condition
from offtarget.observables import front_positions, time_to_regrowth, tumor_mass
from offtarget.solvers import exact_wave, simulate


def test_pure_kill_matches_exp_formula():
    """Bez dyfuzji i wzrostu: u(T) = u0 exp(-beta R(x) * liczba frakcji), bo każdy impuls ma całkę 1."""
    x, _ = grid(40.0, 200)
    p = Params(D=0.0, rho=0.0, beta=0.7, s=3.0, r=5.0)
    sch = Schedule.regular(n=4, start=1.0, interval=0.5)
    u0 = initial_condition(x)
    u = simulate(p, sch, x, [0.0, 4.0], u0=u0)
    expected = u0 * np.exp(-p.beta * beam(x, p) * 4)
    assert np.max(np.abs(u[-1] - expected)) < 1e-6  # błąd całkowania impulsu przez RK4 to ok. 4e-8


def test_diffusion_conserves_mass():
    """Sama dyfuzja z warunkiem Neumanna nie zmienia masy guza."""
    x, _ = grid(40.0, 400)
    p = Params(rho=0.0, beta=0.0)
    u = simulate(p, NO_TREATMENT, x, [0.0, 5.0])
    M = tumor_mass(u, x)
    assert abs(M[-1] - M[0]) < 1e-10 * M[0]


def test_travelling_wave_second_order():
    """Wobec dokładnej fali biegnącej błąd maleje jak dx^2."""
    errors = []
    for n in (400, 800):
        x, _ = grid(80.0, n)
        x = x + 10.0  # obszar [-30, 50]
        u = simulate(Params(beta=0.0), NO_TREATMENT, x, [0.0, 2.0], u0=exact_wave(x, 0.0))
        errors.append(np.max(np.abs(u[-1] - exact_wave(x, 2.0))))
    order = np.log2(errors[0] / errors[1])
    assert 1.8 < order < 2.2


def test_shifted_beam_regrows_on_untreated_side():
    """Wiązka przesunięta w prawo zostawia lewy brzeg: guz odrasta i przesuwa się w lewo."""
    x, _ = grid(80.0, 400)
    p = Params(beta=1.5, s=6.0, r=8.0)
    sch = Schedule.regular(n=10, start=1.0, interval=0.4)
    t = np.linspace(0.0, 10.0, 101)
    u = simulate(p, sch, x, t)
    M = tumor_mass(u, x)
    assert np.isfinite(time_to_regrowth(t, M))
    left, right = front_positions(x, u[-1])
    assert -left > right  # więcej guza po stronie, której wiązka nie objęła


def test_save_and_load_roundtrip(tmp_path):
    x, _ = grid(10.0, 10)
    p, sch = Params(beta=0.3), Schedule.regular(3, 1.0, 0.5)
    path = save_run(tmp_path / "run.npz", x, np.array([0.0]), x[None, :], p, sch)
    back = load_run(path)
    assert back["params"] == p and back["schedule"] == sch
    assert np.array_equal(back["x"], x)
