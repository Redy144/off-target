"""Testy poprawności solverów. Uruchomienie: uv run pytest"""
import numpy as np
import pytest

from offtarget.data import load_run, save_run
from offtarget.model import NO_TREATMENT, Params, Schedule, beam, grid, initial_condition
from offtarget.observables import front_positions, time_to_regrowth, tumor_mass
from offtarget.solvers import exact_wave, simulate

METHODS = ["rk4", "cn"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dose", ["pulse", "jump"])
def test_pure_kill_matches_exp_formula(method, dose):
    """Bez dyfuzji i wzrostu: u(T) = u0 exp(-beta R(x) * liczba frakcji), bo każdy impuls ma całkę 1."""
    x, _ = grid(40.0, 200)
    p = Params(D=0.0, rho=0.0, beta=0.7, s=3.0, r=5.0)
    sch = Schedule.regular(n=4, start=1.0, interval=0.5)
    u0 = initial_condition(x)
    u = simulate(p, sch, x, [0.0, 4.0], u0=u0, method=method, dose=dose)
    expected = u0 * np.exp(-p.beta * beam(x, p) * 4)
    assert np.max(np.abs(u[-1] - expected)) < 1e-6  # RK4 całkuje impuls z błędem ok. 4e-8, reszta dokładnie


@pytest.mark.parametrize("method", METHODS)
def test_diffusion_conserves_mass(method):
    """Sama dyfuzja z warunkiem Neumanna nie zmienia masy guza."""
    x, _ = grid(40.0, 400)
    u = simulate(Params(rho=0.0, beta=0.0), NO_TREATMENT, x, [0.0, 5.0], method=method)
    M = tumor_mass(u, x)
    assert abs(M[-1] - M[0]) < 1e-10 * M[0]


@pytest.mark.parametrize("method", METHODS)
def test_travelling_wave_second_order(method):
    """Wobec dokładnej fali biegnącej błąd maleje jak dx^2 (dla CN krok czasu rośnie razem z dx)."""
    errors = []
    for n in (400, 800):
        x, dx = grid(80.0, n)
        x = x + 10.0  # obszar [-30, 50]
        dt = dx / 4 if method == "cn" else None
        u = simulate(Params(beta=0.0), NO_TREATMENT, x, [0.0, 2.0], u0=exact_wave(x, 0.0), method=method, dt_max=dt)
        errors.append(np.max(np.abs(u[-1] - exact_wave(x, 2.0))))
    order = np.log2(errors[0] / errors[1])
    assert 1.8 < order < 2.2


def test_front_speed_approaches_fisher_limit():
    """Bez leczenia front zbliża się do 2 sqrt(D rho) z poprawką Bramsona: x(t) ≈ 2t - (3/2) ln t.

    Średnia prędkość na [20, 30] powinna wynosić ok. 2 - 1,5 ln(1,5) / 10 = 1,939 (wzór z pamięci;
    zgodność z solverem sprawdzona wcześniej na fkpp.py).
    """
    x, _ = grid(140.0, 1400)
    x = x + 70.0
    u0 = 1.0 / (1.0 + np.exp((x - 5.0) / 0.5))
    u = simulate(Params(beta=0.0), NO_TREATMENT, x, [0.0, 20.0, 30.0], u0=u0, method="cn", dt_max=0.01)
    right = front_positions(x, u)[:, 1]
    speed = (right[2] - right[1]) / 10.0
    assert abs(speed - (2.0 - 1.5 * np.log(1.5) / 10.0)) < 0.005


def test_solvers_agree_with_treatment():
    """Solver A (RK4) i B (CN) dają tę samą masę guza dla wiązki przesuniętej."""
    x, _ = grid(80.0, 400)
    p = Params(beta=1.5, s=6.0, r=8.0)
    sch = Schedule.regular(n=10, start=1.0, interval=0.4)
    t = np.linspace(0.0, 10.0, 51)
    MA = tumor_mass(simulate(p, sch, x, t, method="rk4"), x)
    MB = tumor_mass(simulate(p, sch, x, t, method="cn"), x)
    assert np.max(np.abs(MB - MA) / MA) < 1e-4


def test_pulse_and_jump_agree_after_treatment():
    """Po leczeniu impulsy i skok LQ dają prawie tę samą masę: różnią się tylko szerokością impulsu."""
    x, _ = grid(80.0, 400)
    p = Params(beta=0.45)
    sch = Schedule.regular(n=10, start=1.0, interval=0.4)
    t = np.concatenate([[0.0], np.linspace(5.0, 10.0, 11)])  # zapis od chwili 0, porównanie po leczeniu
    MP = tumor_mass(simulate(p, sch, x, t, method="cn"), x)[1:]
    MJ = tumor_mass(simulate(p, sch, x, t, method="cn", dose="jump"), x)[1:]
    assert np.max(np.abs(MJ - MP) / MP) < 0.01


@pytest.mark.parametrize("method", METHODS)
def test_shifted_beam_regrows_on_untreated_side(method):
    """Wiązka przesunięta w prawo zostawia lewy brzeg: guz odrasta i przesuwa się w lewo."""
    x, _ = grid(80.0, 400)
    p = Params(beta=1.5, s=6.0, r=8.0)
    sch = Schedule.regular(n=10, start=1.0, interval=0.4)
    t = np.linspace(0.0, 10.0, 101)
    u = simulate(p, sch, x, t, method=method)
    M = tumor_mass(u, x)
    assert np.isfinite(time_to_regrowth(t, M))
    left, right = front_positions(x, u[-1])
    assert -left > right  # więcej guza po stronie, której wiązka nie objęła


def test_schedule_integral_counts_fractions():
    sch = Schedule.regular(n=3, start=1.0, interval=0.5)
    assert abs(sch.integral(0.0, 5.0) - 3.0) < 1e-12
    assert abs(sch.integral(0.0, 1.0) - 0.5) < 1e-12  # połowa pierwszego impulsu


@pytest.mark.parametrize("kwargs", [{"method": "euler"}, {"dose": "jumps"}])
def test_unknown_method_or_dose_raises(kwargs):
    x, _ = grid(10.0, 10)
    with pytest.raises(ValueError):
        simulate(Params(), NO_TREATMENT, x, [0.0, 1.0], **kwargs)


def test_unsorted_t_save_raises():
    x, _ = grid(10.0, 10)
    with pytest.raises(ValueError):
        simulate(Params(), Schedule.regular(3, 1.0, 0.5), x, [0.0, 10.0, 2.0], dose="jump")


# Skok LQ: same zabijanie (D = rho = 0), więc stan zmienia się tylko w chwilach frakcji.
KILL_ONLY = Params(D=0.0, rho=0.0, beta=0.7)


@pytest.mark.parametrize("method", METHODS)
def test_jump_save_at_fraction_is_pre_jump_despite_rounding(method):
    """Frakcje z Schedule.regular(10, 1, 0,4) to m.in. 3,4000000000000004; zapis w 3,4 ma widzieć stan przed nią,
    tak samo jak przy frakcjach, które trafiają w chwilę zapisu dokładnie (np. 3,0)."""
    x, _ = grid(40.0, 200)
    sch = Schedule.regular(n=10, start=1.0, interval=0.4)
    assert 3.4 not in sch.times  # warunek testu: ta frakcja mija się z 3,4 o błąd zaokrąglenia
    t = np.round(np.linspace(0.0, 5.0, 101), 10)
    u0 = initial_condition(x)
    u = simulate(KILL_ONLY, sch, x, t, u0=u0, method=method, dose="jump")
    survival = np.exp(-KILL_ONLY.beta * beam(x, KILL_ONLY))
    for i, tf in enumerate(sch.times):
        k = int(np.argmin(np.abs(t - tf)))
        np.testing.assert_allclose(u[k], u0 * survival**i, rtol=1e-12, err_msg=f"frakcja {tf!r}")
        np.testing.assert_allclose(u[k + 1], u0 * survival ** (i + 1), rtol=1e-12, err_msg=f"frakcja {tf!r}")


@pytest.mark.parametrize("method", METHODS)
def test_jump_fraction_at_start_is_applied(method):
    """Frakcja w chwili t_save[0] działa: u0 to stan tuż przed nią, a skok i impuls dają tę samą frakcję przeżycia."""
    x, _ = grid(40.0, 200)
    sch = Schedule((0.0,), width=0.01)
    u0 = initial_condition(x)
    u = simulate(KILL_ONLY, sch, x, [0.0, 1.0], u0=u0, method=method, dose="jump")
    np.testing.assert_array_equal(u[0], u0)
    np.testing.assert_allclose(u[1], u0 * np.exp(-KILL_ONLY.beta * beam(x, KILL_ONLY)), rtol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_jump_runs_can_be_chained(method):
    """Przebieg [0, 2] i dalej [2, 4] od u(2) daje to samo co [0, 4], także z frakcją dokładnie w t = 2."""
    x, _ = grid(80.0, 400)
    p = Params(beta=0.45)
    sch = Schedule.regular(n=5, start=1.0, interval=0.5)  # frakcje 1,0 ... 3,0, w tym 2,0
    full = simulate(p, sch, x, [0.0, 2.0, 4.0], method=method, dose="jump")
    first = simulate(p, sch, x, [0.0, 2.0], method=method, dose="jump")
    second = simulate(p, sch, x, [2.0, 4.0], u0=first[-1], method=method, dose="jump")
    np.testing.assert_allclose(second[-1], full[-1], rtol=1e-12, atol=1e-15)


def test_save_and_load_roundtrip(tmp_path):
    x, _ = grid(10.0, 10)
    p, sch = Params(beta=0.3), Schedule.regular(3, 1.0, 0.5)
    path = save_run(tmp_path / "run.npz", x, np.array([0.0]), x[None, :], p, sch)
    back = load_run(path)
    assert back["params"] == p and back["schedule"] == sch
    assert np.array_equal(back["x"], x)


def test_rk4_stable_for_huge_beta():
    """Bardzo duże beta (#12): domyślny krok RK4 uwzględnia tempo zabijania, więc wynik zgadza się z solverem B."""
    x, _ = grid(80.0, 400)
    t = np.linspace(0.0, 3.0, 31)
    schedule = Schedule.regular(n=3, start=1.0, interval=0.4)
    p = Params(beta=200.0, r=20.0)
    u_rk4 = simulate(p, schedule, x, t, method="rk4")
    u_cn = simulate(p, schedule, x, t, method="cn", dt_max=0.002)
    assert np.all(np.isfinite(u_rk4)) and u_rk4.min() > -1e-12
    M_rk4, M_cn = tumor_mass(u_rk4, x), tumor_mass(u_cn, x)
    assert np.max(np.abs(M_rk4 - M_cn) / M_cn[0]) < 1e-3


@pytest.mark.parametrize("method", METHODS)
def test_dirichlet_matches_sine_decay(method):
    """Sama dyfuzja z u = 0 na brzegach: sin(pi (x - a) / L) gaśnie jak exp(-D (pi / L)^2 t)."""
    x, _ = grid(10.0, 400)
    L = x[-1] - x[0]
    u0 = np.sin(np.pi * (x - x[0]) / L)
    u = simulate(Params(rho=0.0, beta=0.0), NO_TREATMENT, x, [0.0, 2.0], u0=u0, method=method, bc="dirichlet")
    expected = u0 * np.exp(-((np.pi / L) ** 2) * 2.0)
    assert np.max(np.abs(u[-1] - expected)) < 1e-4
    assert u[-1, 0] == 0.0 and u[-1, -1] == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_bc_irrelevant_far_from_boundary(method):
    """Guz daleko od brzegu nie czuje warunku brzegowego."""
    x, _ = grid(80.0, 400)
    t = [0.0, 3.0]
    sch = Schedule.regular(n=3, start=0.5, interval=0.5)
    un = simulate(Params(beta=1.0, r=4.0), sch, x, t, method=method)
    ud = simulate(Params(beta=1.0, r=4.0), sch, x, t, method=method, bc="dirichlet")
    assert np.max(np.abs(un - ud)) < 1e-8
