"""Testy jednostkowe surogatów ODE."""
import numpy as np
import pytest

from offtarget.model import NO_TREATMENT, Params, Schedule, grid, initial_condition
from offtarget.ode import ODE1Params, initial_beam_fraction, simulate_ode1


def test_ode1_no_treatment_matches_logistic():
    M0, rho, K_eff = 12.0, 0.8, 35.0
    t = np.linspace(0.0, 10.0, 101)
    p = ODE1Params(rho=rho, K_eff=K_eff, beta=1.5, c=0.7)
    M = simulate_ode1(M0, p, NO_TREATMENT, t)
    exact = K_eff * M0 / (M0 + (K_eff - M0) * np.exp(-rho * t))
    assert np.allclose(M, exact, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dose", ["pulse", "jump"])
def test_ode1_zero_growth_exact_dose_decay(dose):
    M0, beta, c = 12.0, 1.5, 0.6
    sched = Schedule.regular(n=5, start=1.0, interval=0.5, width=0.05)
    t = np.linspace(0.0, 5.0, 101)
    p = ODE1Params(rho=0.0, K_eff=40.0, beta=beta, c=c)
    M = simulate_ode1(M0, p, sched, t, dose=dose)
    expected = M0 * np.exp(-5 * beta * c)
    assert M[-1] == pytest.approx(expected, rel=1e-6)


def test_ode1_c_zero_ignores_schedule():
    t = np.linspace(0.0, 6.0, 61)
    sched = Schedule.regular(n=5, start=1.0, interval=0.4)
    p = ODE1Params(rho=1.0, K_eff=30.0, beta=2.0, c=0.0)
    m_treat = simulate_ode1(12.0, p, sched, t)
    m_free = simulate_ode1(12.0, p, NO_TREATMENT, t)
    assert np.allclose(m_treat, m_free, rtol=1e-12, atol=1e-12)


def test_ode1_pulse_vs_jump_close():
    t = np.linspace(0.0, 10.0, 201)
    sched = Schedule.regular(n=10, start=1.0, interval=0.4, width=0.05)
    p = ODE1Params(rho=1.0, K_eff=30.0, beta=1.5, c=0.8)
    m_pulse = simulate_ode1(12.0, p, sched, t, dose="pulse")
    m_jump = simulate_ode1(12.0, p, sched, t, dose="jump")
    # Po zakończeniu leczenia impuls (width=0.05) i skok LQ są bardzo bliskie.
    assert abs(m_pulse[-1] - m_jump[-1]) / m_pulse[-1] < 0.01


def test_ode1_nonnegative_for_huge_beta():
    t = np.linspace(0.0, 5.0, 101)
    sched = Schedule.regular(n=4, start=1.0, interval=0.5)
    p = ODE1Params(rho=1.0, K_eff=25.0, beta=50.0, c=1.0)
    M = simulate_ode1(12.0, p, sched, t, dose="pulse")
    assert np.all(np.isfinite(M))
    assert np.all(M >= 0.0)


def test_initial_beam_fraction_bounds():
    x, _ = grid(80.0, 400)
    u0 = initial_condition(x)
    c_full = initial_beam_fraction(x, u0, Params(s=0.0, r=20.0))
    c_narrow = initial_beam_fraction(x, u0, Params(s=0.0, r=3.0))
    c_far = initial_beam_fraction(x, u0, Params(s=35.0, r=2.0))
    assert c_full > 0.99
    assert 0.35 < c_narrow < 0.65
    assert c_far < 1e-4


def test_ode2_diffusion_conserves_mass_and_relaxes_to_c():
    from offtarget.ode import ODE2Params, simulate_ode2

    t = np.linspace(0.0, 10.0, 101)
    p = ODE2Params(rho=0.0, K_eff=30.0, beta=0.0, c=0.7, gamma=1.5)
    mi, mo = simulate_ode2(2.0, 10.0, p, NO_TREATMENT, t)
    assert np.allclose(mi + mo, 12.0, rtol=1e-10, atol=1e-10)
    # Równowaga wymiany: M_in -> c * M = 8.4
    assert mi[-1] == pytest.approx(0.7 * 12.0, rel=1e-5)


def test_ode2_full_coverage_matches_ode1():
    from offtarget.ode import ODE2Params, simulate_ode2

    t = np.linspace(0.0, 10.0, 201)
    sched = Schedule.regular(n=10, start=1.0, interval=0.4)
    p1 = ODE1Params(rho=1.0, K_eff=30.0, beta=1.5, c=1.0)
    p2 = ODE2Params(rho=1.0, K_eff=30.0, beta=1.5, c=1.0, gamma=1.0)
    m1 = simulate_ode1(12.0, p1, sched, t)
    mi, mo = simulate_ode2(12.0, 0.0, p2, sched, t)
    assert np.allclose(mo, 0.0, atol=1e-12)
    assert np.allclose(mi, m1, rtol=1e-3, atol=1e-3)


def test_ode2_fast_mixing_approaches_ode1():
    from offtarget.ode import ODE2Params, simulate_ode2

    t = np.linspace(0.0, 8.0, 161)
    sched = Schedule.regular(n=6, start=1.0, interval=0.5)
    c = 0.65
    p1 = ODE1Params(rho=1.0, K_eff=30.0, beta=1.2, c=c)
    p2 = ODE2Params(rho=1.0, K_eff=30.0, beta=1.2, c=c, gamma=200.0)
    m1 = simulate_ode1(12.0, p1, sched, t)
    mi, mo = simulate_ode2(12.0 * c, 12.0 * (1.0 - c), p2, sched, t)
    assert np.allclose(mi + mo, m1, rtol=0.05, atol=0.1)


@pytest.mark.parametrize("dose", ["pulse", "jump"])
def test_ode2_uncoupled_margin_survives_huge_dose(dose):
    from offtarget.ode import ODE2Params, simulate_ode2

    t = np.linspace(0.0, 6.0, 121)
    sched = Schedule.regular(n=8, start=1.0, interval=0.4)
    p2 = ODE2Params(rho=1.0, K_eff=30.0, beta=20.0, c=0.7, gamma=0.0)
    mi, mo = simulate_ode2(8.4, 3.6, p2, sched, t, dose=dose)
    assert np.all(mi >= 0.0) and np.all(mo >= 0.0)
    assert mi[-1] < 1e-6
    assert mo[-1] > 20.0  # margines poza wiązką odrasta logistycznie
