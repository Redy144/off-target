"""Surogaty ODE uzyskane z całkowania przestrzennego modelu PDE.

1. Jednoprzedziałowy model ODE_1 opisuje całkowitą masę guza M(t) = całka u(x, t) dx:

    dM/dt = rho * M * (1 - M / K_eff) - beta * c * U(t) * M

   gdzie K_eff to efektywna pojemność środowiska (w jednostkach masy), a c in [0, 1]
   to ułamek masy guza objęty wiązką promieniowania.

2. Dwuprzedziałowy model ODE_2 rozdziela masę na część w polu wiązki M_in i poza nim M_out:

    dM_in/dt  = rho * M_in  * (1 - M / K_eff) - beta * U(t) * M_in + gamma * (c * M_out - (1 - c) * M_in)
    dM_out/dt = rho * M_out * (1 - M / K_eff)                      - gamma * (c * M_out - (1 - c) * M_in)

   gdzie M = M_in + M_out, a gamma >= 0 to tempo wymiany dyfuzyjnej przez brzeg wiązki
   (proporcjonalne do D). Przy gamma -> inf wymiana utrzymuje M_in = c * M i ODE_2
   redukuje się dokładnie do ODE_1; przy c = 1 i M_out(0) = 0 oba modele są tożsame.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .model import NO_TREATMENT, Params, Schedule, beam
from .observables import tumor_mass
from .solvers import DOSES, reaction_exact


@dataclass(frozen=True)
class ODE1Params:
    """Parametry jednoprzedziałowego modelu ODE_1."""

    rho: float = 1.0
    K_eff: float = 20.0
    beta: float = 1.0
    c: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ODE2Params:
    """Parametry dwuprzedziałowego modelu ODE_2."""

    rho: float = 1.0
    K_eff: float = 30.0
    beta: float = 1.0
    c: float = 0.5
    gamma: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def initial_beam_fraction(x: np.ndarray, u0: np.ndarray, params: Params) -> float:
    """Początkowy ułamek masy guza w wiązce: c_0 = całka R(x) u0(x) dx / całka u0(x) dx."""
    m0 = float(tumor_mass(u0, x))
    if m0 <= 0.0:
        return 0.0
    return float(np.clip(tumor_mass(beam(x, params) * u0, x) / m0, 0.0, 1.0))


def _substeps(t0: float, t1: float, schedule: Schedule, dt_max: float) -> np.ndarray:
    """Węzły czasowe na [t0, t1], zagęszczone wokół impulsów dawki."""
    if t1 <= t0:
        return np.array([t0], dtype=float)
    nodes = [t0, t1]
    if schedule.times:
        half_win = 6.0 * schedule.width
        dt_pulse = min(dt_max, schedule.width / 8.0)
        for ti in schedule.times:
            a = max(t0, ti - half_win)
            b = min(t1, ti + half_win)
            if b > a:
                n = max(1, int(np.ceil((b - a) / dt_pulse)))
                nodes.extend(np.linspace(a, b, n + 1).tolist())
    pts = np.unique(np.asarray(nodes, dtype=float))
    refined = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a > dt_max:
            n = int(np.ceil((b - a) / dt_max))
            refined.extend(np.linspace(a, b, n + 1)[1:].tolist())
        else:
            refined.append(b)
    return np.asarray(refined, dtype=float)


def simulate_ode1(
    M0: float,
    params: ODE1Params,
    schedule: Schedule = NO_TREATMENT,
    t: np.ndarray | None = None,
    *,
    dose: str = "pulse",
    dt_max: float = 0.05,
) -> np.ndarray:
    """Rozwiązuje ODE_1 w chwilach `t` (domyślnie [0, 10] co 0,05)."""
    if dose not in DOSES:
        raise ValueError(f"nieznany model dawki {dose!r}; dostępne: {DOSES}")
    if t is None:
        t = np.round(np.linspace(0.0, 10.0, 201), 10)
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) == 0 or np.any(np.diff(t) <= 0):
        raise ValueError("t musi być rosnącym wektorem chwil czasu")

    out = np.empty(len(t), dtype=float)
    M = float(M0)
    out[0] = M

    if dose == "jump":
        ftimes = sorted(schedule.times)
        eps = 1e-12
        for k in range(len(t) - 1):
            t0, t1 = float(t[k]), float(t[k + 1])
            inside = [ti for ti in ftimes if (t0 < ti <= t1 + eps) or (k == 0 and abs(ti - t0) <= eps)]
            cur = t0
            for ti in inside:
                ti_c = min(max(ti, cur), t1)
                if ti_c > cur:
                    M = float(reaction_exact(np.array([M]), ti_c - cur, params.rho, params.K_eff, 0.0)[0])
                    cur = ti_c
                M *= float(np.exp(-params.beta * params.c))
            if t1 > cur:
                M = float(reaction_exact(np.array([M]), t1 - cur, params.rho, params.K_eff, 0.0)[0])
            out[k + 1] = M
        return out

    for k in range(len(t) - 1):
        pts = _substeps(float(t[k]), float(t[k + 1]), schedule, dt_max)
        for a, b in zip(pts[:-1], pts[1:]):
            dt = float(b - a)
            kill = params.beta * params.c * schedule.integral(float(a), float(b)) / dt
            M = float(reaction_exact(np.array([M]), dt, params.rho, params.K_eff, kill)[0])
        out[k + 1] = M
    return out


def _ode2_growth_diff_step(mi: float, mo: float, dt: float, p: ODE2Params) -> tuple[float, float]:
    """Krok RK4 dla członów wzrostu logistycznego i wymiany dyfuzyjnej w ODE_2."""
    # Ogranicz podkrok, aby RK4 był stabilny również dla dużego gamma lub rho.
    rate = max(p.rho, p.gamma, 1e-12)
    n_sub = max(1, int(np.ceil(dt * rate / 0.25)))
    h = dt / n_sub
    rho, K_eff, c, g = p.rho, p.K_eff, p.c, p.gamma

    def rhs(a: float, b: float) -> tuple[float, float]:
        sat = 1.0 - (a + b) / K_eff if np.isfinite(K_eff) else 1.0
        flux = g * (c * b - (1.0 - c) * a)
        return rho * a * sat + flux, rho * b * sat - flux

    for _ in range(n_sub):
        k1i, k1o = rhs(mi, mo)
        k2i, k2o = rhs(mi + 0.5 * h * k1i, mo + 0.5 * h * k1o)
        k3i, k3o = rhs(mi + 0.5 * h * k2i, mo + 0.5 * h * k2o)
        k4i, k4o = rhs(mi + h * k3i, mo + h * k3o)
        mi = max(0.0, mi + (h / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i))
        mo = max(0.0, mo + (h / 6.0) * (k1o + 2.0 * k2o + 2.0 * k3o + k4o))
    return mi, mo


def simulate_ode2(
    M_in0: float,
    M_out0: float,
    params: ODE2Params,
    schedule: Schedule = NO_TREATMENT,
    t: np.ndarray | None = None,
    *,
    dose: str = "pulse",
    dt_max: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Rozwiązuje ODE_2 w chwilach `t`; zwraca parę tablic (M_in, M_out).

    W trybie `dose="pulse"` stosuje rozszczepienie Stranga: pół kroku dokładnego
    wykładniczego zabijania w M_in, krok wzrostu i wymiany dyfuzyjnej, pół kroku zabijania.
    W trybie `dose="jump"` aplikuje skok M_in <- M_in * exp(-beta) w chwilach frakcji.
    """
    if dose not in DOSES:
        raise ValueError(f"nieznany model dawki {dose!r}; dostępne: {DOSES}")
    if t is None:
        t = np.round(np.linspace(0.0, 10.0, 201), 10)
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) == 0 or np.any(np.diff(t) <= 0):
        raise ValueError("t musi być rosnącym wektorem chwil czasu")

    m_in = np.empty(len(t), dtype=float)
    m_out = np.empty(len(t), dtype=float)
    mi, mo = float(M_in0), float(M_out0)
    m_in[0], m_out[0] = mi, mo

    if dose == "jump":
        ftimes = sorted(schedule.times)
        eps = 1e-12
        for k in range(len(t) - 1):
            t0, t1 = float(t[k]), float(t[k + 1])
            inside = [ti for ti in ftimes if (t0 < ti <= t1 + eps) or (k == 0 and abs(ti - t0) <= eps)]
            cur = t0
            for ti in inside:
                ti_c = min(max(ti, cur), t1)
                if ti_c > cur:
                    mi, mo = _ode2_growth_diff_step(mi, mo, ti_c - cur, params)
                    cur = ti_c
                mi *= float(np.exp(-params.beta))
            if t1 > cur:
                mi, mo = _ode2_growth_diff_step(mi, mo, t1 - cur, params)
            m_in[k + 1], m_out[k + 1] = mi, mo
        return m_in, m_out

    for k in range(len(t) - 1):
        pts = _substeps(float(t[k]), float(t[k + 1]), schedule, dt_max)
        for a, b in zip(pts[:-1], pts[1:]):
            dt = float(b - a)
            dose_int = params.beta * schedule.integral(float(a), float(b))
            half_decay = float(np.exp(-0.5 * dose_int)) if dose_int > 0.0 else 1.0
            mi *= half_decay
            mi, mo = _ode2_growth_diff_step(mi, mo, dt, params)
            mi *= half_decay
        m_in[k + 1], m_out[k + 1] = mi, mo
    return m_in, m_out
