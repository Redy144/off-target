"""Surogaty ODE uzyskane z całkowania przestrzennego modelu PDE.

Jednoprzedziałowy model ODE_1 opisuje całkowitą masę guza M(t) = całka u(x, t) dx:

    dM/dt = rho * M * (1 - M / K_eff) - beta * c * U(t) * M

gdzie K_eff to efektywna pojemność środowiska (w jednostkach masy, tj. K * długość efektywna),
a c in [0, 1] to ułamek masy guza objęty wiązką promieniowania.
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
    # Zapewnij maksymalny krok dt_max również między impulsami.
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
    """Rozwiązuje ODE_1 w chwilach `t` (domyślnie [0, 10] co 0,05).

    Dla `dose="pulse"` całkuje dokładnie człon logistyczny + zabijanie na podkrokach
    zagęszczonych wokół impulsów (z dokładną całką U(t) na każdym podkroku).
    Dla `dose="jump"` stosuje dokładny krok logistyczny między frakcjami oraz skok
    M <- M * exp(-beta * c) w chwili każdej frakcji.
    """
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

    # dose == "pulse"
    for k in range(len(t) - 1):
        pts = _substeps(float(t[k]), float(t[k + 1]), schedule, dt_max)
        for a, b in zip(pts[:-1], pts[1:]):
            dt = float(b - a)
            kill = params.beta * params.c * schedule.integral(float(a), float(b)) / dt
            M = float(reaction_exact(np.array([M]), dt, params.rho, params.K_eff, kill)[0])
        out[k + 1] = M
    return out
