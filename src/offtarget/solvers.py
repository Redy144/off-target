"""Solvery równania z `model.py`.

Na start jest jeden solver (A): metoda linii, różnice centralne w x, klasyczny RK4 w czasie,
warunek Neumanna. Pochodzi ze zweryfikowanego `fkpp.py` (rząd zbieżności 2, prędkość frontu
zgodna z teorią). Solver niejawny i wariant ze skokiem LQ to zadania z Zadania 2.
"""
import numpy as np

from .model import Params, Schedule, beam, initial_condition


def laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    """Druga pochodna po x z warunkiem du/dx = 0 na obu brzegach (punkt-duch u[-1] = u[1])."""
    lap = np.empty_like(u)
    lap[..., 1:-1] = (u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]) / dx**2
    lap[..., 0] = 2.0 * (u[..., 1] - u[..., 0]) / dx**2
    lap[..., -1] = 2.0 * (u[..., -2] - u[..., -1]) / dx**2
    return lap


def rk4_step(f, t: float, u: np.ndarray, dt: float) -> np.ndarray:
    """Jeden krok RK4 dla du/dt = f(t, u)."""
    k1 = f(t, u)
    k2 = f(t + 0.5 * dt, u + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, u + 0.5 * dt * k2)
    k4 = f(t + dt, u + dt * k3)
    return u + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def stable_dt(dx: float, D: float, schedule: Schedule, safety: float = 0.4, dt_cap: float = 0.05) -> float:
    """Krok czasowy dla jawnego RK4.

    Dyfuzja: RK4 jest stabilny dla dt <= 0,696 dx^2 / D, bierzemy zapas `safety`.
    Dawka: krok musi rozdzielić impuls, więc dt <= width / 5.
    """
    dt = dt_cap
    if D > 0:
        dt = min(dt, safety * dx**2 / D)
    if schedule.times:
        dt = min(dt, schedule.width / 5)
    return dt


def simulate(params: Params, schedule: Schedule, x: np.ndarray, t_save, u0: np.ndarray | None = None,
             dt_max: float | None = None) -> np.ndarray:
    """Całkuje model od t_save[0] i zwraca pola u w chwilach t_save, tablica (len(t_save), len(x)).

    Siatka x musi być równomierna. Krok dobierany jest tak, żeby trafić dokładnie w każdą chwilę zapisu.
    """
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    t_save = np.asarray(t_save, dtype=float)
    u = initial_condition(x, K=params.K) if u0 is None else np.array(u0, dtype=float)
    R = beam(x, params)
    p = params

    def rhs(t, v):
        return p.D * laplacian_neumann(v, dx) + p.rho * v * (1.0 - v / p.K) - p.beta * R * schedule.U(t) * v

    dt_max = stable_dt(dx, p.D, schedule) if dt_max is None else dt_max
    out = np.empty((len(t_save), len(x)))
    out[0] = u
    t = t_save[0]
    for j in range(1, len(t_save)):
        span = t_save[j] - t_save[j - 1]
        n_steps = max(1, int(np.ceil(span / dt_max - 1e-9)))
        h = span / n_steps
        for _ in range(n_steps):
            u = rk4_step(rhs, t, u, h)
            t += h
        t = t_save[j]
        out[j] = u
    return out


def exact_wave(x, t, D: float = 1.0, rho: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """Dokładna fala biegnąca Fishera-KPP bez leczenia (K = 1), do testów (Ablowitz, Zeppetella 1979).

    U(z) = (1 + exp(z / sqrt(6)))^(-2), z = (x - x0) / sqrt(D / rho) - (5 / sqrt(6)) rho t.
    """
    z = (np.asarray(x) - x0) / np.sqrt(D / rho) - 5.0 / np.sqrt(6.0) * rho * t
    return (1.0 + np.exp(z / np.sqrt(6.0))) ** -2
