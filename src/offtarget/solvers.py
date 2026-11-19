"""Solvery równania z `model.py`.

Dwa schematy numeryczne (argument `method`):

- "rk4" (solver A): metoda linii, różnice centralne w x, jawny RK4 w czasie. Pochodzi ze
  zweryfikowanego `fkpp.py`. Krok ograniczają dyfuzja (dt <= 0,4 dx^2 / D) i szerokość impulsu dawki.
- "cn" (solver B): rozszczepienie Stranga. Pół kroku reakcji, pełny krok dyfuzji schematem
  Cranka-Nicolsona (niejawny, stabilny dla każdego dt), pół kroku reakcji. Reakcję w każdym węźle
  liczymy dokładnie (wzór na rozwiązanie równania logistycznego), a dawkę przez dokładną całkę impulsu,
  więc krok nie musi ani spełniać warunku dyfuzji, ani rozdzielać impulsu. Rząd 2 w czasie i w przestrzeni.

Dwa modele dawki (argument `dose`):

- "pulse": człon -beta R(x) U(t) u z impulsami gaussowskimi o całce 1 (model bazowy).
- "jump": frakcja jako natychmiastowy skok u <- u exp(-beta R(x)) w chwili frakcji, między frakcjami
  samo Fisher-KPP (jak w TumorTwin). Przy tym samym beta daje tę samą frakcję przeżycia co impuls.
  Zapis w chwili frakcji to stan tuż przed skokiem.

Wszędzie warunek Neumanna (brak przepływu przez brzeg).
"""
from collections import Counter

import numpy as np
from scipy.linalg import solve_banded

from .model import NO_TREATMENT, Params, Schedule, beam, initial_condition

METHODS = ("rk4", "cn")
DOSES = ("pulse", "jump")


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


def reaction_exact(u: np.ndarray, h: float, rho: float, K: float, kill: np.ndarray | float = 0.0) -> np.ndarray:
    """Dokładne rozwiązanie du/dt = (rho - kill) u - (rho/K) u^2 po czasie h, osobno w każdym węźle.

    Dla a = rho - kill i b = rho / K: u(h) = u0 e^(a h) / (1 + b u0 (e^(a h) - 1) / a),
    a przy a = 0 granica u0 / (1 + b u0 h). `kill` to tempo zabijania uśrednione po kroku.
    """
    a = rho - np.asarray(kill, dtype=float)
    ah = a * h
    small = np.abs(ah) < 1e-12
    g = np.where(small, h, np.expm1(ah) / np.where(small, 1.0, a))  # (e^(a h) - 1) / a
    return u * np.exp(ah) / (1.0 + (rho / K) * u * g)


def _cn_matrix(n: int, dx: float, c: float) -> np.ndarray:
    """Macierz pasmowa (format solve_banded) dla I - c A, gdzie A to laplasjan Neumanna."""
    r = c / dx**2
    ab = np.zeros((3, n))
    ab[0, 1:] = -r          # nad przekątną
    ab[1, :] = 1.0 + 2.0 * r
    ab[2, :-1] = -r         # pod przekątną
    ab[0, 1] = -2.0 * r     # wiersz 0: punkt-duch daje 2 (u1 - u0) / dx^2
    ab[2, -2] = -2.0 * r    # ostatni wiersz: 2 (u_{n-2} - u_{n-1}) / dx^2
    return ab


def _make_stepper(method: str, p: Params, schedule: Schedule, R: np.ndarray, dx: float, n: int):
    """Zwraca funkcję step(t, u, h) dla wybranego schematu; `schedule` jest pusty dla dawki „jump”."""
    if method == "rk4":
        def rhs(t, v):
            out = p.D * laplacian_neumann(v, dx) + p.rho * v * (1.0 - v / p.K)
            if schedule.times:
                out -= p.beta * R * schedule.U(t) * v
            return out

        return lambda t, u, h: rk4_step(rhs, t, u, h)

    cache: dict[float, np.ndarray] = {}

    def reaction(t, u, h):
        kill = p.beta * R * schedule.integral(t, t + h) / h if schedule.times else 0.0
        return reaction_exact(u, h, p.rho, p.K, kill)

    def diffusion(u, h):
        if p.D == 0:
            return u
        c = 0.5 * h * p.D
        if h not in cache:
            cache[h] = _cn_matrix(n, dx, c)
        return solve_banded((1, 1), cache[h], u + c * laplacian_neumann(u, dx))

    def step(t, u, h):
        u = reaction(t, u, 0.5 * h)
        u = diffusion(u, h)
        return reaction(t + 0.5 * h, u, 0.5 * h)

    return step


def _jump_times(schedule: Schedule, t_save: np.ndarray, rtol: float = 1e-9) -> Counter:
    """Frakcje z [t_save[0], t_save[-1]) jako licznik {chwila: liczba frakcji} dla dawki „jump”.

    Frakcję, która mija się z chwilą zapisu tylko o błąd zaokrąglenia (np. 1 + 6 * 0,4 = 3,4000000000000004
    wobec zapisu w 3,4), przesuwamy dokładnie na chwilę zapisu. Inaczej to, czy zapis widzi stan przed skokiem,
    czy po nim, zależałoby od ostatniego bitu liczby.
    """
    tol = rtol * max(1.0, float(np.max(np.abs(t_save))))
    jumps: Counter = Counter()
    for tf in schedule.times:
        k = int(np.argmin(np.abs(t_save - tf)))
        tf = float(t_save[k]) if abs(t_save[k] - tf) <= tol else float(tf)
        if t_save[0] <= tf < t_save[-1]:
            jumps[tf] += 1
    return jumps


def simulate(params: Params, schedule: Schedule, x: np.ndarray, t_save, u0: np.ndarray | None = None,
             method: str = "rk4", dose: str = "pulse", dt_max: float | None = None) -> np.ndarray:
    """Całkuje model od t_save[0] i zwraca pola u w chwilach t_save, tablica (len(t_save), len(x)).

    method: "rk4" (solver A) albo "cn" (solver B), zob. opis modułu.
    dose: "pulse" (impulsy w U(t)) albo "jump" (skok LQ w chwili frakcji). Przy „jump” liczą się frakcje
        z [t_save[0], t_save[-1]), a zapis w chwili frakcji to stan tuż przed nią. u0 to więc stan przed
        frakcją w t_save[0], a przebieg [t0, t1], potem [t1, t2] z u0 = u(t1) daje to samo co [t0, t2].
    dt_max: największy krok; domyślnie warunek stabilności dla "rk4" i 0,02 dla "cn".
    Siatka x musi być równomierna, a t_save niemalejące. Krok trafia w każdą chwilę zapisu i frakcji.
    """
    if method not in METHODS:
        raise ValueError(f"method musi być jednym z {METHODS}, jest {method!r}")
    if dose not in DOSES:
        raise ValueError(f"dose musi być jednym z {DOSES}, jest {dose!r}")
    t_save = np.asarray(t_save, dtype=float)
    if np.any(np.diff(t_save) < 0):
        raise ValueError("t_save musi być niemalejące")
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    u = initial_condition(x, K=params.K) if u0 is None else np.array(u0, dtype=float)
    R = beam(x, params)

    smooth = schedule if dose == "pulse" else NO_TREATMENT
    jumps = _jump_times(schedule, t_save) if dose == "jump" else Counter()
    if dt_max is None:
        dt_max = stable_dt(dx, params.D, smooth) if method == "rk4" else 0.02
    step = _make_stepper(method, params, smooth, R, dx, len(x))

    save_at: dict[float, list[int]] = {}
    for j, ts in enumerate(t_save):
        save_at.setdefault(float(ts), []).append(j)
    out = np.empty((len(t_save), len(x)))

    t = float(t_save[0])
    for te in sorted(set(save_at) | set(jumps)):
        if te > t:
            n_steps = max(1, int(np.ceil((te - t) / dt_max - 1e-9)))
            h = (te - t) / n_steps
            for _ in range(n_steps):
                u = step(t, u, h)
                t += h
            t = te
        for j in save_at.get(te, ()):
            out[j] = u  # stan tuż przed frakcją w tej chwili
        if te in jumps:
            u = u * np.exp(-params.beta * R * jumps[te])
    return out


def exact_wave(x, t, D: float = 1.0, rho: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """Dokładna fala biegnąca Fishera-KPP bez leczenia (K = 1), do testów (Ablowitz, Zeppetella 1979).

    U(z) = (1 + exp(z / sqrt(6)))^(-2), z = (x - x0) / sqrt(D / rho) - (5 / sqrt(6)) rho t.
    """
    z = (np.asarray(x) - x0) / np.sqrt(D / rho) - 5.0 / np.sqrt(6.0) * rho * t
    return (1.0 + np.exp(z / np.sqrt(6.0))) ** -2
