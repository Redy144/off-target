"""Redukcja rzędu POD–Galerkin dla równania Fishera–KPP z radioterapią.

Metoda (Holmes, Lumley, Berkooz 1996, rozdz. 3–4; Rowley 2005, sekcje 2–3):
1. Ze zbioru migawek pola u(x, t) budujemy bazę ortonormalną w L^2([-L/2, L/2])
   poprzez ważone SVD (z wagami kwadratury trapezów W):
       W^{1/2} S = U Sigma V^T,   Phi = W^{-1/2} U[:, :k],   Phi^T W Phi = I_k.
2. Podstawiamy u(x, t) ≈ sum_{j=1}^k a_j(t) phi_j(x) i rzutujemy PDE metodą Galerkina
   na bazę Phi, otrzymując k-wymiarowy układ ODE:
       da_i/dt = sum_j L_{ij} a_j - (rho/K) sum_{j,l} Q_{ijl} a_j a_l - beta U(t) sum_j B_{ij} a_j.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import Params, Schedule, beam


class GalerkinDivergenceError(RuntimeError):
    """Sygnalizuje wybuch numeryczny zredukowanego układu POD–Galerkin.

    W równaniu Fishera–KPP obcięta baza POD może wywołać ujemne oscylacje Gibbsa (u < 0),
    które człon kwadratowy -(rho/K) u^2 napędza do -inf w skończonym czasie.
    """


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Wagi kwadratury trapezów dla (być może nierównej) siatki 1D."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Siatka x musi być wektorem 1D o co najmniej 2 węzłach.")
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def _neumann_laplacian(dx: float, n: int) -> np.ndarray:
    """Dyskretny operator d2/dx2 drugiego rzędu z jednorodnym warunkiem Neumanna.

    Z wagami trapezów W iloczyn W @ L_fd jest dokładnie symetryczny i niedodatnio określony.
    """
    L = np.zeros((n, n), dtype=float)
    inv_dx2 = 1.0 / (dx * dx)
    idx = np.arange(1, n - 1)
    L[idx, idx - 1] = inv_dx2
    L[idx, idx] = -2.0 * inv_dx2
    L[idx, idx + 1] = inv_dx2
    L[0, 0] = -2.0 * inv_dx2
    L[0, 1] = 2.0 * inv_dx2
    L[-1, -2] = 2.0 * inv_dx2
    L[-1, -1] = -2.0 * inv_dx2
    return L


@dataclass(frozen=True)
class PODBasis:
    """Baza POD ortonormalna w dyskretnym iloczynie skalarnym L^2 (kwadratura trapezów)."""

    x: np.ndarray               # (N_x,)
    weights: np.ndarray         # (N_x,) wagi trapezów W
    modes: np.ndarray           # (N_x, k) kolumny phi_1, ..., phi_k
    singular_values: np.ndarray # pełne widmo wartości osobliwych sigma
    mass_weights: np.ndarray    # (k,) całki int phi_i(x) dx

    @property
    def n_modes(self) -> int:
        return int(self.modes.shape[1])

    @property
    def cumulative_energy(self) -> np.ndarray:
        """Skumulowany udział energii (sum_{i<=k} sigma_i^2) / (sum_i sigma_i^2)."""
        s2 = self.singular_values**2
        total = float(np.sum(s2))
        if total == 0.0:
            return np.ones_like(s2)
        return np.cumsum(s2) / total

    def truncate(self, k: int) -> "PODBasis":
        """Zwraca tę samą bazę obciętą do pierwszych k modów."""
        if not (1 <= k <= self.n_modes):
            raise ValueError(f"k musi należeć do [1, {self.n_modes}], otrzymano {k}.")
        return PODBasis(
            x=self.x,
            weights=self.weights,
            modes=self.modes[:, :k],
            singular_values=self.singular_values,
            mass_weights=self.mass_weights[:k],
        )

    def project(self, u: np.ndarray) -> np.ndarray:
        """Rzut L^2 pola u (kształt (..., N_x)) na współrzędne modalne a (..., k)."""
        u = np.asarray(u, dtype=float)
        return (u * self.weights) @ self.modes

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Rekonstrukcja pola u_hat = sum_i a_i phi_i (kształt (..., N_x))."""
        a = np.asarray(a, dtype=float)
        return a @ self.modes.T

    def mass(self, a: np.ndarray) -> np.ndarray:
        """Masa guza M(t) = int u_hat(x, t) dx = sum_i a_i(t) int phi_i(x) dx."""
        a = np.asarray(a, dtype=float)
        return a @ self.mass_weights


def fit_pod(
    snapshots: np.ndarray | list[np.ndarray],
    x: np.ndarray,
    max_modes: int | None = None,
) -> PODBasis:
    """Wyznacza bazę POD z macierzy migawek (N_t, N_x) lub listy takich trajektorii."""
    x = np.asarray(x, dtype=float)
    if isinstance(snapshots, list):
        S_rows = np.vstack([np.asarray(s, dtype=float) for s in snapshots])
    else:
        S_rows = np.asarray(snapshots, dtype=float)
    if S_rows.ndim != 2 or S_rows.shape[1] != len(x):
        raise ValueError(f"Oczekiwano migawek o kształcie (N_t, {len(x)}), otrzymano {S_rows.shape}.")

    w = trapezoid_weights(x)
    sqrt_w = np.sqrt(w)

    # S_w ma kształt (N_x, N_s); SVD daje lewe wektory osobliwe ortonormalne w Euklidesowym R^{N_x}
    S_w = (S_rows * sqrt_w[None, :]).T
    U, s, _ = np.linalg.svd(S_w, full_matrices=False)

    rank = int(np.sum(s > 1e-14 * max(float(s[0]), 1.0))) if len(s) else 0
    rank = max(rank, 1)
    k = rank if max_modes is None else min(int(max_modes), rank)

    modes = U[:, :k] / sqrt_w[:, None]
    mass_w = w @ modes
    return PODBasis(x=x, weights=w, modes=modes, singular_values=s, mass_weights=mass_w)


@dataclass(frozen=True)
class GalerkinOperators:
    """Zredukowane tensory projekcji Galerkina w bazie POD."""

    basis: PODBasis
    params: Params
    D_mat: np.ndarray  # (k, k) rzut dyfuzji D <phi_i, d2/dx2 phi_j>
    L_mat: np.ndarray  # (k, k) część liniowa: D_mat + rho * I
    B_mat: np.ndarray  # (k, k) rzut wiązki <phi_i, R phi_j>
    Q_ten: np.ndarray  # (k, k, k) rzut nieliniowości <phi_i, phi_j phi_l>


def build_galerkin(basis: PODBasis, p: Params) -> GalerkinOperators:
    """Buduje macierze L, B oraz tensor Q projekcji Galerkina dla danych parametrów PDE."""
    x = basis.x
    dx_arr = np.diff(x)
    if not np.allclose(dx_arr, dx_arr[0], rtol=1e-9, atol=1e-12):
        raise ValueError("build_galerkin wymaga równomiernej siatki x.")
    dx = float(dx_arr[0])

    Phi = basis.modes       # (N_x, k)
    w = basis.weights       # (N_x,)
    k = basis.n_modes

    # Rzut dyfuzji przy użyciu symetrycznego operatora Neumanna (W @ L_fd)
    L_fd = _neumann_laplacian(dx, len(x))
    D_raw = p.D * (Phi.T @ (w[:, None] * (L_fd @ Phi)))
    D_mat = 0.5 * (D_raw + D_raw.T)

    L_mat = D_mat + p.rho * np.eye(k)

    R = beam(x, p)
    B_raw = (Phi * (w * R)[:, None]).T @ Phi
    B_mat = 0.5 * (B_raw + B_raw.T)

    Q_ten = np.einsum("x,xi,xj,xl->ijl", w, Phi, Phi, Phi, optimize=True)
    return GalerkinOperators(basis=basis, params=p, D_mat=D_mat, L_mat=L_mat, B_mat=B_mat, Q_ten=Q_ten)


def simulate_pod_galerkin(
    u0: np.ndarray,
    basis: PODBasis,
    p: Params,
    schedule: Schedule,
    t: np.ndarray,
    *,
    k: int | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    blowup_norm: float = 1e4,
) -> tuple[np.ndarray, np.ndarray]:
    """Całkuje zredukowany układ POD–Galerkin metodą RK45 i zwraca (a_traj, u_hat).

    a_traj: (len(t), k_used) współrzędne modalne,
    u_hat:  (len(t), N_x) zrekonstruowane pole gęstości guza.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Wektor t musi być ściśle rosnący i mieć co najmniej 2 punkty.")

    b = basis if k is None else basis.truncate(k)
    ops = build_galerkin(b, p)
    rho_over_K = p.rho / p.K
    L_mat, B_mat, Q_ten = ops.L_mat, ops.B_mat, ops.Q_ten

    def rhs(ti: float, a: np.ndarray) -> np.ndarray:
        quad = np.einsum("ijl,j,l->i", Q_ten, a, a)
        u_rate = schedule.U(ti)
        out = L_mat @ a - rho_over_K * quad
        if u_rate != 0.0:
            out = out - (p.beta * u_rate) * (B_mat @ a)
        return out

    def blowup_event(_ti: float, a: np.ndarray) -> float:
        return blowup_norm - float(np.linalg.norm(a))

    blowup_event.terminal = True
    blowup_event.direction = -1.0

    # Dzielimy horyzont na przedziały impulsowe i swobodne, żeby adaptacyjne RK45
    # dokładnie rozdzieliło wąskie impulsy U(t) (krok <= width / 4), a pomiędzy nimi biegło szybko.
    t0, t1 = float(t[0]), float(t[-1])
    w = schedule.width
    cuts = {t0, t1}
    for ti in schedule.times:
        for edge in (ti - 5.0 * w, ti + 5.0 * w):
            if t0 < edge < t1:
                cuts.add(float(edge))
    segments = sorted(cuts)

    a_out = np.empty((len(t), b.n_modes), dtype=float)
    a_curr = b.project(u0)
    a_out[0] = a_curr

    for seg_start, seg_end in zip(segments[:-1], segments[1:]):
        mask = (t > seg_start) & (t <= seg_end)
        t_eval = t[mask]
        mid = 0.5 * (seg_start + seg_end)
        in_pulse = any(abs(mid - ti) <= 5.0 * w for ti in schedule.times)
        max_step = (w / 4.0) if in_pulse else 0.1

        sol = solve_ivp(
            rhs,
            (seg_start, seg_end),
            a_curr,
            method="RK45",
            t_eval=t_eval if len(t_eval) else None,
            events=blowup_event,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )
        if sol.status == 1:
            t_div = float(sol.t_events[0][0]) if len(sol.t_events[0]) else seg_end
            raise GalerkinDivergenceError(
                f"Układ POD–Galerkin (k={b.n_modes}) wybuchł numerycznie przy t ≈ {t_div:.2f} "
                f"(||a|| przekroczyło {blowup_norm:g})."
            )
        if not sol.success:
            raise GalerkinDivergenceError(
                f"Integracja POD–Galerkin (k={b.n_modes}) nie powiodła się: {sol.message}"
            )
        if len(t_eval):
            a_out[mask] = sol.y.T
        a_curr = sol.y[:, -1]

    u_hat = b.reconstruct(a_out)
    return a_out, u_hat
