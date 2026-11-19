"""Wielkości, które mierzymy we wszystkich zadaniach, żeby porównania w Zadaniu 8 liczyły to samo."""
import numpy as np

from .model import Params


def tumor_mass(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Masa guza M = całka u dx (wzór trapezów); działa na ostatniej osi, więc też dla u[t, x].

    Wzór trapezów zgadza się z dyskretnym laplasjanem Neumanna: sama dyfuzja zachowuje M dokładnie.
    """
    return np.trapezoid(u, x, axis=-1)


def front_positions(x: np.ndarray, u: np.ndarray, level: float = 0.5) -> np.ndarray:
    """Lewy i prawy brzeg obszaru u >= level dla każdego pola, tablica (..., 2); NaN, gdy obszaru brak.

    Interpolacja liniowa między węzłami.
    """
    U = np.atleast_2d(u)
    out = np.full((len(U), 2), np.nan)
    for j, v in enumerate(U):
        above = np.flatnonzero(v >= level)
        if len(above) == 0:
            continue
        i, k = above[0], above[-1]
        if i > 0:
            out[j, 0] = x[i - 1] + (level - v[i - 1]) / (v[i] - v[i - 1]) * (x[i] - x[i - 1])
        else:
            out[j, 0] = x[0]
        if k < len(x) - 1:
            out[j, 1] = x[k] + (v[k] - level) / (v[k] - v[k + 1]) * (x[k + 1] - x[k])
        else:
            out[j, 1] = x[-1]
    return out if np.ndim(u) > 1 else out[0]


def min_mass(t: np.ndarray, M: np.ndarray) -> tuple[float, float]:
    """Najmniejsza masa i chwila, w której wystąpiła."""
    i = int(np.argmin(M))
    return float(M[i]), float(t[i])


def time_to_regrowth(t: np.ndarray, M: np.ndarray, threshold: float | None = None) -> float:
    """Pierwsza chwila po minimum masy, w której M wraca do progu (domyślnie M(0)); NaN, gdy nie wraca."""
    threshold = M[0] if threshold is None else threshold
    i = int(np.argmin(M))
    after = np.flatnonzero(M[i:] >= threshold)
    return float(t[i + after[0]]) if len(after) else float("nan")


def coverage(x: np.ndarray, u: np.ndarray, params: Params, level: float = 0.5) -> float | np.ndarray:
    """Pokrycie planu η = |V_REC ∩ CTV| / |V_REC| (Lipková i in. 2019, sec. III-C, eq. 11).

    W 1D objętości to długości odcinków.

    V_REC to obszar u >= level * K, czyli guz, który odrósł; pole u w chwili odrostu podaje wywołujący.
    CTV to pole wiązki [s - r, s + r]. η bliskie 1: guz odrósł w polu (za mała dawka); bliskie 0: odrósł poza
    polem (pole za wąskie albo przesunięte). Samo M(t) tych dwóch przypadków nie rozróżnia.

    Między węzłami u interpolujemy liniowo, jak w `front_positions`, więc długości nie skaczą o dx, gdy brzeg
    guza przesuwa się między węzłami. Działa na ostatniej osi (też dla u[t, x]); NaN, gdy V_REC jest pusty.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    threshold = level * params.K

    # W każdej komórce [x_i, x_{i+1}] interpolacja liniowa przekracza próg na jednym odcinku [lo, hi] (może pustym).
    xa, xb = x[:-1], x[1:]
    a, b = u[..., :-1], u[..., 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = xa + (threshold - a) / (b - a) * (xb - xa)  # używane tylko, gdy próg leży między a i b
    lo = np.where(a >= threshold, xa, np.where(b >= threshold, cross, xa))
    hi = np.where(b >= threshold, xb, np.where(a >= threshold, cross, xa))

    rec = np.sum(hi - lo, axis=-1)
    inside = np.sum(np.clip(np.minimum(hi, params.s + params.r) - np.maximum(lo, params.s - params.r), 0.0, None),
                    axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(rec > 0, inside / rec, np.nan)
    return float(eta) if eta.ndim == 0 else eta
