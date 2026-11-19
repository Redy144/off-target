"""Model bazowy: guz 1D pod radioterapią.

    du/dt = D d2u/dx2 + rho u (1 - u/K) - beta R(x) U(t) u

Jednostki są bezwymiarowe: czas w 1/rho, długość w sqrt(D/rho), gęstość w K.
Bazowo D = rho = K = 1.

U(t) to suma wąskich impulsów gaussowskich o całce 1, po jednym na frakcję dawki.
Dzięki temu beta to „log-zabicie” jednej frakcji: w środku wiązki przeżywa ułamek
exp(-beta) komórek. Odpowiednik modelu liniowo-kwadratowego (LQ): beta = alpha d + beta_LQ d^2.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import erf


@dataclass(frozen=True)
class Params:
    """Parametry modelu. Wartości domyślne są ilustracyjne, nie kliniczne."""

    D: float = 1.0      # dyfuzja: naciekanie tkanki
    rho: float = 1.0    # tempo podziałów
    K: float = 1.0      # pojemność
    beta: float = 1.0   # log-zabicie jednej frakcji w środku wiązki
    s: float = 0.0      # środek wiązki
    r: float = 20.0     # połowa szerokości wiązki
    w: float = 0.5      # szerokość półcienia, czyli brzegu wiązki

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Schedule:
    """Harmonogram frakcji: chwile podania dawki i szerokość impulsu."""

    times: tuple[float, ...] = ()
    width: float = 0.05

    @classmethod
    def regular(cls, n: int, start: float, interval: float, width: float = 0.05) -> "Schedule":
        """n frakcji co `interval`, pierwsza w chwili `start`."""
        return cls(tuple(float(start + i * interval) for i in range(n)), width)

    def U(self, t: float) -> float:
        """Natężenie dawki w chwili t; każdy impuls ma całkę 1."""
        if not self.times:
            return 0.0
        z = (t - np.asarray(self.times)) / self.width
        return float(np.sum(np.exp(-(z**2))) / (self.width * np.sqrt(np.pi)))

    def integral(self, t0: float, t1: float) -> float:
        """Dokładna całka U(t) dt po [t0, t1]: każdy impuls daje ½[erf((t1 - ti)/w) - erf((t0 - ti)/w)]."""
        if not self.times:
            return 0.0
        ti = np.asarray(self.times)
        return float(0.5 * np.sum(erf((t1 - ti) / self.width) - erf((t0 - ti) / self.width)))

    def to_dict(self) -> dict:
        return {"times": list(self.times), "width": self.width}


NO_TREATMENT = Schedule()


def beam(x: np.ndarray, p: Params) -> np.ndarray:
    """Profil wiązki R(x): około 1 w polu [s - r, s + r], 0 poza nim, gładki brzeg o szerokości w."""
    return 0.5 * (np.tanh((x - p.s + p.r) / p.w) - np.tanh((x - p.s - p.r) / p.w))


def initial_condition(x: np.ndarray, center: float = 0.0, radius: float = 6.0, edge: float = 1.0,
                      K: float = 1.0) -> np.ndarray:
    """Guz przed leczeniem: plateau o gęstości K i promieniu `radius`, z łagodnym brzegiem."""
    return K / (1.0 + np.exp((np.abs(x - center) - radius) / edge))


def grid(L: float, n: int) -> tuple[np.ndarray, float]:
    """Siatka n + 1 węzłów na [-L/2, L/2], z węzłami na obu brzegach. Zwraca (x, dx)."""
    return np.linspace(-L / 2, L / 2, n + 1), L / n
