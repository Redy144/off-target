"""Jednostki fizyczne a jednostki modelu i zakresy parametrów z literatury (issue #10).

Model liczymy w jednostkach pacjenta odniesienia: czas w 1/rho0, długość w l0 = sqrt(D0 / rho0), gęstość w K.
Pacjent o innych D i rho ma wtedy w modelu D / D0 i rho / rho0. Beta na frakcję się nie zmienia, bo impuls dawki
ma całkę 1 w każdych jednostkach czasu, więc exp(-beta) to frakcja przeżycia niezależnie od skali.

Tabela z wartościami i źródłami: report/README.md, sekcja „Jednostki i zakresy parametrów”.
"""
from dataclasses import dataclass

import numpy as np

from .model import Schedule


@dataclass(frozen=True)
class Scales:
    """Skale pacjenta odniesienia: D w mm^2/dzień, rho w 1/dzień."""

    D: float
    rho: float

    @property
    def length_mm(self) -> float:
        """Jednostka długości modelu w mm: sqrt(D / rho)."""
        return float(np.sqrt(self.D / self.rho))

    @property
    def time_days(self) -> float:
        """Jednostka czasu modelu w dniach: 1 / rho."""
        return 1.0 / self.rho

    def length(self, mm):
        """Długość w mm -> jednostki modelu."""
        return np.asarray(mm) / self.length_mm

    def time(self, days):
        """Czas w dniach -> jednostki modelu."""
        return np.asarray(days) * self.rho

    def model_D(self, D_mm2_per_day):
        """D pacjenta w mm^2/dzień -> D w modelu."""
        return np.asarray(D_mm2_per_day) / self.D

    def model_rho(self, rho_per_day):
        """rho pacjenta w 1/dzień -> rho w modelu."""
        return np.asarray(rho_per_day) / self.rho


# Swanson i in. 2008, Br J Cancer 98:113-119, „Data analysis”: rho = 0,012/dzień przyjęte dla wszystkich
# 70 pacjentów, mediana D/rho z MRI 9,83 mm^2 (zakres 0,24-35,92 mm^2).
SWANSON_RHO = 0.012
SWANSON_D_OVER_RHO = 9.83
REFERENCE = Scales(D=SWANSON_D_OVER_RHO * SWANSON_RHO, rho=SWANSON_RHO)

# Lipková i in. 2019, materiały uzupełniające, sec. A: rozkład a priori log-jednostajny na tych przedziałach,
# za Harpold i in. 2007. D to D_w (istota biała); w istocie szarej D_g = D_w / 10 (sec. II-A).
D_RANGE_MM2_PER_DAY = (0.013, 3.8)
RHO_RANGE_PER_DAY = (0.0027, 0.19)

# Kapteyn i in. 2025 (TumorTwin), sec. 4.2: alpha_RT = 0,05 Gy^-1, beta_RT = 0,005 Gy^-2, czyli alpha/beta = 10 Gy.
# Przykład LUMIERE w repo TumorTwin (examples/lumiere_pipeline.py) startuje kalibrację od alpha = 0,025 Gy^-1.
ALPHA_RANGE_PER_GY = (0.025, 0.05)
ALPHA_BETA_GY = 10.0


def log_kill_per_fraction(dose_gy: float, alpha: float = 0.05, alpha_beta: float = ALPHA_BETA_GY) -> float:
    """Beta modelu dla jednej frakcji z modelu LQ: alpha d + beta_LQ d^2, beta_LQ = alpha / (alpha/beta).

    Frakcja przeżycia w środku wiązki to exp(-beta). Dla 2 Gy i parametrów TumorTwin: 0,1 + 0,02 = 0,12.
    """
    return alpha * dose_gy + alpha / alpha_beta * dose_gy**2


def model_ranges(scales: Scales = REFERENCE, dose_gy: float = 2.0) -> dict[str, tuple[float, float]]:
    """Zakresy D, rho i beta na frakcję w jednostkach modelu, do przeglądów parametrów (#12) i SA (#27).

    D i rho z rozkładu a priori Lipkovej, który jest log-jednostajny, więc też tak je losujemy.
    """
    return {
        "D": tuple(float(v) for v in scales.model_D(D_RANGE_MM2_PER_DAY)),
        "rho": tuple(float(v) for v in scales.model_rho(RHO_RANGE_PER_DAY)),
        "beta": tuple(log_kill_per_fraction(dose_gy, a) for a in ALPHA_RANGE_PER_GY),
    }


def clinical_schedule(n_fractions: int = 30, start_day: float = 0.0, scales: Scales = REFERENCE,
                      width_days: float = 0.05) -> Schedule:
    """Frakcje od poniedziałku do piątku z przerwą weekendową, w jednostkach modelu.

    Domyślnie 30 frakcji w 6 tygodni, jak w TumorTwin (sec. 4.1); `start_day` to poniedziałek pierwszej frakcji.
    Szerokość impulsu podajemy w dniach (domyślnie 0,05 dnia, ok. 1 h), bo domyślna szerokość `Schedule`
    (0,05 jednostki modelu) to u pacjenta odniesienia ok. 4 dni, czyli więcej niż odstęp między frakcjami.
    Przy tak wąskim impulsie solver A bierze krok width / 5, więc do długich przebiegów lepszy jest solver B
    (method="cn") albo dawka „jump”.
    """
    days, day = [], 0
    while len(days) < n_fractions:
        if day % 7 < 5:
            days.append(start_day + day)
        day += 1
    return Schedule(tuple(float(t) for t in scales.time(days)), width=float(scales.time(width_days)))
