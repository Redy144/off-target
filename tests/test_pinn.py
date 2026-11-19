"""Testy PINN-u z terapią: postać sieci i reszta równania na rozwiązaniach dokładnych. Uruchomienie: uv run pytest"""
import numpy as np
import torch

from offtarget.model import NO_TREATMENT, Params, Schedule, initial_condition
from offtarget.pinn import (
    LogPINN,
    TrainConfig,
    beam_torch,
    cumulative_dose,
    dose_rate,
    log_initial_condition,
    residual,
    train,
)

SCHEDULE = Schedule.regular(n=4, start=1.0, interval=0.4)


def points(x, t):
    """Tensory kolumnowe float64 z requires_grad, do residual()."""
    as_col = lambda a: torch.tensor(np.asarray(a, dtype=float).reshape(-1, 1)).requires_grad_(True)
    return as_col(x), as_col(t)


def model64(params, schedule=SCHEDULE):
    torch.manual_seed(0)
    return LogPINN(params, schedule, (-40.0, 40.0), 10.0, log_initial_condition()).double()


def test_cumulative_dose_and_rate_match_schedule():
    t = np.linspace(0.0, 4.0, 81)
    C = cumulative_dose(torch.tensor(t).reshape(-1, 1), SCHEDULE).numpy().ravel()
    U = dose_rate(torch.tensor(t).reshape(-1, 1), SCHEDULE).numpy().ravel()
    np.testing.assert_allclose(C, [SCHEDULE.integral(0.0, s) for s in t], atol=1e-12)
    np.testing.assert_allclose(U, [SCHEDULE.U(s) for s in t], rtol=1e-12)


def test_initial_condition_is_exact():
    """Postać w0 + t * (...) spełnia warunek początkowy dokładnie, też daleko w ogonie (u ~ 1e-15)."""
    x = np.linspace(-40.0, 40.0, 161)
    u = model64(Params())(*[torch.tensor(a.reshape(-1, 1)) for a in (x, np.zeros_like(x))])
    np.testing.assert_allclose(u.detach().numpy().ravel(), initial_condition(x), rtol=1e-12)


def test_residual_vanishes_for_pure_kill():
    """D = rho = 0: dokładne rozwiązanie to w0(x) - beta R(x) C(t). Sprawdza znak i skalę członu dawki."""

    class PureKill:
        params, schedule = Params(D=0.0, rho=0.0, beta=1.5, s=3.0, r=5.0), SCHEDULE

        def log_u(self, x, t):
            return log_initial_condition()(x) - self.params.beta * beam_torch(x, self.params) * cumulative_dose(
                t, self.schedule)

    x, t = points(np.linspace(-20.0, 20.0, 41).repeat(5), np.tile([0.3, 1.0, 1.02, 1.4, 3.0], 41))
    assert residual(PureKill(), x, t).abs().max().item() < 1e-9


def test_residual_vanishes_for_travelling_wave():
    """Bez leczenia (D = rho = K = 1) fala Ablowitza-Zeppetelli u = (1 + e^(z/sqrt 6))^-2, z = x - 5t/sqrt 6,
    spełnia równanie dokładnie, więc reszta dla w = log u ma być zerem. Sprawdza człony dyfuzji i wzrostu."""

    class Wave:
        params, schedule = Params(beta=0.0), NO_TREATMENT

        def log_u(self, x, t):
            z = x - 5.0 / np.sqrt(6.0) * t
            return -2.0 * torch.nn.functional.softplus(z / np.sqrt(6.0))

    x, t = points(np.linspace(-20.0, 20.0, 81), np.linspace(0.0, 3.0, 81))
    assert residual(Wave(), x, t).abs().max().item() < 1e-12


def test_training_smoke():
    """Kilkadziesiąt kroków na małej liczbie punktów: strata skończona i maleje."""
    m = LogPINN(Params(beta=1.5, s=6.0, r=8.0), SCHEDULE, (-40.0, 40.0), 5.0, log_initial_condition())
    history = train(m, TrainConfig(n_res=512, iterations=60, log_every=20))
    assert np.all(np.isfinite(history["loss"]))
    assert history["loss"][-1] < history["loss"][0]
