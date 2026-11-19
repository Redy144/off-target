"""PINN dla modelu z terapią (issue #24).

    du/dt = D d2u/dx2 + rho u (1 - u/K) - beta R(x) U(t) u

Na podstawie PINN-ów bez terapii z eksperymentów do magisterki (`eksperymenty/fisher_kpp/pinn.py`). Tam sieć
ucząca się samego u przy horyzoncie T = 10 dawała rozwiązanie trywialne u = 0 (błąd 99%), a sieć ucząca się
zmiennej logarytmicznej trzymała błąd ok. 3%. Tu idziemy tą drugą drogą, z w = log(u / K):

    w_t = D (w_xx + w_x^2) + rho (1 - e^w) - beta R(x) U(t)

1. Błąd w w to błąd względny w u, więc sieć nie myli się o stałe 10^-3 przed frontem, gdzie u jest o wiele rzędów
   mniejsze (taki błąd równanie rozmnaża jak e^(rho t)).
2. Człon dawki jest addytywny i ograniczony: frakcja obniża w o beta R(x), zamiast mnożyć u.

Postać sieci:

    w(x, t) = w0(x) + t * scale * N(x, t, C(t))

W t = 0 warunek początkowy jest spełniony dokładnie. C(t) = całka U od 0 do t to liczba frakcji podanych do chwili t
(„zegar dawki”, gładka funkcja schodkowa). Jako trzecie wejście sieci pozwala jej łatwo zrobić skok w czasie przy
każdej frakcji, a przestrzenny kształt skoku sieć uczy się sama. W stracie jest pełne równanie z członem dawki,
a punkty kolokacji są zagęszczone w czasie przy impulsach.

Co nie zadziałało (próby wstępne, scenariusz „pole przesunięte”, 2 tys. iteracji, jedno ziarno):
- Dawka wpisana w postać sieci, w = w0 - beta R(x) C(t) + t * scale * N: błąd 86%. Po 10 frakcjach daje to w w schodek
  głębokości 15 na półcieniu szerokości 0,5 (w_x ~ 30), a w prawdziwym rozwiązaniu dyfuzja rozmywa go od razu
  na ok. sqrt(4 D 0,4) = 1,3. Sieć musiałaby ten sztuczny schodek znosić.
- Bez zegara dawki (wejście tylko x, t): błąd 49% wobec 35% z zegarem przy tej samej liczbie iteracji.

Warunków brzegowych nie wymuszamy: oba brzegi obszaru są daleko od guza, gdzie u ~ e^-30, więc strumień
u_x = u w_x przez brzeg jest pomijalny (Neumann w u spełnia się sam), a w w nie narzuca żadnego warunku.
"""
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import Params, Schedule

# Na Macu z GPU Apple (MPS) trening jest kilka razy szybszy niż na CPU; MPS liczy w float32.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32


def beam_torch(x: torch.Tensor, p: Params) -> torch.Tensor:
    """R(x) jak `model.beam`, na tensorach."""
    return 0.5 * (torch.tanh((x - p.s + p.r) / p.w) - torch.tanh((x - p.s - p.r) / p.w))


def cumulative_dose(t: torch.Tensor, schedule: Schedule) -> torch.Tensor:
    """C(t) = całka U od 0 do t, czyli liczba frakcji podanych do chwili t (C(0) = 0). Pochodna to U(t)."""
    if not schedule.times:
        return torch.zeros_like(t)
    ti = torch.tensor(schedule.times, device=t.device, dtype=t.dtype)
    w = schedule.width
    return 0.5 * (torch.special.erf((t - ti) / w) - torch.special.erf(-ti / w)).sum(dim=-1, keepdim=True)


def dose_rate(t: torch.Tensor, schedule: Schedule) -> torch.Tensor:
    """U(t) jak `Schedule.U`, na tensorach: suma impulsów gaussowskich o całce 1."""
    if not schedule.times:
        return torch.zeros_like(t)
    ti = torch.tensor(schedule.times, device=t.device, dtype=t.dtype)
    w = schedule.width
    return torch.exp(-(((t - ti) / w) ** 2)).sum(dim=-1, keepdim=True) / (w * np.sqrt(np.pi))


def log_initial_condition(center: float = 0.0, radius: float = 6.0, edge: float = 1.0):
    """log(u0 / K) dla `model.initial_condition`: -log(1 + e^z), z = (|x - c| - radius) / edge.

    Liczone jako -logaddexp(0, z), bez utraty precyzji w ogonie (softplus z torcha od z = 20 przybliża log(1 + e^z) przez z).
    """
    def w0(x: torch.Tensor) -> torch.Tensor:
        z = (torch.abs(x - center) - radius) / edge
        return -torch.logaddexp(torch.zeros_like(z), z)
    return w0


def mlp(width: int, depth: int, n_in: int = 2) -> nn.Sequential:
    """Sieć w pełni połączona n_in -> 1 z aktywacją tanh."""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(n_in, width), nn.Tanh()]
        n_in = width
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class LogPINN(nn.Module):
    """w(x, t) = log(u / K) = w0(x) + t * scale * N(x, t, C(t)); zwraca u = K e^w.

    `scale` to typowe tempo zmian w (rzędu rho + D przed frontem), żeby wyjście N było rzędu 1.
    """

    def __init__(self, params: Params, schedule: Schedule, x_range: tuple[float, float], T: float, w0,
                 scale: float = 2.0, width: int = 64, depth: int = 4):
        super().__init__()
        self.params, self.schedule, self.w0 = params, schedule, w0
        self.x_range, self.T, self.scale = (float(x_range[0]), float(x_range[1])), float(T), float(scale)
        self.n_fractions = max(1, len(schedule.times))
        self.net = mlp(width, depth, n_in=3)

    def log_u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, b = self.x_range
        # Wejścia sieci przeskalowane do [-1, 1]: tanh pracuje wtedy w zakresie, w którym się nie nasyca.
        C = cumulative_dose(t, self.schedule)
        z = torch.cat([2.0 * (x - a) / (b - a) - 1.0, 2.0 * t / self.T - 1.0, 2.0 * C / self.n_fractions - 1.0], dim=1)
        return self.w0(x) + t * self.scale * self.net(z)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.params.K * torch.exp(self.log_u(x, t))


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx punkt po punkcie; create_graph pozwala różniczkować wynik jeszcze raz."""
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


def residual(model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Reszta równania dla w: w_t - D (w_xx + w_x^2) - rho (1 - e^w) + beta R U. Dla dokładnego rozwiązania zero.

    `model` musi mieć `log_u(x, t)`, `params` i `schedule`; x i t muszą mieć requires_grad.
    """
    p = model.params
    w = model.log_u(x, t)
    w_x = derivative(w, x)
    return (derivative(w, t) - p.D * (derivative(w_x, x) + w_x**2) - p.rho * (1.0 - torch.exp(w))
            + p.beta * beam_torch(x, p) * dose_rate(t, model.schedule))


@dataclass
class TrainConfig:
    n_res: int = 8192            # punkty kolokacji, losowane od nowa w każdej iteracji
    pulse_fraction: float = 0.25  # część punktów zagęszczona w czasie przy impulsach dawki
    pulse_spread: float = 3.0    # odchylenie czasu tych punktów od środka impulsu, w szerokościach impulsu
    iterations: int = 30_000     # kroki Adama (10 tys. dają ok. 10% błędu, 30 tys. ok. 3%)
    lr: float = 2e-3             # początkowy krok uczenia
    lr_decay: float = 0.01       # krok końcowy = lr * lr_decay (zanik wykładniczy)
    seed: int = 0
    log_every: int = 250


def sample_collocation(model: LogPINN, cfg: TrainConfig, generator: torch.Generator):
    """Punkty kolokacji na nową iterację: x równomiernie, t równomiernie albo (część pulse_fraction) blisko impulsu.

    Losowanie w każdej iteracji zamiast stałego zbioru: sieć nie może dopasować się do kilku tysięcy ustalonych
    punktów, a z czasem widzi cały obszar. Przy impulsach odpowiedź guza zmienia się najszybciej, więc tam zagęszczamy.
    """
    a, b = model.x_range
    n, sch = cfg.n_res, model.schedule
    x = a + (b - a) * torch.rand(n, 1, generator=generator)
    t = model.T * torch.rand(n, 1, generator=generator)
    if sch.times and cfg.pulse_fraction > 0:
        k = int(cfg.pulse_fraction * n)
        centers = torch.tensor(sch.times)[torch.randint(len(sch.times), (k,), generator=generator)]
        near = centers + cfg.pulse_spread * sch.width * torch.randn(k, generator=generator)
        t[:k, 0] = near.clamp(0.0, model.T)
    return (x.to(DEVICE, DTYPE).requires_grad_(True), t.to(DEVICE, DTYPE).requires_grad_(True))


def train(model: LogPINN, cfg: TrainConfig, monitor=None) -> dict:
    """Uczy model (Adam z wykładniczym zanikiem kroku) i zwraca historię: iteracje, strata, monitor, czas.

    `monitor(model)` to opcjonalna miara jakości liczona co `log_every` iteracji (np. błąd wobec solvera).
    """
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    model.to(DEVICE, DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay ** (1.0 / cfg.iterations))
    history = {"iteration": [], "loss": [], "monitor": []}

    def log(iteration, loss):
        history["iteration"].append(iteration)
        history["loss"].append(loss)
        history["monitor"].append(float("nan") if monitor is None else float(monitor(model)))

    tic = time.perf_counter()
    for iteration in range(cfg.iterations):
        x, t = sample_collocation(model, cfg, generator)
        optimizer.zero_grad()
        loss = residual(model, x, t).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % cfg.log_every == 0:
            log(iteration, loss.item())
    x, t = sample_collocation(model, cfg, generator)
    log(cfg.iterations, residual(model, x, t).pow(2).mean().item())
    if DEVICE == "mps":
        torch.mps.synchronize()  # bez tego zegar staje, zanim GPU skończy liczyć
    history["seconds"] = time.perf_counter() - tic
    return {key: np.array(value) for key, value in history.items()}


def predict(model: LogPINN, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pole u na siatce: tablica (len(t), len(x)) jak z `simulate`."""
    X, T = np.meshgrid(x, t)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        u = model(torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype),
                  torch.tensor(T.reshape(-1, 1), device=device, dtype=dtype))
    return u.cpu().numpy().astype(float).reshape(X.shape)


def train_cached(path, params: Params, schedule: Schedule, x_range, T: float, w0, cfg: TrainConfig,
                 monitor=None, scale: float = 2.0, width: int = 64, depth: int = 4, w0_key: str = "plateau"):
    """Trenuje model albo wczytuje go z pliku, jeśli zapisano go z tą samą konfiguracją. Zwraca (model, historia).

    Trening trwa minuty, więc notatnik liczy go raz. `w0_key` opisuje warunek początkowy w kluczu pliku,
    bo funkcji w0 nie da się porównać.
    """
    path = Path(path)
    torch.manual_seed(cfg.seed)  # ziarno ustala też początkowe wagi sieci
    model = LogPINN(params, schedule, x_range, T, w0, scale, width, depth)
    key = {"params": params.to_dict(), "schedule": schedule.to_dict(), "x_range": list(model.x_range), "T": T,
           "w0": w0_key, "scale": scale, "width": width, "depth": depth, **asdict(cfg)}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["key"] == key:
            model.load_state_dict(saved["state"])
            return model.to(DEVICE, DTYPE), saved["history"]
    history = train(model, cfg, monitor)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "state": model.state_dict(), "history": history}, path)
    return model, history
