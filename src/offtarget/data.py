"""Jeden format danych dla całego projektu: plik .npz z polami x, t, u, params, schedule.

params i schedule są zapisane jako JSON, żeby plik dało się odczytać bez tego pakietu.
"""
from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from .config import DATA_PROCESSED
from .model import Params, Schedule
from .observables import tumor_mass
from .solvers import simulate


def save_run(path: str | Path, x: np.ndarray, t: np.ndarray, u: np.ndarray, params: Params,
             schedule: Schedule, **extra) -> Path:
    """Zapisuje jedną symulację (albo obserwacje) do .npz. Dodatkowe tablice idą przez `extra`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, t=t, u=u, params=json.dumps(params.to_dict()),
                        schedule=json.dumps(schedule.to_dict()), **extra)
    return path


def load_run(path: str | Path) -> dict:
    """Wczytuje plik z `save_run`; params i schedule wracają jako obiekty Params i Schedule."""
    with np.load(path) as f:
        out = {k: f[k] for k in f.files}
    out["params"] = Params(**json.loads(str(out["params"])))
    sch = json.loads(str(out["schedule"]))
    out["schedule"] = Schedule(tuple(sch["times"]), sch["width"])
    return out


def make_dataset(params_list: Sequence[Params], schedules: Sequence[Schedule], x: np.ndarray,
                 t: np.ndarray, out_dir: str | Path = DATA_PROCESSED, *, prefix: str = "run",
                 u0: np.ndarray | None = None, method: str = "cn", dose: str = "pulse",
                 dt_max: float | None = None) -> list[Path]:
    """Generuje serię symulacji PDE i zapisuje każdą jako `.npz` przez `save_run`.

    Dodatkowo w każdym pliku zapisuje masę guza `M(t)`.
    """
    if len(params_list) != len(schedules):
        raise ValueError(f"Różna liczba parametrów ({len(params_list)}) i harmonogramów ({len(schedules)}).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (p, sched) in enumerate(zip(params_list, schedules)):
        u = simulate(p, sched, x, t, u0=u0, method=method, dose=dose, dt_max=dt_max)
        M = tumor_mass(u, x)
        path = save_run(out_dir / f"{prefix}_{i:04d}.npz", x, t, u, p, sched, M=M)
        paths.append(path)
    return paths


def load_dataset(directory: str | Path, pattern: str = "*.npz") -> list[dict]:
    """Wczytuje wszystkie pliki `.npz` z katalogu w kolejności nazw."""
    return [load_run(p) for p in sorted(Path(directory).glob(pattern))]


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_scenarios(n: int, rng: np.random.Generator, split: str = "train") -> tuple[list[Params], list[Schedule]]:
    """Losuje pary (Params, Schedule) ze stałego generatora `rng`.

    - `train`: regularne harmonogramy (6–12 frakcji, odstęp 0,35–0,55) i plany wiązki r ∈ [3, 16], s ∈ [0, 8].
    - `test`: rozłączny zbiór z harmonogramami (inna liczba frakcji / odstęp / przerwy) i planami wiązki
      spoza treningu (r poza [3, 16] albo s > 8).
    """
    if split not in ("train", "test"):
        raise ValueError(f"Nieznany podział: {split!r}")

    params_list: list[Params] = []
    schedules: list[Schedule] = []
    for i in range(n):
        D = _log_uniform(rng, 0.3, 3.0)
        rho = _log_uniform(rng, 0.5, 2.0)
        beta = float(rng.uniform(0.3, 2.0))
        if split == "train":
            r = float(rng.uniform(3.0, 16.0))
            s = float(rng.uniform(0.0, 8.0))
            n_frac = int(rng.integers(6, 13))
            start = float(rng.uniform(0.8, 1.5))
            interval = float(rng.uniform(0.35, 0.55))
            sched = Schedule.regular(n=n_frac, start=start, interval=interval)
        else:
            # plan wiązki spoza prostokąta treningowego [3, 16] x [0, 8]
            mode = i % 3
            if mode == 0:
                r = float(rng.uniform(1.5, 2.8))
                s = float(rng.uniform(0.0, 8.0))
            elif mode == 1:
                r = float(rng.uniform(16.5, 22.0))
                s = float(rng.uniform(0.0, 8.0))
            else:
                r = float(rng.uniform(3.0, 16.0))
                s = float(rng.uniform(8.5, 13.0))
            # harmonogram spoza treningu: gęstszy, rzadszy albo blokowy z przerwą „weekendową”
            smode = i % 3
            start = float(rng.uniform(0.5, 2.0))
            if smode == 0:
                n_frac = int(rng.integers(14, 19))
                interval = float(rng.uniform(0.20, 0.32))
                sched = Schedule.regular(n=n_frac, start=start, interval=interval)
            elif smode == 1:
                n_frac = int(rng.integers(3, 6))
                interval = float(rng.uniform(0.60, 0.90))
                sched = Schedule.regular(n=n_frac, start=start, interval=interval)
            else:
                # 2 bloki po 5 frakcji z przerwą równą 3 odstępom (wzór 5+2)
                dt_f = float(rng.uniform(0.25, 0.33))
                times = [start + k * dt_f for k in range(5)] + [start + (k + 7) * dt_f for k in range(5)]
                sched = Schedule(tuple(round(v, 6) for v in times))
        params_list.append(Params(D=D, rho=rho, beta=beta, s=s, r=r))
        schedules.append(sched)
    return params_list, schedules
