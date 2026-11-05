"""Jeden format danych dla całego projektu: plik .npz z polami x, t, u, params, schedule.

params i schedule są zapisane jako JSON, żeby plik dało się odczytać bez tego pakietu.
"""
import json
from pathlib import Path

import numpy as np

from .model import Params, Schedule


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
