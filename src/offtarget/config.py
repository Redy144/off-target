"""Ścieżki projektu i stałe ziarno losowe.

Każdy notatnik zaczyna się od `rng = seed_everything()`. Losowość bierzemy potem
z `rng` (NumPy) albo z ustawionego ziarna PyTorcha, nigdy z niezasianego generatora.
"""
import os
import random
from pathlib import Path

import numpy as np

SEED = 2026

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = RESULTS / "models"


def seed_everything(seed: int = SEED) -> np.random.Generator:
    """Ustawia ziarno w random, NumPy i PyTorchu; zwraca generator NumPy do dalszego użycia."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    for d in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    return np.random.default_rng(seed)
