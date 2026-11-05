"""Jedna komenda, która odtwarza wszystkie wyniki: `uv run reproduce`.

Uruchamia po kolei wszystkie notatniki z notebooks/ (w kolejności nazw) w czystym jądrze.
Notatniki zapisują wykresy do results/figures, tabele do results/tables, a dane do data/.
Opcje:
    uv run reproduce --list            pokaż, co zostanie uruchomione
    uv run reproduce --only zad2       tylko notatniki z „zad2” w nazwie
    uv run reproduce --save            zapisz wykonane notatniki (z wynikami) do results/notebooks/
"""
import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from .config import ROOT, SEED, seed_everything

NOTEBOOKS = ROOT / "notebooks"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reproduce", description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="uruchom tylko notatniki zawierające ten tekst w nazwie")
    ap.add_argument("--list", action="store_true", help="tylko wypisz notatniki")
    ap.add_argument("--save", action="store_true", help="zapisz wykonane notatniki do results/notebooks/")
    ap.add_argument("--timeout", type=int, default=3600, help="limit czasu na komórkę w sekundach")
    args = ap.parse_args(argv)

    paths = sorted(p for p in NOTEBOOKS.glob("*.ipynb") if args.only in p.name)
    if args.list or not paths:
        print("\n".join(p.name for p in paths) or "Brak notatników do uruchomienia.")
        return 0

    seed_everything(SEED)
    os.environ["OFFTARGET_SEED"] = str(SEED)
    out_dir = ROOT / "results" / "notebooks"
    for p in paths:
        start = time.perf_counter()
        print(f"▶ {p.name} ...", end=" ", flush=True)
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(NOTEBOOKS)}})
        try:
            client.execute()
        except CellExecutionError as e:  # pokaż, który notatnik padł, i przerwij
            print("BŁĄD")
            print(e, file=sys.stderr)
            return 1
        print(f"ok ({time.perf_counter() - start:.1f} s)")
        if args.save:
            out_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, out_dir / p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
