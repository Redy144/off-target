# off-target

Projekt z Informatyki Systemów Złożonych (AGH, 2026/27, prowadzący W. Dzwinel): **radioterapia guza przy niedopasowanym polu napromieniania**. Pytanie projektu: co się dzieje z guzem, gdy pole wiązki nie pokrywa go dokładnie, i czy da się to szybko przewidzieć. Idziemy od solvera PDE, przez surogat ODE, analizę wrażliwości, asymilację danych, NODE i UDE, aż do PINN-ów i SuperNetu.

Model bazowy (1D, jednostki bezwymiarowe):

```latex
\frac{\partial u}{\partial t} = D\,\partial_{xx} u + \rho\, u\left(1-\frac{u}{K}\right) - \beta\, R(x)\, U(t)\, u
```

u to gęstość komórek guza, R(x) to pole wiązki (środek s, połowa szerokości r), U(t) to harmonogram frakcji (impulsy o całce 1), β to log-zabicie jednej frakcji. Mierzymy masę guza M(t) = ∫u dx, najmniejszą masę po leczeniu i czas do odrostu.

**Plan tydzień po tygodniu: [PLANS.md](PLANS.md).** Zadania są w [issue](https://github.com/Redy144/off-target/issues), pogrupowane w kamienie milowe według prezentacji.

## Start

Potrzebny jest tylko [uv](https://docs.astral.sh/uv/); Pythona 3.12 uv pobierze sam.

```bash
uv sync                      # środowisko z uv.lock, te same wersje u obu osób
uv run nbstripout --install  # raz po sklonowaniu: wyniki z notatników nie trafiają do gita
uv run pytest                # testy solvera
uv run jupyter lab           # praca w notatnikach
```

Nową bibliotekę dodajemy przez `uv add nazwa` (zmienia `pyproject.toml` i `uv.lock`, oba commitujemy).

## Odtworzenie wszystkich wyników jedną komendą

```bash
uv run reproduce
```

Komenda uruchamia po kolei wszystkie notatniki z `notebooks/` w czystym jądrze, ze stałym ziarnem. Notatniki zapisują wykresy do `results/figures/`, tabele do `results/tables/`, a dane do `data/`. Przydatne opcje: `--list`, `--only zad2`, `--save` (zapisuje wykonane notatniki do `results/notebooks/`).

**Stałe ziarno:** `SEED = 2026` w `src/offtarget/config.py`. Każdy notatnik zaczyna się od `rng = seed_everything()` i bierze losowość tylko z `rng` albo z zasianego PyTorcha.

## Struktura

```
src/offtarget/        kod wspólny dla wszystkich zadań
  config.py           ścieżki i ziarno
  model.py            parametry, pole wiązki R(x), harmonogram U(t), warunek początkowy, siatka
  solvers.py          simulate(params, schedule, x, t_save, u0) -> u[t, x]
  observables.py      masa guza, fronty, najmniejsza masa, czas do odrostu
  data.py             zapis i odczyt .npz
  reproduce.py        komenda `uv run reproduce`
notebooks/            notatniki Jupytera, nazwy zadN_KK_opis.ipynb, uruchamiane w kolejności nazw
tests/                testy (pytest)
data/raw/             dane „rzeczywiste”: prawda i obserwacje z szumem (odtwarzane, poza gitem)
data/processed/       zbiory do uczenia surogatów (odtwarzane, poza gitem)
results/figures/      wykresy (w gicie)
results/tables/       tabele CSV (w gicie)
results/models/       wytrenowane sieci (poza gitem)
report/               prezentacje, notatki mówców, notatki z literatury
```

## Biblioteki

Wszystkie są w `pyproject.toml`; najważniejsze według zadań:

| Zadanie | Biblioteki |
| --- | --- |
| 2. Solvery | numpy, scipy |
| 3. Surogat ODE, sieć rezerwuarowa | scipy, reservoirpy |
| 4. SA | SALib |
| 5. DA | pymc (ABC przez `pm.Simulator` i SMC), scipy (Levenberg-Marquardt) |
| 6. NODE, UDE | torch, torchdiffeq |
| 7–8. PINN, SuperNet | torch |
| Analiza i dodatki | pandas, openpyxl, scikit-learn, shap, pykan, tabpfn, matplotlib |
| Notatniki | jupyterlab, ipykernel, jupytext, nbclient, nbstripout |

## Konwencje

- **Jedno API:** każdy model przyjmuje `Params` i `Schedule` z `model.py` i zwraca pola na siatce `x` w chwilach `t_save`.
- **Jeden format danych:** `.npz` z polami `x`, `t`, `u`, `params`, `schedule` (`data.save_run` i `data.load_run`).
- **Logika w `src/`, notatniki tylko ją wywołują i rysują.** Nowa funkcja w `src/` dostaje test w `tests/`.
- **Zbiór testowy zawsze ma harmonogramy dawek i plany wiązki spoza treningu.**
- **Jednostki bezwymiarowe:** czas w 1/ρ, długość w √(D/ρ), gęstość w K. Bazowo D = ρ = K = 1. Dawka: β = α·d + β_LQ·d², dla 2 Gy ≈ 0,12 na frakcję. Tabela przeliczeń na mm i dni powstanie w issue #10.
- Komentarze, opisy i notatniki po polsku.

## Podział ról

| Osoba | Zadania prowadzącego | Odpowiada za |
| --- | --- | --- |
| Krzysztof Kapera | 2–6 | Solvery, surogat ODE i sieć rezerwuarowa, SA, DA, NODE i UDE |
| Kacper Bieniasz | 7–8 | PINN-y, supermodel z modeli ODE, SuperNet; miara pokrycia; pomoc w solverach i SA; opieka nad repo |
| Razem | 1, prezentacje, tabela końcowa | Prezentacje co dwa tygodnie, zakresy parametrów, porównanie wszystkich modeli |

Issue mają etykietę `osoba: …`; etykieta `magisterka` oznacza wyniki, z których Kacper skorzysta w pracy magisterskiej (z podaniem autorstwa).

## Jak pracujemy

- Gałąź na każde issue (`zad2/solver-b`), PR do `main` z „Closes #N”, review przez drugą osobę.
- Przed PR: `uv run pytest`, a jeśli zmieniły się notatniki, `uv run reproduce --only <nazwa>`.
- Raz w tygodniu 30 minut: co zrobione, co blokuje, przegląd tablicy.
- Laboratoria są indywidualne, więc ich kodu tu nie trzymamy.
