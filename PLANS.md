# Plan projektu tydzień po tygodniu

Każdy wiersz to issue na GitHubie; wszystkie są też na [tablicy projektu](https://github.com/users/Redy144/projects/2).
Tydzień przy issue to tydzień, w którym zaczynamy pracę; termin oddania wyznacza kamień milowy, czyli najbliższa prezentacja.

## Terminy prezentacji (piątki)

| Data | Zadanie prowadzącego | Prowadzi |
| --- | --- | --- |
| 23.10 | 1. Literatura, model PDE i jego zachowanie (zrobione, opis w `report/README.md`) | razem |
| 6.11 | 2. Solvery i rozdzielczości, skrajne parametry, warunki brzegowe, uproszczenia członów | Krzysiek |
| 20.11 | 3. Surogat ODE dla masy guza i sieć rezerwuarowa | Krzysiek |
| 4.12 | 4. SA (Morris, Sobol), PDE_simple i ODE_simple; 5. DA (ABC, 4D-Var) | Krzysiek, Kacper pomaga w SA |
| 18.12 | 6. NODE i UDE na ODE wobec samej DA | Krzysiek |
| 15.01 | 7–8. PINN, supermodel z modeli ODE, SuperNet z PINN-ów, porównanie | Kacper, tabela razem |

Na każdej prezentacji Kacper krótko pokazuje postęp z PINN-ami, żeby Zadania 7–8 nie pojawiły się dopiero w styczniu.

## Ścieżka krytyczna

```
solver i generator danych (Zad. 2) → SA → PDE_simple i ODE_simple (Zad. 4) → DA (Zad. 5) → NODE/UDE (Zad. 6)
                                                                                 └→ rodzina PINN-ów → SuperNet (Zad. 8)
PINN z terapią (Zad. 7) potrzebuje tylko solvera, więc idzie równolegle od listopada.
```

## Tydzień 1: 2–8.11 · Organizacja i minimum Zadania 2

Repo, środowisko i zasady pracy. Na prezentację 6.11 (pt): solver niejawny, wariant ze skokiem LQ, testy, tabela błąd a czas i trzy scenariusze.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#1](https://github.com/Redy144/off-target/issues/1) Setup repo | Organizacja | Kacper |
| [#2](https://github.com/Redy144/off-target/issues/2) Uruchomić środowisko u siebie (uv, Jupyter, testy, reproduce) | Organizacja | Krzysiek |
| [#3](https://github.com/Redy144/off-target/issues/3) Ustalić zasady pracy i cotygodniowe spotkanie | Organizacja | razem |
| [#4](https://github.com/Redy144/off-target/issues/4) Solver B: schemat niejawny (Crank-Nicolson/IMEX albo BDF) | Zad. 2 | Krzysiek |
| [#5](https://github.com/Redy144/off-target/issues/5) Solver C: dawka jako natychmiastowy skok LQ | Zad. 2 | Krzysiek |
| [#6](https://github.com/Redy144/off-target/issues/6) Testy poprawności i zgodności solverów A, B, C | Zad. 2 | Krzysiek |
| [#7](https://github.com/Redy144/off-target/issues/7) Rozdzielczość i czas obliczeń: tabela błąd a czas | Zad. 2 | Krzysiek |
| [#8](https://github.com/Redy144/off-target/issues/8) Scenariusze z tematu: pole pełne, za wąskie, przesunięte | Zad. 2 | Krzysiek |
| [#9](https://github.com/Redy144/off-target/issues/9) Prezentacja 6.11: Zadanie 2 | Zad. 2 | razem |

## Tydzień 2: 9–15.11 · Reszta Zadania 2, start Zadania 3

Zakresy parametrów z literatury, warunki brzegowe, skrajne parametry, uproszczenia członów, mapa niedopasowania i generator danych. Start ODE dla masy guza. Kacper: miara pokrycia η i PINN z terapią. Laboratorium 13.11 (SPH).

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#10](https://github.com/Redy144/off-target/issues/10) Zakresy parametrów z literatury i przeliczenie jednostek | Zad. 2 | razem |
| [#11](https://github.com/Redy144/off-target/issues/11) Warunki brzegowe i początkowe | Zad. 2 | Krzysiek |
| [#12](https://github.com/Redy144/off-target/issues/12) Skrajne parametry | Zad. 2 | Krzysiek |
| [#13](https://github.com/Redy144/off-target/issues/13) Uproszczenia członów i porównanie z modelem bazowym | Zad. 2 | Krzysiek |
| [#14](https://github.com/Redy144/off-target/issues/14) Mapa niedopasowania w płaszczyźnie (r, s) | Zad. 2 | Krzysiek |
| [#15](https://github.com/Redy144/off-target/issues/15) Generator danych do Zadań 3–8 | Zad. 2 | Krzysiek |
| [#16](https://github.com/Redy144/off-target/issues/16) Miara pokrycia η w observables | Zad. 2 | Kacper |
| [#17](https://github.com/Redy144/off-target/issues/17) Literatura do Zadania 3: sieci rezerwuarowe, Galerkin/POD | Zad. 3 | Krzysiek |
| [#18](https://github.com/Redy144/off-target/issues/18) ODE dla masy guza: wyprowadzenie i model jednoprzedziałowy | Zad. 3 | Krzysiek |
| [#24](https://github.com/Redy144/off-target/issues/24) PINN dla modelu z terapią (wersja podstawowa) | Zad. 7 | Kacper |

## Tydzień 3: 16–22.11 · Zadanie 3, prezentacja 20.11

ODE z dwoma przedziałami, ręczny dobór parametrów, sieć rezerwuarowa, opcjonalnie POD. Czytamy literaturę do SA i DA.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#19](https://github.com/Redy144/off-target/issues/19) ODE_2: dwa przedziały (w polu i poza polem) | Zad. 3 | Krzysiek |
| [#20](https://github.com/Redy144/off-target/issues/20) Ręczny dobór parametrów ODE i porównanie z PDE | Zad. 3 | Krzysiek |
| [#21](https://github.com/Redy144/off-target/issues/21) Sieć rezerwuarowa (ReservoirPy) w oknie uczącym | Zad. 3 | Krzysiek |
| [#22](https://github.com/Redy144/off-target/issues/22) Opcja: Galerkin albo POD z migawek PDE | Zad. 3 | Krzysiek |
| [#23](https://github.com/Redy144/off-target/issues/23) Prezentacja 20.11: Zadanie 3 i reszta Zadania 2 | Zad. 3 | razem |
| [#26](https://github.com/Redy144/off-target/issues/26) Literatura do Zadań 4–5: Morris, Sobol, ABC, 4D-Var | Zad. 4, Zad. 5 | Krzysiek |

## Tydzień 4: 23–29.11 · Analiza wrażliwości

Konfiguracja SA, Morris i Sobol na PDE i ODE, dane syntetyczne do DA. Kacper: PINN logitowy z terapią. Laboratorium 27.11 (SA/DA).

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#25](https://github.com/Redy144/off-target/issues/25) PINN logitowy z terapią | Zad. 7 | Kacper |
| [#27](https://github.com/Redy144/off-target/issues/27) SA: wejścia, zakresy i wyjścia | Zad. 4 | razem |
| [#28](https://github.com/Redy144/off-target/issues/28) SA metodą Morrisa na PDE i ODE | Zad. 4 | Krzysiek |
| [#29](https://github.com/Redy144/off-target/issues/29) SA indeksami Sobola na PDE i ODE | Zad. 4 | Krzysiek |
| [#31](https://github.com/Redy144/off-target/issues/31) Dane syntetyczne do asymilacji | Zad. 5 | Krzysiek |

## Tydzień 5: 30.11–6.12 · Asymilacja danych, prezentacja 4.12

PDE_simple i ODE_simple, ABC, 4D-Var, eksperymenty DA, state of research o SA. Kacper: PINN a solver.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#30](https://github.com/Redy144/off-target/issues/30) PDE_simple i ODE_simple | Zad. 4 | Krzysiek |
| [#32](https://github.com/Redy144/off-target/issues/32) DA metodą ABC | Zad. 5 | Krzysiek |
| [#33](https://github.com/Redy144/off-target/issues/33) DA metodą 4D-Var z silnym więzem | Zad. 5 | Krzysiek |
| [#34](https://github.com/Redy144/off-target/issues/34) Eksperymenty DA: prognoza w przód i w tył, liczba próbek, budżety | Zad. 5 | Krzysiek |
| [#35](https://github.com/Redy144/off-target/issues/35) Prezentacja 4.12: Zadania 4–5 i state of research (SA) | Zad. 4, Zad. 5 | razem |
| [#36](https://github.com/Redy144/off-target/issues/36) PINN a solver: dokładność i koszt | Zad. 7 | Kacper |

## Tydzień 6: 7–13.12 · NODE

Dane do NODE i UDE, NODE. Kacper: PINN w zadaniu odwrotnym. Laboratorium 11.12 (NODE/UDE).

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#37](https://github.com/Redy144/off-target/issues/37) Dane do NODE i UDE | Zad. 6 | Krzysiek |
| [#38](https://github.com/Redy144/off-target/issues/38) NODE: cała prawa strona siecią | Zad. 6 | Krzysiek |
| [#42](https://github.com/Redy144/off-target/issues/42) PINN w zadaniu odwrotnym (nieznane D, ρ, β) | Zad. 7 | Kacper |

## Tydzień 7: 14–20.12 · UDE, prezentacja 18.12

UDE i porównanie z NODE i samą DA.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#39](https://github.com/Redy144/off-target/issues/39) UDE: znane ODE plus sieć na domknięcie | Zad. 6 | Krzysiek |
| [#40](https://github.com/Redy144/off-target/issues/40) Porównanie NODE, UDE i samej DA | Zad. 6 | Krzysiek |
| [#41](https://github.com/Redy144/off-target/issues/41) Prezentacja 18.12: Zadanie 6 | Zad. 6 | razem |

## Tydzień 8: 21–27.12 · Przerwa świąteczna: supermodel ODE i rodzina PINN-ów

25.12 bez zajęć. Kacper: supermodel z modeli ODE i rodzina PINN-ów.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#43](https://github.com/Redy144/off-target/issues/43) Supermodel z modeli ODE (ważenie prawych stron) | Zad. 8 | Kacper |
| [#44](https://github.com/Redy144/off-target/issues/44) Rodzina PINN-ów dla parametrów z SA i DA | Zad. 8 | Kacper |

## Tydzień 9: 28.12–3.01 · SuperNet

1.01 bez zajęć. Kacper: SuperNet w trzech wariantach.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#45](https://github.com/Redy144/off-target/issues/45) SuperNet: wspólna warstwa nad zamrożonymi PINN-ami | Zad. 8 | Kacper |

## Tydzień 10: 4–10.01 · Test trudny i tabela końcowa

Nowy plan wiązki i prawda spoza klasy modeli; tabela wszystkich modeli. Laboratorium 8.01 (supermodele).

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#46](https://github.com/Redy144/off-target/issues/46) Test trudny: nowy plan wiązki i prawda spoza klasy modeli | Zad. 8 | Kacper |
| [#47](https://github.com/Redy144/off-target/issues/47) Tabela końcowa wszystkich modeli | Zad. 8 | razem |

## Tydzień 11: 11–17.01 · Prezentacja 15.01

Zadania 7–8 i state of research.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#48](https://github.com/Redy144/off-target/issues/48) Prezentacja 15.01: Zadania 7–8 i state of research | Zad. 8 | razem |

## Tydzień 12: 18–24.01 · Zamknięcie

Odtworzenie wyników z czystego klonu. Wykład o PINN-ach i supermodelingu 21.01.

| Issue | Zadanie | Kto |
| --- | --- | --- |
| [#49](https://github.com/Redy144/off-target/issues/49) Odtworzenie wszystkich wyników z czystego klonu | Organizacja | razem |

## Ryzyka

- **Obciążenie Krzyśka w tygodniu 2:** siedem issue z Zadań 2 i 3 naraz. Kacper bierze miarę pokrycia, a zakresy parametrów robimy razem; jeśli to za mało, Kacper przejmuje warunki brzegowe albo generator danych.
- **Zadanie 2 jest 6.11, dzień po założeniu repo.** Na prezentację idzie minimum z tygodnia 1, resztę pokazujemy 20.11.
- **PINN z frakcjonowaną radioterapią:** w literaturze nie znaleźliśmy gotowej recepty; zaczynamy od impulsów gładkich, skok LQ porównujemy później.
