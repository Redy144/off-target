# Projekt ISZ: opis do prezentacji 23.10

Stan na 2026-10-23, dzień prezentacji Zadania 1. Opis dla nas obu: o czym jest projekt, co zrobimy w kolejnych zadaniach i skąd pochodzi każda informacja ze slajdów. Projekt dopiero zaczynamy, więc na 23.10 nie mamy jeszcze żadnych wyników.

## Projekt w skrócie

Temat z listy prowadzącego: „Radioterapia — wpływ niedopasowania pola promieniowania do guza”. Guz opisujemy równaniem dyfuzji-reakcji z członem radioterapii:

```latex
\frac{\partial u}{\partial t} = D\,\nabla^2 u + \rho\, u\left(1-\frac{u}{K}\right) - \beta\, R(x)\, U(t)\, u
```

- u(x, t) to gęstość komórek guza w miejscu x i chwili t.
- D∇²u: komórki wędrują do sąsiedniej tkanki (naciekanie).
- ρu(1 − u/K): komórki się dzielą, aż wypełnią dostępne miejsce K.
- βR(x)U(t)u: wiązka zabija komórki. R(x) mówi, gdzie jest pole wiązki, U(t), kiedy podajemy dawki, a β, jak wrażliwe są komórki.

Pytanie projektu: co się dzieje z guzem, gdy pole napromieniania nie pokrywa go dokładnie, i czy da się to szybko przewidzieć. Główna wielkość, którą mierzymy, to masa guza M(t) = ∫u dx.

Projekt idzie sześcioma terminami prowadzącego: 23.10, 6.11, 20.11, 4.12, 18.12 i 15.01. Krzysiek prowadzi Zadania 2–6, Kacper Zadania 7–8. Liczymy w 1D; 2D tylko, jeśli starczy czasu.

## Co mamy na 23.10, a co przed nami

| Element | Stan na 23.10 | Termin |
| --- | --- | --- |
| Temat, model i jego parametry | Wybrane i opisane na slajdach 2–5 | Zadanie 1, 23.10 |
| Przegląd literatury | Cztery prace na slajdzie 3 | Zadanie 1, 23.10 |
| Solver równania 1D | Do napisania | Zadanie 2, 6.11 |
| Model ODE dla masy guza, sieć rezerwuarowa | Do zrobienia | Zadanie 3, 20.11 |
| Analiza wrażliwości, asymilacja danych | Do zrobienia | Zadania 4–5, 4.12 |
| NODE i UDE | Do zrobienia | Zadanie 6, 18.12 |
| PINN-y, supermodel z modeli ODE, SuperNet z PINN-ów | Do zrobienia | Zadania 7–8, 15.01 |

## Slajd po slajdzie: skąd są informacje

| Slajd | Co pokazuje | Skąd |
| --- | --- | --- |
| 1. Tytuł | Izodozy wiązki przesunięte względem konturu guza | Rysunek poglądowy, nie dane |
| 2. Problem | MRI nie pokazuje komórek na obrzeżach guza | GliODIL (Balcerak i in. 2025, Nature Communications 16:5982): model odtwarza komórki także poza granicą widoczną w MRI |
| 2. Problem | Dopasowanie modelu 3D do pacjenta trwa dziesiątki minut | GliODIL, s. 14: 45 ± 20 min na pacjenta na GPU |
| 3. Literatura | Cztery prace i co z nich bierzemy | Notatka z przeglądu artykułów; szczegóły niżej |
| 4. Model | Równanie i RYS. 1 | Postać równania z literatury; RYS. 1 to schemat narysowany ręcznie, nie symulacja (opis niżej) |
| 5. Parametry | Pokrętła modelu i czego się spodziewamy | Oczekiwania wynikają z budowy równania; sprawdzimy je w Zadaniach 2 i 4 |
| 6. Plan | Daty i zadania | Prezentacja prowadzącego `Labs_Projects_2026.pdf`, s. 4–6 i 8–9 |
| 7. SuperNet | Zamrożone PINN-y i jedna uczona warstwa W | Notatka robocza prof. Dzwinela (rys. 1–2) i slajd prowadzącego (s. 25) |
| 8. Koniec | Odpowiedzi na pytania w notatce mówcy | Patrz „O co mogą zapytać” |

Prace ze slajdu 3:

- **Rockne i in., J. Math. Biol. 58:** model glejaka z radioterapią, stąd człon promieniowania. Pozycja z bibliografii artykułu Szuberta i in.; rok 2009 podany z pamięci, niesprawdzony.
- **Unkelbach i in. 2014, Phys. Med. Biol. 59, 747:** pole napromieniania wyznaczane z modelu wzrostu guza.
- **Balcerak i in. 2025 (GliODIL):** plan z modelu obejmuje 67,8% późniejszego nawrotu, standardowy margines 15 mm 63,6% (rys. 5d, 152 pacjentów). Liczby są tylko w notatce mówcy.
- **Szubert, Bialy, Dzwinel 2026, wersja robocza (AGH):** to samo równanie w 2D, ale z terapią jednorodną w przestrzeni, bez profilu wiązki R(x) (eq. 1–4). Dwa modele zastępcze: sieć VAE (100 symulacji w 0,77 s wobec 17 s solvera, tab. 4) i PINN osobny dla każdego scenariusza (sec. 7–9).

## RYS. 1 na slajdzie 4: schemat, nie symulacja

Rysunek jest jednowymiarowy: oś pozioma to położenie x, zielona krzywa to gęstość komórek u(x), żółty prostokąt to pole wiązki R(x), a czerwony kolor oznacza komórki poza polem. Pokazuje trzy sytuacje, które chcemy zbadać:

| Sytuacja | Co widać | Czego się spodziewamy |
| --- | --- | --- |
| Pole pokrywa guz | Całe u(x) leży w polu | Przy dużej dawce guz znika |
| Pole za wąskie | Oba brzegi guza są poza polem | Guz odrasta z obu brzegów |
| Pole przesunięte | Jedna strona guza jest poza polem | Guz odrasta po tej stronie |

Skąd te oczekiwania: poza polem R(x) = 0, więc człon radioterapii tam znika. Komórki, które tam zostały, dalej się dzielą (człon wzrostu) i wędrują z powrotem w leczony obszar (człon dyfuzji). Czy tak jest naprawdę i jak szybko guz odrasta, sprawdzimy solverem w Zadaniu 2.

## Co sprawdzimy w kolejnych zadaniach

- **Zadanie 2, 6.11 (solvery):** piszemy solver 1D metodą różnic skończonych, z dwoma schematami czasowymi. Porównujemy kilka siatek i czasy obliczeń, sprawdzamy skrajne parametry i uproszczenia członów, np. bez dyfuzji. Test poprawności: bez leczenia front guza powinien biec z prędkością 2√(Dρ); to wynik teoretyczny dla równania Fishera-KPP, podany z pamięci, do sprawdzenia w literaturze. Potem liczymy trzy sytuacje z RYS. 1.
- **Zadanie 3, 20.11 (model ODE):** proste równanie różniczkowe zwyczajne dla samej masy guza M(t) z członem dawki i sieć rezerwuarowa uczona na danych z solvera.
- **Zadania 4–5, 4.12:** analiza wrażliwości metodami Morrisa i Sobola mówi, które parametry naprawdę wpływają na M(t); z niej modele uproszczone. Asymilacja danych dwiema metodami, ABC i 4D-Var: dopasowujemy parametry do danych.
- **Zadanie 6, 18.12:** sieci neuronowe wbudowane w model ODE, czyli NODE i UDE.
- **Zadania 7–8, 15.01:** PINN dla równania z radioterapią, supermodel z modeli ODE i SuperNet z PINN-ów. Na koniec porównanie wszystkich modeli.

Dane do Zadań 4–8 będą syntetyczne: z naszego solvera, w kilku chwilach, z szumem, tak jak kolejne badania MRI.

## O co mogą zapytać

- **Czy macie już wyniki?** Nie, projekt dopiero zaczynamy. RYS. 1 to schemat, nie symulacja. Pierwsze wyniki pokażemy 6.11.
- **Skąd dane?** Na początek z naszego solvera: kilka chwil pomiaru z szumem, jak kolejne badania MRI.
- **Dlaczego 1D?** To najprostszy przypadek, w którym widać skutek niedopasowania pola, a obliczenia są szybkie. 2D zrobimy, jeśli starczy czasu.
- **Jak działa jedna dawka w modelu?** Dawka to krótki impuls w U(t). Jeśli impuls ma całkę 1, w polu wiązki przeżywa ułamek e^(−β) komórek (S = exp(−∫βU dt)). Szubert i in. zapisują terapię podobnie, jako gładkie impulsy (eq. 1–4).
- **Czym jest PINN?** Sieć neuronowa, która uczy się rozwiązania równania; w jej funkcji straty jest samo równanie modelu.
- **Czym jest SuperNet?** Kilka PINN-ów uczonych dla różnych parametrów, połączonych jedną wspólną warstwą W. Dla nowego pacjenta PINN-y są zamrożone, a uczymy tylko W.
- **Czy to nie jest zwykłe dopasowanie parametrów?** Jeśli guz zachowuje się dokładnie jak model, to w praktyce tak. Zysk ma się pojawić, gdy guz od modelu odbiega, i w czasie: uczymy jedną warstwę, zamiast wiele razy uruchamiać solver.
- **Po co PINN, skoro jest solver?** W 1D solver będzie pewnie szybszy. Sieci uczy się jednak raz, a dla pacjenta douczamy tylko jedną warstwę; to ma się opłacić w większych modelach.
