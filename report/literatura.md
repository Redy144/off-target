# Literatura do Zadania 3: surogaty ODE, sieci rezerwuarowe (ESN) i Galerkin/POD

Notatka zbiera źródła do budowy niskowymiarowych surogatów modelu PDE radioterapii guza
(issue #17): co dokładnie bierzemy z każdej pozycji i w których zadaniach (#18–#22) z tego korzystamy.

---

## 1. Sieci rezerwuarowe (Echo State Networks) i ReservoirPy

### 1.1. Lukoševičius (2012)
**M. Lukoševičius**, *A Practical Guide to Applying Echo State Networks*, w: G. Montavon, G. B. Orr, K.-R. Müller (red.), *Neural Networks: Tricks of the Trade*, LNCS 7700, Springer, 2012, s. 659–686 (`doi:10.1007/978-3-642-35289-8_36`).

**Co bierzemy:**
- **Równania stanu rezerwuaru z wyciekiem (*leaky-integrated units*), równ. (2)–(3):**
  $$\tilde{\mathbf{x}}(n) = \tanh\!\big(\mathbf{W}_{\text{in}}[1;\,\mathbf{u}(n)] + \mathbf{W}\mathbf{x}(n-1)\big), \qquad \mathbf{x}(n) = (1 - \alpha)\,\mathbf{x}(n-1) + \alpha\,\tilde{\mathbf{x}}(n),$$
  gdzie $\alpha \in (0, 1]$ to współczynnik wycieku (*leaking rate*), odpowiadający skali czasowej dynamiki.
- **Dobór promienia spektralnego $\rho(\mathbf{W})$ i własność echa (*echo state property*), sekcja 3.2.4:** losową rzadką macierz $\mathbf{W}$ normalizujemy przez jej największą wartość własną co do modułu i skalujemy docelowym promieniem spektralnym; zbyt duży promień niszczy własność zanikającej pamięci (*echo state property*) i destabilizuje prognozę.
- **Uczenie warstwy wyjściowej przez regresję grzbietową (Tichonowa), równ. (7)–(8) i sekcja 4.2:** jedyne uczone wagi to macierz liniowego odczytu $\mathbf{W}_{\text{out}}$ z regularyzacją $L^2$ (parametr $\beta$/`ridge` dobierany na siatce logarytmicznej), po odrzuceniu początkowego rozbiegu (*washout*).

**Zastosowanie w projekcie:** dobór hiperparametrów ESN w `zad3_02_esn.ipynb` (#21): rozmiar rezerwuaru (`units`), promień spektralny (`sr`), wyciek (`lr`) i regularyzacja `ridge`.

### 1.2. Jaeger (2001) i Pathak i in. (2018)
- **H. Jaeger**, *The "echo state" approach to analysing and training recurrent neural networks*, GMD Report 148, 2001 — praca źródłowa wprowadzająca ESN i formalny warunek *echo state property*.
- **J. Pathak, B. Hunt, M. Girvan, Z. Lu, E. Ott**, *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach*, *Physical Review Letters* 120, 024102, 2018 (`doi:10.1103/PhysRevLett.120.024102`).

**Co bierzemy:**
- **Schemat uczenia w oknie otwartym i prognozy autonomicznej (*closed-loop*):** w oknie uczącym $[0, T_{\text{train}}]$ sieć dostaje prawdziwy stan $M(t)$ i wymuszenie $U(t)$, a poza oknem predykcja $\hat{M}(t + \Delta t)$ wraca jako wejście w kolejnym kroku przy zadanym z zewnątrz harmonogramie $U(t)$.
- **Ocena horyzontu prognozy:** błąd w trybie otwartym (jednokrokowym) nie wystarcza — kluczowe jest sprawdzenie stabilności trajektorii w pętli zamkniętej oraz uogólnienia na harmonogram spoza treningu.

**Zastosowanie w projekcie:** protokół eksperymentu z oknem uczącym $[0, T_{\text{train}}]$ i prognozą autonomiczną w #21.

### 1.3. ReservoirPy (Trouvain i in., 2020; dokumentacja v0.4)
**N. Trouvain, L. Pedrelli, T. T. Dinh, X. Hinaut**, *ReservoirPy: An Efficient and User-Friendly Library to Design Echo State Networks*, ICANN 2020, Springer, s. 494–505; dokumentacja biblioteki `reservoirpy` (w projekcie v0.4.x).

**Co bierzemy:**
- **Konwencja API:** węzły `Reservoir(units, lr, sr, input_scaling, rc_connectivity, seed)` i `Ridge(ridge)` łączone potokiem `reservoir >> readout`.
- **Aktualizacja w `Reservoir`:** $\mathbf{x}[t+1] = (1 - \mathrm{lr})\mathbf{x}[t] + \mathrm{lr}\cdot\tanh(\mathbf{W}_{\text{in}}\mathbf{u}[t+1] + \mathbf{W}\mathbf{x}[t])$, krok po kroku przez `node.run` (uczenie z `warmup`) i `node.call` (pętla autonomiczna z podawaniem zewnętrznego $U(t)$).
- **Powtarzalność:** przekazywanie deterministycznego ziarna z `seed_everything()` (`SEED = 2026`).

---

## 2. Redukcja wymiaru PDE → ODE: Galerkin i POD

### 2.1. Materiały prowadzącego (W. Dzwinel, `Labs_Projects_2026.pdf`, s. 12–20)
**Co bierzemy:**
- **4-krokowy schemat redukcji (s. 12–15, 18):**
  1. Rozwinięcie pola w bazie niezależnej od czasu: $u(x, t) = \sum_{i=1}^N y_i(t)\,\phi_i(x)$ (s. 13).
  2. Podstawienie do PDE i rzut ortogonalny (Galerkina); przy bazie ortonormalnej $\int_\Omega \phi_i(x)\phi_j(x)\,dx = \delta_{ij}$ macierz masy upraszcza się do $M = I$, dając jawny układ ODE $\dot{\mathbf{y}} = \mathbf{F}(\mathbf{y})$ (s. 14).
  3. Obcięcie do $r \ll N$ dominujących modów jako model surogatowy ODE (s. 15).
  4. Przegląd alternatyw (s. 17): POD (z danych symulacyjnych), DMD, PINN/Neural ODE.
- **Surogat kompartmentowy i okna czasowe (s. 19–20):** fizycznie interpretowalny układ kilku sprzężonych ODE (s. 19 — w naszym projekcie `ODE_1` i `ODE_2` w #18–#19) oraz podział przebiegu na okna A/B/C do oceny prognozy poza oknem uczącym (s. 20).

### 2.2. Holmes, Lumley, Berkooz (1996)
**P. Holmes, J. L. Lumley, G. Berkooz**, *Turbulence, Coherent Structures, Dynamical Systems and Symmetry*, Cambridge University Press, 1996 (cytowane na slajdzie 17; wyd. 2 z C. W. Rowleyem, 2012).

**Co bierzemy:**
- **Rozdział 3 (*Proper orthogonal decomposition*, wyd. 1 s. 86–128):** konstrukcja empirycznej bazy ortonormalnej $\{\phi_i(x)\}$ z zespołu migawek pola $u(x, t_k)$ przez SVD macierzy migawek. Baza POD jest optymalna w normie $L^2$: dla każdego $r$ maksymalizuje uchwyconą energię $\sum_{i=1}^r \sigma_i^2 / \sum_j \sigma_j^2$.
- **Rozdział 4 (*Galerkin projection*, sekcje 4.1–4.2 i 4.4):** rzutowanie PDE na podprzestrzeń rozpiętą przez $r$ modów empirycznych. Dla naszego równania
  $$\partial_t u = D\,\partial_{xx} u + \rho u\!\left(1 - \frac{u}{K}\right) - \beta R(x) U(t)\,u$$
  rzut Galerkina z bazą ortonormalną $\langle \phi_i, \phi_j \rangle = \delta_{ij}$ daje układ $r$ równań ODE o strukturze liniowo-kwadratowej z wymuszeniem:
  $$\dot{y}_i = \sum_{j=1}^r \big(D\,L_{ij} + \rho\,\delta_{ij} - \beta U(t)\,B_{ij}\big)\,y_j - \frac{\rho}{K}\sum_{j,k=1}^r Q_{ijk}\,y_j y_k,$$
  gdzie $L_{ij} = \langle \phi_i, \partial_{xx}\phi_j \rangle$, $B_{ij} = \langle \phi_i, R\phi_j \rangle$, $Q_{ijk} = \langle \phi_i, \phi_j \phi_k \rangle$, a masa guza to kombinacja liniowa $M(t) = \sum_{i=1}^r y_i(t) \int_\Omega \phi_i(x)\,dx$.
- **Uwaga z sekcji 4.4:** drastyczne obcięcie bazy empirycznej może powodować niestabilność lub błędy przy profilach spoza zbioru migawek (np. zmiana położenia wiązki $s$ lub szerokości $r$ generuje fronty nieobecne w bazie).

**Zastosowanie w projekcie:** konstrukcja i ocena zredukowanego modelu POD-Galerkin w #22.

### 2.3. Rowley (2005)
**C. W. Rowley**, *Model reduction for fluids, using balanced proper orthogonal decomposition*, *International Journal of Bifurcation and Chaos* 15(3), 2005, s. 997–1013 (`doi:10.1142/S0218127405012429`; cytowane na slajdzie 17).

**Co bierzemy:**
- **Sekcja 2:** zestawienie klasycznego POD/Galerkin z *balanced truncation* — kluczowe ograniczenie samego POD to fakt, że szereguje mody wyłącznie według energii stanu $u$, ignorując ich wpływ na obserwowane wyjście ($M(t)$) i sprzężenie z wymuszeniem ($U(t)$).
- **Sekcja 3 (3.1 *method of snapshots*, 3.2 *output projection* z oszacowaniem błędu w równ. (27)):** w układach liniowych metoda *balanced POD* pozwala równoważyć sterowalność i obserwowalność bezpośrednio z migawek (układu prostego i sprzężonego) bez liczenia pełnych macierzy Gramiana. W naszym nieliniowym PDE (#22) stosujemy klasyczny POD/Galerkin (sekcja 2), ale diagnozując liczbę potrzebnych modów, weryfikujemy osobno błąd pola $u$ i błąd obserwabli $M(t)$ oraz wrażliwość na zmianę planu wiązki.

---

## 3. Mapa źródeł → zadania w repozytorium

| Issue | Co budujemy | Źródło z notatki |
| --- | --- | --- |
| #18, #19 | `ODE_1` (1 przedział) i `ODE_2` (w polu / poza polem) z całkowania PDE | Całkowanie PDE + schemat kompartmentowy (slajdy s. 19) |
| #20 | Ręczny dobór $K_{\text{eff}}, c, \gamma$ i porównanie PDE vs `ODE_1` vs `ODE_2` | Protokół porównania na scenariuszach z #8 i mapie (r, s) z #14 |
| #21 | Sieć rezerwuarowa (ESN) w oknie uczącym + prognoza autonomiczna | Lukoševičius (2012, równ. 2–3, sek. 3.2.4, 4.2), Pathak i in. (2018), ReservoirPy |
| #22 | Redukcja POD-Galerkin z migawek PDE | Slajdy s. 12–18, Holmes–Lumley–Berkooz (1996, rozdz. 3–4), Rowley (2005, sek. 2–3) |
