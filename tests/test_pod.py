import numpy as np
import pytest

from offtarget.model import NO_TREATMENT, Params, Schedule, grid, initial_condition
from offtarget.observables import tumor_mass
from offtarget.pod import GalerkinDivergenceError, build_galerkin, fit_pod, simulate_pod_galerkin
from offtarget.solvers import simulate


def test_pod_basis_orthonormal_and_eckart_young():
    x, _ = grid(80.0, 200)
    t = np.linspace(0.0, 6.0, 61)
    p = Params(beta=1.5, s=6.0, r=8.0)
    sched = Schedule.regular(n=6, start=1.0, interval=0.5)
    U = simulate(p, sched, x, t, method="cn")

    basis = fit_pod(U, x, max_modes=12)
    gram = basis.modes.T @ (basis.weights[:, None] * basis.modes)
    np.testing.assert_allclose(gram, np.eye(basis.n_modes), atol=1e-12)

    # Twierdzenie Eckarta–Younga: błąd projekcji w ważonej normie Frobeniusa == suma kwadratów ogona sigma
    sqrt_w = np.sqrt(basis.weights)
    for k in (1, 3, 6):
        bk = basis.truncate(k)
        U_proj = bk.reconstruct(bk.project(U))
        err2 = float(np.sum(((U - U_proj) * sqrt_w[None, :]) ** 2))
        tail2 = float(np.sum(basis.singular_values[k:] ** 2))
        assert err2 == pytest.approx(tail2, rel=1e-9, abs=1e-12)


def test_galerkin_diffusion_is_symmetric_negative_semidefinite():
    x, _ = grid(60.0, 150)
    t = np.linspace(0.0, 4.0, 41)
    p = Params(beta=1.0, s=4.0, r=8.0)
    U = simulate(p, Schedule.regular(4, 1.0, 0.5), x, t, method="cn")
    basis = fit_pod(U, x, max_modes=10)
    ops = build_galerkin(basis, p)

    np.testing.assert_allclose(ops.D_mat, ops.D_mat.T, atol=1e-13)
    eigvals = np.linalg.eigvalsh(ops.D_mat)
    assert eigvals.max() <= 1e-12


def test_pod_galerkin_converges_to_pde_mass_and_field():
    x, _ = grid(80.0, 200)
    t = np.round(np.linspace(0.0, 10.0, 201), 10)
    u0 = initial_condition(x)
    p = Params(beta=1.5, s=6.0, r=8.0)
    sched = Schedule.regular(n=10, start=1.0, interval=0.4)
    U_pde = simulate(p, sched, x, t, method="cn")
    M_pde = tumor_mass(U_pde, x)

    basis = fit_pod(U_pde, x, max_modes=16)

    a2, uh2 = simulate_pod_galerkin(u0, basis, p, sched, t, k=2)
    a15, uh15 = simulate_pod_galerkin(u0, basis, p, sched, t, k=15)
    M2 = basis.truncate(2).mass(a2)
    M15 = basis.truncate(15).mass(a15)

    np.testing.assert_allclose(M15, tumor_mass(uh15, x), rtol=1e-12, atol=1e-12)
    err2 = float(np.linalg.norm(M2 - M_pde) / np.linalg.norm(M_pde))
    err15 = float(np.linalg.norm(M15 - M_pde) / np.linalg.norm(M_pde))
    err_u15 = float(np.linalg.norm(uh15 - U_pde) / np.linalg.norm(U_pde))

    assert err15 < err2
    assert err15 < 0.005
    assert err_u15 < 0.005


def test_pod_galerkin_detects_blowup_gracefully():
    x, _ = grid(80.0, 200)
    t = np.round(np.linspace(0.0, 10.0, 201), 10)
    u0 = initial_condition(x)
    p = Params(beta=1.5, s=6.0, r=8.0)
    sched = Schedule.regular(n=10, start=1.0, interval=0.4)
    U_pde = simulate(p, sched, x, t, method="cn")
    basis = fit_pod(U_pde, x, max_modes=8)

    with pytest.raises(GalerkinDivergenceError):
        simulate_pod_galerkin(u0, basis, p, sched, t, k=5)


def test_pod_galerkin_pure_diffusion_matches_pde():
    x, _ = grid(40.0, 100)
    t = np.linspace(0.0, 3.0, 31)
    u0 = initial_condition(x)
    p = Params(D=1.0, rho=0.0, beta=0.0)
    U_pde = simulate(p, NO_TREATMENT, x, t, method="cn")
    basis = fit_pod(U_pde, x, max_modes=6)
    a_traj, _ = simulate_pod_galerkin(u0, basis, p, NO_TREATMENT, t, k=6)
    M_pod = basis.mass(a_traj)
    M_pde = tumor_mass(U_pde, x)
    np.testing.assert_allclose(M_pod, M_pde, rtol=1e-3)
