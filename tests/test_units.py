"""Testy przeliczenia jednostek i zakresów parametrów. Uruchomienie: uv run pytest"""
import numpy as np
import pytest

from offtarget.units import REFERENCE, Scales, clinical_schedule, log_kill_per_fraction, model_ranges


def test_reference_scales_match_swanson():
    """rho0 = 0,012/dzień i D0/rho0 = 9,83 mm^2 dają skale ok. 3,1 mm i 83 dni, a front 2 sqrt(D0 rho0)
    to mediana prędkości z Swanson i in. 2008 (0,075 mm/dzień). Swanson liczy ją tym samym wzorem Fishera,
    więc to sprawdzenie stałych, a nie niezależny pomiar."""
    assert abs(REFERENCE.length_mm - np.sqrt(9.83)) < 1e-12
    assert abs(REFERENCE.time_days - 1 / 0.012) < 1e-9
    assert abs(2 * np.sqrt(REFERENCE.D * REFERENCE.rho) - 0.075) < 5e-4


def test_reference_patient_is_one_in_model_units():
    assert REFERENCE.model_D(REFERENCE.D) == pytest.approx(1.0)
    assert REFERENCE.model_rho(REFERENCE.rho) == pytest.approx(1.0)
    other = Scales(D=0.5, rho=0.02)
    assert other.length(other.length_mm) == pytest.approx(1.0)
    assert other.time(other.time_days) == pytest.approx(1.0)


def test_log_kill_per_fraction_matches_lq():
    assert log_kill_per_fraction(2.0) == pytest.approx(0.12)               # TumorTwin: 0,05·2 + 0,005·4
    assert log_kill_per_fraction(2.0, alpha=0.025) == pytest.approx(0.06)  # start kalibracji LUMIERE
    assert log_kill_per_fraction(0.0) == 0.0


def test_model_ranges_contain_reference_and_follow_sources():
    ranges = model_ranges()
    assert ranges["D"] == pytest.approx((0.013 / REFERENCE.D, 3.8 / REFERENCE.D))
    assert ranges["rho"] == pytest.approx((0.0027 / 0.012, 0.19 / 0.012))
    assert ranges["beta"] == pytest.approx((0.06, 0.12))
    for lo, hi in (ranges["D"], ranges["rho"]):
        assert lo < 1.0 < hi  # pacjent odniesienia leży w środku zakresów


def test_clinical_schedule_weekdays_and_narrow_pulses():
    """30 frakcji od poniedziałku do piątku: odstępy 1 dzień, a co piątą frakcję 3 dni (weekend);
    ostatnia w dniu 39. Impuls jest dużo węższy niż odstęp, więc frakcje się nie nakładają."""
    sch = clinical_schedule()
    days = np.array(sch.times) / REFERENCE.rho
    gaps = np.round(np.diff(days), 9)
    assert len(sch.times) == 30
    assert days[0] == 0.0 and days[-1] == pytest.approx(39.0)
    assert set(gaps) == {1.0, 3.0} and np.sum(gaps == 3.0) == 5
    assert sch.width < 0.1 * REFERENCE.time(1.0)
    assert sch.integral(-1.0, REFERENCE.time(50.0)) == pytest.approx(30.0)
