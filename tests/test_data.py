import numpy as np
import pytest

from offtarget.data import load_dataset, make_dataset, sample_scenarios
from offtarget.model import grid


def test_make_dataset_saves_and_reloads(tmp_path):
    rng = np.random.default_rng(123)
    params_list, schedules = sample_scenarios(3, rng, split="train")
    x, _ = grid(40.0, 100)
    t = np.linspace(0.0, 2.0, 21)
    paths = make_dataset(params_list, schedules, x, t, out_dir=tmp_path, prefix="train", dt_max=0.05)
    assert len(paths) == 3
    assert all(p.exists() for p in paths)
    runs = load_dataset(tmp_path, "train_*.npz")
    assert len(runs) == 3
    for run, p, s in zip(runs, params_list, schedules):
        assert run["params"] == p
        assert run["schedule"] == s
        assert run["u"].shape == (len(t), len(x))
        assert run["M"].shape == (len(t),)
        assert np.all(np.isfinite(run["u"]))


def test_make_dataset_length_mismatch(tmp_path):
    rng = np.random.default_rng(1)
    params_list, schedules = sample_scenarios(2, rng, split="train")
    x, _ = grid(20.0, 40)
    t = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        make_dataset(params_list, schedules[:1], x, t, out_dir=tmp_path)


def test_train_and_test_splits_are_disjoint_and_deterministic():
    p_tr1, s_tr1 = sample_scenarios(15, np.random.default_rng(2026), split="train")
    p_tr2, s_tr2 = sample_scenarios(15, np.random.default_rng(2026), split="train")
    assert p_tr1 == p_tr2 and s_tr1 == s_tr2

    p_te, s_te = sample_scenarios(18, np.random.default_rng(2027), split="test")
    for p in p_tr1:
        assert 3.0 <= p.r <= 16.0 and 0.0 <= p.s <= 8.0
    for p, sched in zip(p_te, s_te):
        assert not (3.0 <= p.r <= 16.0 and 0.0 <= p.s <= 8.0)
        diffs = np.diff(sched.times)
        n = len(sched.times)
        regular_in_train = (6 <= n <= 12) and np.allclose(diffs, diffs[0]) and (0.35 <= diffs[0] <= 0.55)
        assert not regular_in_train
