import functools

import numpy as np
import pytest

from phyloframe._auxlib._run_seeded_replicates import run_seeded_replicates


def _draw(*, seed: int, scale: float = 1.0) -> float:
    return scale * np.random.default_rng(seed).random()


@pytest.mark.parametrize("parallel_backend", [None, "threading", "loky"])
def test_reproducible_across_backends(parallel_backend):
    expected = run_seeded_replicates(_draw, num_replicates=8, seed=1)
    res = run_seeded_replicates(
        _draw, num_replicates=8, seed=1, parallel_backend=parallel_backend
    )
    assert res == expected
    assert len(set(res)) == 8  # replicates get distinct seeds


def test_seed_changes_results():
    a = run_seeded_replicates(_draw, num_replicates=3, seed=1)
    b = run_seeded_replicates(_draw, num_replicates=3, seed=2)
    assert a != b


def test_unseeded():
    np.random.seed(1)
    a = run_seeded_replicates(_draw, num_replicates=3)
    np.random.seed(1)
    b = run_seeded_replicates(_draw, num_replicates=3)
    assert a == b


def test_partial_and_progress_wrap():
    calls = []

    def wrap(jobs):
        jobs = list(jobs)
        calls.append(len(jobs))
        return jobs

    res = run_seeded_replicates(
        functools.partial(_draw, scale=0.0),
        num_replicates=4,
        seed=1,
        progress_wrap=wrap,
    )
    assert res == [0.0] * 4
    assert calls == [4]


def test_zero_replicates():
    assert run_seeded_replicates(_draw, num_replicates=0, seed=1) == []


@pytest.mark.parametrize("parallel_backend", [None, "threading"])
def test_tqdm_progress_wrap(parallel_backend):
    # tqdm needs a sized iterable, which joblib also queries for its length
    import tqdm

    res = run_seeded_replicates(
        lambda seed: seed,
        num_replicates=3,
        seed=1,
        parallel_backend=parallel_backend,
        progress_wrap=functools.partial(tqdm.tqdm, desc="test"),
    )
    assert res == run_seeded_replicates(
        lambda seed: seed, num_replicates=3, seed=1
    )
