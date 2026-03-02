import itertools as it

import numpy as np

from phyloframe._auxlib import seed_random
from phyloframe._auxlib._jit import jit


@jit(nopython=True)
def _generate_jitted_random_values(n: int) -> np.ndarray:
    """Generate random values from within a jitted context."""
    result = np.empty(n)
    for i in range(n):
        result[i] = np.random.random()
    return result


def test_seed_random_jitted_deterministic():
    """Regression test: seed_random must seed numba's internal PRNG so that
    jitted code produces deterministic random values."""
    n = 10
    results = []
    for _rep in range(3):
        seed_random(42)
        results.append(_generate_jitted_random_values(n))

    for a, b in it.combinations(results, 2):
        np.testing.assert_array_equal(a, b)

    # control test: different seeds produce different values
    results_different = []
    for rep in range(3):
        seed_random(rep)
        results_different.append(_generate_jitted_random_values(n))

    for a, b in it.combinations(results_different, 2):
        assert not np.array_equal(a, b)
