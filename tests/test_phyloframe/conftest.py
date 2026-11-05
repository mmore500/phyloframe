"""Configuration file for pytest."""

import random

import numpy as np
import numpy.random
import pandas  # noqa: F401
import polars  # noqa: F401
import pytest

# Eagerly import heavy dependencies in the pytest-xdist controller process.
# The controller re-imports the module of each warning category reported by
# workers (e.g., pandas.errors) from per-worker execnet IO threads; first-time
# imports of numpy.random/pandas racing across those threads can raise
# _DeadlockError, crashing a worker and then the whole session.


@pytest.fixture(autouse=True)
def reseed_random(request: pytest.FixtureRequest) -> None:
    """Reseed random number generator to ensure determinstic test."""
    random.seed(request.node.name)
    np.random.seed(random.randint(1, 2**32))
