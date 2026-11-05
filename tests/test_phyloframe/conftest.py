"""Configuration file for pytest."""

import random

import numpy as np
import numpy.random
import pandas  # noqa: F401
import polars  # noqa: F401
import pytest

# pre-import to avoid xdist controller import deadlock between worker threads


@pytest.fixture(autouse=True)
def reseed_random(request: pytest.FixtureRequest) -> None:
    """Reseed random number generator to ensure determinstic test."""
    random.seed(request.node.name)
    np.random.seed(random.randint(1, 2**32))
