"""Configuration file for pytest."""

import random

import numpy as np
import numpy.random  # pre-imports prevent worker deadlock
import pandas  # noqa: F401
import polars  # noqa: F401
import pytest


@pytest.fixture(autouse=True)
def reseed_random(request: pytest.FixtureRequest) -> None:
    """Reseed random number generator to ensure determinstic test."""
    random.seed(request.node.name)
    np.random.seed(random.randint(1, 2**32))
