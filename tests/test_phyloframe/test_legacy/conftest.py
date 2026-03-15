"""Configuration for legacy test suite."""

import polars as pl
import pytest


@pytest.fixture
def engine_affinity(request):
    """Set Polars engine affinity for the duration of a test, then restore."""
    affinity = request.param
    original = pl.Config.state().get("POLARS_ENGINE_AFFINITY", None)
    pl.Config.set_engine_affinity(affinity)
    yield affinity
    pl.Config.set_engine_affinity(original)
