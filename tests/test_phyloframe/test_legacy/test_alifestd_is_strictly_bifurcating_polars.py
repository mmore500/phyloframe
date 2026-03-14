import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy._alifestd_is_strictly_bifurcating_polars import (
    alifestd_is_strictly_bifurcating_polars as alifestd_is_strictly_bifurcating_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_is_strictly_bifurcating_polars = enforce_dtype_stability_polars(
    alifestd_is_strictly_bifurcating_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_bifurcating(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        )
    )
    assert alifestd_is_strictly_bifurcating_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_not_bifurcating_trifurcation(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 0],
            }
        )
    )
    assert not alifestd_is_strictly_bifurcating_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_not_bifurcating_unifurcation(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 1],
            }
        )
    )
    # root has 1 child (node 1), node 1 has 2 children => not all internal have 2
    # Actually: root (0) children are [1] (only node with ancestor_id=0 and id!=0),
    # so root has 1 child => unifurcation => not strictly bifurcating
    assert not alifestd_is_strictly_bifurcating_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_root(apply: typing.Callable):
    """A single root with no children is vacuously bifurcating."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        )
    )
    assert alifestd_is_strictly_bifurcating_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_bifurcating_asset(apply: typing.Callable):
    phylogeny_df = pd.read_csv(
        f"{assets_path}/example-standard-toy-asexual-bifurcating-phylogeny.csv"
    )
    df = apply(pl.from_pandas(phylogeny_df))
    assert alifestd_is_strictly_bifurcating_polars(df)
