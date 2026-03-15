import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import alifestd_to_working_format
from phyloframe.legacy._alifestd_find_distance_pair_asexual import (
    alifestd_find_distance_pair_asexual,
)
from phyloframe.legacy._alifestd_find_distance_pair_polars import (
    alifestd_find_distance_pair_polars as alifestd_find_distance_pair_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_find_distance_pair_polars = enforce_dtype_stability_polars(
    alifestd_find_distance_pair_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple1(apply: typing.Callable):
    # Tree:  0 -> 1 -> 2, 0 -> 3
    # origin_times: 0=0, 1=10, 2=20, 3=15
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 0],
                "origin_time": [0, 10, 20, 15],
            }
        )
    )
    # MRCA of 2 and 3 is 0; distance = (20-0) + (15-0) = 35
    assert alifestd_find_distance_pair_polars(df_pl, 2, 3) == pytest.approx(
        35.0
    )
    # MRCA of 1 and 2 is 1; distance = (10-10) + (20-10) = 10
    assert alifestd_find_distance_pair_polars(df_pl, 1, 2) == pytest.approx(
        10.0
    )
    # MRCA of a node with itself; distance = 0
    assert alifestd_find_distance_pair_polars(df_pl, 2, 2) == pytest.approx(
        0.0
    )
    assert alifestd_find_distance_pair_polars(df_pl, 0, 0) == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_no_common_ancestor_returns_none(apply: typing.Callable):
    """Disjoint trees should return None."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
                "origin_time": [0, 5, 10],
            }
        )
    )
    assert alifestd_find_distance_pair_polars(df_pl, 0, 1) is None
    assert alifestd_find_distance_pair_polars(df_pl, 1, 2) is None


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_node(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [42],
            }
        )
    )
    assert alifestd_find_distance_pair_polars(df_pl, 0, 0) == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_custom_criterion(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 0],
                "origin_time": [0, 10, 20, 15],
                "depth": [0.0, 1.0, 2.0, 1.0],
            }
        )
    )
    # MRCA of 2 and 3 is 0; depth distance = (2.0-0.0) + (1.0-0.0) = 3.0
    assert alifestd_find_distance_pair_polars(
        df_pl, 2, 3, criterion="depth"
    ) == pytest.approx(3.0)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_chain(apply: typing.Callable):
    """Straight chain: 0 -> 1 -> 2 -> 3."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 2],
                "origin_time": [0, 10, 20, 30],
            }
        )
    )
    assert alifestd_find_distance_pair_polars(df_pl, 0, 3) == pytest.approx(
        30.0
    )
    assert alifestd_find_distance_pair_polars(df_pl, 1, 3) == pytest.approx(
        20.0
    )
    assert alifestd_find_distance_pair_polars(df_pl, 2, 3) == pytest.approx(
        10.0
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_multiple_roots_partial(apply: typing.Callable):
    """Forest: tree1 = {0, 1}, tree2 = {2, 3}."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 2],
                "origin_time": [0, 5, 10, 15],
            }
        )
    )
    assert alifestd_find_distance_pair_polars(df_pl, 0, 1) == pytest.approx(
        5.0
    )
    assert alifestd_find_distance_pair_polars(df_pl, 0, 2) is None


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_fuzz_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars distance matches pandas implementation on real data."""
    df_pl = pl.from_pandas(phylogeny_df)
    ids = phylogeny_df["id"].tolist()
    for i in ids[:5]:
        for j in ids[:5]:
            expected = alifestd_find_distance_pair_asexual(
                phylogeny_df, i, j, mutate=False
            )
            actual = alifestd_find_distance_pair_polars(apply(df_pl), i, j)
            assert actual == pytest.approx(expected)
