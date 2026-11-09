import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_consolidate_roots_asexual,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_consolidate_roots_polars import (
    alifestd_consolidate_roots_polars as alifestd_consolidate_roots_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_consolidate_roots_polars = enforce_dtype_stability_polars(
    alifestd_consolidate_roots_polars_
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_forest(num_nodes: int, seed: int) -> pl.DataFrame:
    """Random topologically sorted forest, with about a tenth of nodes as
    roots, and origin times after each node's ancestor's."""
    rng = np.random.default_rng(seed)
    ancestor_ids = np.array(
        [
            i if i == 0 or rng.random() < 0.1 else rng.integers(i)
            for i in range(num_nodes)
        ],
    )
    origin_times = np.zeros(num_nodes)
    for i in range(num_nodes):
        if ancestor_ids[i] != i:
            origin_times[i] = origin_times[ancestor_ids[i]] + rng.random()
        else:
            origin_times[i] = rng.random() * 5
    return pl.DataFrame(
        {
            "id": np.arange(num_nodes),
            "ancestor_id": ancestor_ids,
            "origin_time": origin_times,
        },
    )


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
            },
        ),
    )
    assert alifestd_consolidate_roots_polars(df_pl).lazy().collect().is_empty()


@apply_params
def test_single_root_unchanged(apply: typing.Callable):
    df_pl = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 1]})
    result = alifestd_consolidate_roots_polars(apply(df_pl)).lazy().collect()
    assert result.equals(df_pl)


@apply_params
def test_oldest_by_lowest_id(apply: typing.Callable):
    # roots 0 and 2; root 2 is deleted, so its children join root 0
    df_pl = pl.DataFrame(
        {"id": [0, 1, 2, 3, 4], "ancestor_id": [0, 0, 2, 2, 3]}
    )
    result = alifestd_consolidate_roots_polars(apply(df_pl)).lazy().collect()
    assert result["id"].to_list() == [0, 1, 3, 4]
    assert result["ancestor_id"].to_list() == [0, 0, 0, 3]


@apply_params
def test_oldest_by_lowest_origin_time(apply: typing.Callable):
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 1, 0, 1, 1, 5, 6],
            "origin_time": [3.0, 1.0, 4.0, 2.0, 5.0, 0.5, 9.0],
        },
    )
    result = alifestd_consolidate_roots_polars(apply(df_pl)).lazy().collect()
    # root 5 is oldest; roots 0, 1, and childless 6 are deleted
    assert result["id"].to_list() == [2, 3, 4, 5]
    assert result["ancestor_id"].to_list() == [5, 5, 5, 5]
    assert result["origin_time"].to_list() == [4.0, 2.0, 5.0, 0.5]


@apply_params
def test_ties_in_origin_time_break_by_id(apply: typing.Callable):
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 1, 0, 1],
            "origin_time": [-1.0, -1.0, 2.0, 3.0],
        },
    )
    result = alifestd_consolidate_roots_polars(apply(df_pl)).lazy().collect()
    assert result["id"].to_list() == [0, 2, 3]
    assert result["ancestor_id"].to_list() == [0, 0, 0]


@apply_params
def test_other_columns_preserved(apply: typing.Callable):
    df_pl = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 1, 1], "x": ["a", "b", "c"]},
    )
    result = alifestd_consolidate_roots_polars(apply(df_pl)).lazy().collect()
    assert result.columns == ["id", "ancestor_id", "x"]
    assert result["x"].to_list() == ["a", "c"]


@pytest.mark.parametrize("seed", range(5))
def test_random_forest_invariants(seed: int):
    df_pl = _make_forest(200, seed)
    result = alifestd_consolidate_roots_polars(df_pl)
    roots = df_pl.filter(pl.col("id") == pl.col("ancestor_id"))
    oldest = roots.sort("origin_time", "id")["id"][0]
    # exactly one root remains, the oldest
    is_root = result["id"] == result["ancestor_id"]
    assert result.filter(is_root)["id"].to_list() == [oldest]
    # only the other roots are deleted
    assert len(result) == len(df_pl) - (len(roots) - 1)
    # every remaining ancestor is present
    assert result["ancestor_id"].is_in(result["id"].implode()).all()
    # non-root nodes keep their ancestors, unless those were deleted roots
    expected = df_pl.with_columns(
        ancestor_id=pl.when(
            pl.col("ancestor_id").is_in(roots["id"].implode())
            & (pl.col("id") != pl.col("ancestor_id"))
        )
        .then(oldest)
        .otherwise(pl.col("ancestor_id")),
    ).filter(pl.col("id").is_in(result["id"].implode()))
    assert result.equals(expected)


@pytest.mark.parametrize("seed", range(3))
def test_matches_pandas(seed: int):
    df_pl = _make_forest(100, seed)
    expected = alifestd_consolidate_roots_asexual(
        df_pl.to_pandas(), ignore_topological_sensitivity=True
    )
    result = alifestd_consolidate_roots_polars(df_pl).to_pandas()
    pd.testing.assert_frame_equal(result, expected)


def test_matches_pandas_asset():
    df_pd = alifestd_try_add_ancestor_id_col(
        pd.read_csv(
            f"{assets}/example-standard-toy-asexual-phylogeny-tworoots.csv"
        ),
    ).drop(columns=["ancestor_list"])
    expected = alifestd_consolidate_roots_asexual(
        df_pd, ignore_topological_sensitivity=True
    )
    result = alifestd_consolidate_roots_polars(
        pl.from_pandas(df_pd)
    ).to_pandas()
    assert (result["id"] == result["ancestor_id"]).sum() == 1
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_input_not_mutated():
    df_pl = _make_forest(50, 1)
    original = df_pl.clone()
    alifestd_consolidate_roots_polars(df_pl)
    assert df_pl.equals(original)


@pytest.mark.parametrize("root_ancestor_token", ["none", ""])
@apply_params
def test_ancestor_list_updated(
    apply: typing.Callable, root_ancestor_token: str
):
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_list": ["[none]", "[none]", "[1]", "[2]"],
        },
    )
    result = (
        alifestd_consolidate_roots_polars(
            apply(df_pl), root_ancestor_token=root_ancestor_token
        )
        .lazy()
        .collect()
    )
    assert result["id"].to_list() == [0, 2, 3]
    assert result["ancestor_id"].to_list() == [0, 0, 2]
    assert result["ancestor_list"].to_list() == [
        f"[{root_ancestor_token}]",
        "[0]",
        "[2]",
    ]


def test_ancestor_list_matches_pandas():
    df_pd = pd.read_csv(
        f"{assets}/example-standard-toy-asexual-phylogeny-tworoots.csv"
    )
    expected = alifestd_consolidate_roots_asexual(
        df_pd, ignore_topological_sensitivity=True
    )
    result = alifestd_consolidate_roots_polars(
        pl.from_pandas(df_pd)
    ).to_pandas()
    pd.testing.assert_frame_equal(
        result[expected.columns], expected, check_dtype=False
    )


@pytest.mark.parametrize(
    "origin_time",
    [[None, 0.0, 1.0, 2.0], [float("nan"), 0.0, 1.0, 2.0]],
    ids=["null", "nan"],
)
def test_null_or_nan_root_origin_time_sorts_last(origin_time: list):
    df_pl = pl.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": [0, 1, 0, 1]},
    ).with_columns(origin_time=pl.Series(origin_time, dtype=pl.Float64))
    with pytest.warns(UserWarning, match="origin_time"):
        result = alifestd_consolidate_roots_polars(df_pl)
    # root 1 is oldest, as root 0's time is unknown
    assert result["id"].to_list() == [1, 2, 3]
    assert result["ancestor_id"].to_list() == [1, 1, 1]


def test_null_nonroot_origin_time_ok():
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 1, 0, 1],
            "origin_time": [1.0, 0.0, None, 2.0],
        },
    )
    result = alifestd_consolidate_roots_polars(df_pl)
    assert result["id"].to_list() == [1, 2, 3]
    assert result["ancestor_id"].to_list() == [1, 1, 1]


def test_missing_ancestor_id_raises():
    with pytest.raises(NotImplementedError, match="ancestor_id"):
        alifestd_consolidate_roots_polars(pl.DataFrame({"id": [0, 1]}))


@pytest.mark.parametrize("column", ["num_children", "is_leaf"])
def test_warns_topological_sensitivity(column: str):
    # root 0 is a leaf until consolidation gives it child 2
    df_pl = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 1, 1], column: [0, 1, 0]},
    )
    with pytest.warns(UserWarning, match=column):
        alifestd_consolidate_roots_polars_(df_pl)
