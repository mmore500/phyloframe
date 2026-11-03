import collections
import itertools
import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_find_pair_mrca_id_polars,
    alifestd_mark_leaves_polars,
    alifestd_mark_num_mrca_pairs_asexual,
)
from phyloframe.legacy import (
    alifestd_mark_num_mrca_pairs_polars as alifestd_mark_num_mrca_pairs_polars_,
)
from phyloframe.legacy import (
    alifestd_mark_to_array_polars,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_num_mrca_pairs_polars = enforce_dtype_stability_polars(
    alifestd_mark_num_mrca_pairs_polars_,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_apply = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_df() -> pl.DataFrame:
    """Tree:
    0 (root)
    +-- 1
    |   +-- 4
    |   +-- 5
    +-- 2
    |   +-- 6
    +-- 3
    7 (root)
    +-- 8
    +-- 9
    """
    return pl.DataFrame(
        {"id": [*range(10)], "ancestor_id": [0, 0, 0, 0, 1, 1, 2, 7, 7, 7]},
    )


def _reference_num_mrca_pairs(phylogeny_df: pl.DataFrame) -> list:
    phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)
    leaf_ids = phylogeny_df.filter(pl.col("is_leaf"))["id"].to_list()
    counts = collections.Counter(
        alifestd_find_pair_mrca_id_polars(
            phylogeny_df,
            first,
            second,
            is_topologically_sorted=True,
            has_contiguous_ids=True,
        )
        for first, second in itertools.combinations(leaf_ids, 2)
    )
    return [counts[id_] for id_ in phylogeny_df["id"].to_list()]


@_apply
def test_hand_built_tree(apply: typing.Callable):
    df = _make_df()
    res = alifestd_mark_num_mrca_pairs_polars(apply(df)).lazy().collect()
    assert res["num_mrca_pairs"].to_list() == [5, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    assert res["num_mrca_pairs"].dtype.is_integer()
    assert res["num_mrca_pairs"].to_list() == _reference_num_mrca_pairs(df)


@_apply
def test_mark_as(apply: typing.Callable):
    res = alifestd_mark_num_mrca_pairs_polars(
        apply(_make_df()), mark_as="blueberry"
    )
    names = res.lazy().collect_schema().names()
    assert "blueberry" in names
    assert "num_mrca_pairs" not in names


def test_mark_to_array():
    res = alifestd_mark_to_array_polars(
        _make_df(), alifestd_mark_num_mrca_pairs_polars
    )
    np.testing.assert_array_equal(res, [5, 1, 0, 0, 0, 0, 0, 1, 0, 0])


@_apply
def test_empty(apply: typing.Callable):
    df = pl.DataFrame(
        {"id": [], "ancestor_id": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64},
    )
    res = alifestd_mark_num_mrca_pairs_polars(apply(df)).lazy().collect()
    assert "num_mrca_pairs" in res.columns
    assert res.is_empty()


@_apply
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"id": [0, 2], "ancestor_id": [0, 0]}),  # non-contiguous
        pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 2, 0]}),  # unsorted
        pl.DataFrame(  # sexual
            {"id": [0, 1, 2], "ancestor_list": ["[none]", "[0]", "[0,1]"]},
        ),
    ],
)
def test_unsupported(apply: typing.Callable, df: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_num_mrca_pairs_polars(apply(df))


@pytest.mark.parametrize("seed", range(5))
def test_matches_pair_mrca_random(seed: int):
    rng = np.random.default_rng(seed)
    num_nodes = 60
    df = pl.DataFrame(
        {
            "id": np.arange(num_nodes),
            "ancestor_id": [
                idx
                if idx == 0 or rng.random() < 0.05
                else rng.integers(0, idx)
                for idx in range(num_nodes)
            ],
        },
    )
    res = alifestd_mark_num_mrca_pairs_polars(df)
    assert res["num_mrca_pairs"].to_list() == _reference_num_mrca_pairs(df)


@_apply
@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
    ],
)
def test_matches_pandas(apply: typing.Callable, phylogeny_df: pd.DataFrame):
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df)
    )[["id", "ancestor_id"]]
    expected = alifestd_mark_num_mrca_pairs_asexual(phylogeny_df)
    res = (
        alifestd_mark_num_mrca_pairs_polars(
            apply(pl.from_pandas(phylogeny_df))
        )
        .lazy()
        .collect()
    )
    assert (
        res["num_mrca_pairs"].to_list() == expected["num_mrca_pairs"].tolist()
    )
